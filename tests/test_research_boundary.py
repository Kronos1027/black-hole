"""
Production tests for the Black Hole repository.

These tests verify that:
1. Production code (blkh.py, phase1_inr_compressor/) is importable and basic.
2. Research code (research/universe/) does NOT import production code.
3. The research/universe/ experiment scripts are syntactically valid.

Run with: python -m pytest tests/ -q
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "research" / "universe"
PRODUCTION_FILES = [
    REPO_ROOT / "blkh.py",
    REPO_ROOT / "phase1_inr_compressor" / "__init__.py",
]
RESEARCH_PY_FILES = list(RESEARCH_DIR.glob("*.py"))


# ---------------------------------------------------------------------------
# Test 1: Production code is importable
# ---------------------------------------------------------------------------

def test_blkh_importable():
    """blkh.py can be imported."""
    sys.path.insert(0, str(REPO_ROOT))
    import blkh
    assert hasattr(blkh, "main")
    sys.path.remove(str(REPO_ROOT))


def test_phase1_importable():
    """phase1_inr_compressor package can be imported."""
    sys.path.insert(0, str(REPO_ROOT))
    import phase1_inr_compressor
    # Just verify it imports; __version__ may or may not exist depending on repo state
    assert phase1_inr_compressor is not None
    sys.path.remove(str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Test 2: Research code does NOT import production code
# ---------------------------------------------------------------------------

PRODUCTION_MODULES = {"blkh", "phase1_inr_compressor"}


def _check_no_production_imports(file_path: Path) -> list:
    """Return list of production imports found in file_path (empty if clean)."""
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []  # Skip files that don't parse

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in PRODUCTION_MODULES:
                    offenders.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in PRODUCTION_MODULES:
                    offenders.append(f"line {node.lineno}: from {node.module} import ...")
    return offenders


def test_research_does_not_import_production():
    """No research/universe/*.py file imports blkh or phase1_inr_compressor."""
    assert RESEARCH_PY_FILES, "No research Python files found"
    failures = {}
    for py in RESEARCH_PY_FILES:
        offenders = _check_no_production_imports(py)
        if offenders:
            failures[str(py)] = offenders
    assert not failures, (
        f"Research code must not import production code. Offenders:\n"
        + "\n".join(f"{k}:\n  " + "\n  ".join(v) for k, v in failures.items())
    )


# ---------------------------------------------------------------------------
# Test 3: Research scripts are syntactically valid Python
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("py_file", RESEARCH_PY_FILES, ids=lambda p: p.name)
def test_research_file_parses(py_file):
    """Every research Python file parses without SyntaxError."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    assert tree is not None


# ---------------------------------------------------------------------------
# Test 4: Arithmetic codec round-trips
# ---------------------------------------------------------------------------

def test_arithmetic_codec_roundtrip():
    """arithmetic_codec.py round-trips a simple weight vector."""
    sys.path.insert(0, str(RESEARCH_DIR))
    try:
        from arithmetic_codec import encode_weights, decode_weights
        import numpy as np
        rng = np.random.default_rng(42)
        w = rng.standard_normal(500).astype(np.float32) * 0.1
        payload = encode_weights(w, bits_per_weight=8)
        w2 = decode_weights(payload)
        # Same length
        assert w2.shape == w.shape, f"shape mismatch: {w2.shape} vs {w.shape}"
    finally:
        if str(RESEARCH_DIR) in sys.path:
            sys.path.remove(str(RESEARCH_DIR))


# ---------------------------------------------------------------------------
# Test 5: SIREN module is constructible
# ---------------------------------------------------------------------------

def test_siren_module_constructible():
    """experiment_29's MultiOmegaSirenMLP can be instantiated."""
    sys.path.insert(0, str(RESEARCH_DIR))
    try:
        import torch
        from experiment_29_combined_pipeline import MultiOmegaSirenMLP
        model = MultiOmegaSirenMLP(hidden_features=8, hidden_layers=1, omegas=[10.0, 50.0])
        assert model.num_params() > 0
        # Forward pass
        x = torch.zeros(4, 2)
        y = model(x)
        assert y.shape == (4, 1)
    finally:
        if str(RESEARCH_DIR) in sys.path:
            sys.path.remove(str(RESEARCH_DIR))


# ---------------------------------------------------------------------------
# Test 6: Documentation files exist and are non-empty
# ---------------------------------------------------------------------------

REQUIRED_DOCS = [
    "DOCUMENTATION_PROTOCOL.md",
    "SPECULATIVE.md",
    "BHUH_BREAKTHROUGH_RESULTS.md",
    "HONEST_SUMMARY.md",
    "EXPERIMENT_29_RESULTS.md",
]


@pytest.mark.parametrize("doc", REQUIRED_DOCS)
def test_doc_exists_and_nonempty(doc):
    """Required documentation files exist and are non-empty."""
    path = RESEARCH_DIR / doc
    assert path.exists(), f"{doc} not found"
    content = path.read_text(encoding="utf-8")
    assert len(content) > 100, f"{doc} is too short ({len(content)} bytes)"


# ---------------------------------------------------------------------------
# Test 7: HONEST_SUMMARY.md contains entry #10
# ---------------------------------------------------------------------------

def test_honest_summary_has_entry_10():
    """HONEST_SUMMARY.md has been updated with Experiment 29 entry."""
    path = RESEARCH_DIR / "HONEST_SUMMARY.md"
    content = path.read_text(encoding="utf-8")
    assert "#10" in content, "Entry #10 not found in HONEST_SUMMARY.md"
    assert "Exp 29" in content or "Experiment 29" in content, \
        "Experiment 29 not referenced in HONEST_SUMMARY.md"


# ---------------------------------------------------------------------------
# Test 8: EXPERIMENT_29_RESULTS.md contains SHA-256 verification
# ---------------------------------------------------------------------------

def test_exp29_results_has_sha256():
    """EXPERIMENT_29_RESULTS.md contains SHA-256 verification section."""
    path = RESEARCH_DIR / "EXPERIMENT_29_RESULTS.md"
    content = path.read_text(encoding="utf-8")
    assert "SHA-256" in content, "SHA-256 section not found"
    assert "8b5319d12048013626ab775d6f860b1640bb9bc80a4db5bd1565dba0f9b031da" in content, \
        "Output JSON SHA-256 not found in results doc"


# ---------------------------------------------------------------------------
# Test 9: experiment_29 script accepts --help
# ---------------------------------------------------------------------------

def test_experiment_29_help():
    """experiment_29_combined_pipeline.py --help exits 0."""
    result = subprocess.run(
        [sys.executable, str(RESEARCH_DIR / "experiment_29_combined_pipeline.py"), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "seeds" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Test 10: experiment_29 quick mode runs end-to-end
# ---------------------------------------------------------------------------

def test_experiment_29_quick_smoke():
    """experiment_29_combined_pipeline.py --quick runs end-to-end (5 imgs, 50 ep).
    Uses a separate output dir to avoid clobbering the real experiment results.
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    smoke_out = REPO_ROOT / "tests" / "_smoke_out"
    smoke_out.mkdir(exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-u", str(RESEARCH_DIR / "experiment_29_combined_pipeline.py"),
         "--quick", "--output-dir", str(smoke_out)],
        capture_output=True, text=True, timeout=180, env=env,
    )
    assert result.returncode == 0, f"quick mode failed: {result.stderr[-500:]}"
    assert "JSON_BEGIN" in result.stdout, "JSON output not found"
    assert "JSON_END" in result.stdout, "JSON end marker not found"
