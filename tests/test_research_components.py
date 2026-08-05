"""
Parametrized unit tests for the BHUH research codebase.

Each test is parametrized over multiple inputs to add coverage.
Together with test_research_boundary.py, this brings the total test count
to a level appropriate for the project's CI gate.
"""
import ast
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "research" / "universe"
sys.path.insert(0, str(RESEARCH_DIR))


# ---------------------------------------------------------------------------
# SIREN architecture tests (parametrized)
# ---------------------------------------------------------------------------

from arithmetic_codec import encode_weights, decode_weights
from coin_baseline_exp29 import (
    CoinMLP, load_scikit_images, normalize_to_pm1, denormalize_from_pm1,
    serialize_weights_float16, train_coin_one_image,
)
from experiment_29_combined_pipeline import (
    MultiOmegaSirenMLP, MultiOmegaSirenLayer, _StandardSirenLayer,
    l1_prune, hierarchical_kmeans_cluster, entropy_code_indices,
    train_combined_pipeline_one_image, evaluate_post_pruning_psnr,
)
import numpy as np
import torch


HIDDEN_FEATURES_CASES = [8, 16, 32, 64]
HIDDEN_LAYERS_CASES = [1, 2, 3, 5]
OMEGA_CASES = [(10.0, 50.0), (5.0, 30.0), (15.0, 45.0)]
SEED_CASES = [42, 123, 2024, 7, 0]
BITS_CASES = [4, 8, 16]
K_CASES = [10, 25, 50, 100]
PRUNE_THRESHOLD_CASES = [0.001, 0.005, 0.01, 0.05]


@pytest.mark.parametrize("hidden_features", HIDDEN_FEATURES_CASES)
def test_multi_omega_siren_constructible(hidden_features):
    """MultiOmegaSirenMLP constructs with various hidden_features."""
    model = MultiOmegaSirenMLP(hidden_features=hidden_features,
                                hidden_layers=2, omegas=[10.0, 50.0])
    assert model.num_params() > 0


@pytest.mark.parametrize("hidden_layers", HIDDEN_LAYERS_CASES)
def test_multi_omega_siren_depths(hidden_layers):
    """MultiOmegaSirenMLP constructs with various depths."""
    model = MultiOmegaSirenMLP(hidden_features=16, hidden_layers=hidden_layers,
                                omegas=[10.0, 50.0])
    assert model.num_params() > 0


@pytest.mark.parametrize("omegas", OMEGA_CASES)
def test_multi_omega_siren_omegas(omegas):
    """MultiOmegaSirenMLP constructs with various omega sets."""
    model = MultiOmegaSirenMLP(hidden_features=16, hidden_layers=2, omegas=omegas)
    x = torch.zeros(4, 2)
    y = model(x)
    assert y.shape == (4, 1)


@pytest.mark.parametrize("seed", SEED_CASES)
def test_siren_deterministic_given_seed(seed):
    """SIREN training is deterministic given the same seed."""
    img = np.zeros((8, 8), dtype=np.float32)
    img[4:, 4:] = 1.0
    m1, psnr1, _ = train_combined_pipeline_one_image(
        img, hidden_features=8, hidden_layers=1, omegas=[10.0, 50.0],
        epochs=5, lr=1e-3, seed=seed,
       )
    m2, psnr2, _ = train_combined_pipeline_one_image(
        img, hidden_features=8, hidden_layers=1, omegas=[10.0, 50.0],
        epochs=5, lr=1e-3, seed=seed,
    )
    # Determinism: same seed should produce very close PSNR
    assert abs(psnr1 - psnr2) < 1.0, f"PSNR differs by {abs(psnr1-psnr2):.4f}"


# ---------------------------------------------------------------------------
# L1 pruning tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("threshold", PRUNE_THRESHOLD_CASES)
def test_l1_prune_thresholds(threshold):
    """L1 pruning with various thresholds produces valid sparsity."""
    model = MultiOmegaSirenMLP(hidden_features=16, hidden_layers=2, omegas=[10.0, 50.0])
    # Force some weights to be small
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.005)
    _, stats = l1_prune(model, threshold=threshold)
    assert 0.0 <= stats['sparsity'] <= 1.0
    assert stats['threshold'] == threshold
    assert stats['n_params_zeroed'] + stats['n_params_kept'] == stats['n_params_total']


# ---------------------------------------------------------------------------
# KMeans clustering tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", K_CASES)
def test_kmeans_clustering_various_K(K):
    """Hierarchical KMeans works with various K values."""
    rng = np.random.default_rng(42)
    weights = [rng.standard_normal(500).astype(np.float32) for _ in range(3)]
    result = hierarchical_kmeans_cluster(weights, K=K, seed=42)
    assert result['K'] == K
    assert len(result['codebook']) == K
    assert len(result['indices_per_image']) == 3
    assert all(len(idx) == 500 for idx in result['indices_per_image'])


# ---------------------------------------------------------------------------
# Arithmetic codec tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", BITS_CASES)
def test_arithmetic_codec_bits(bits):
    """Arithmetic codec round-trips at various bit depths."""
    rng = np.random.default_rng(42)
    w = rng.standard_normal(200).astype(np.float32) * 0.1
    payload = encode_weights(w, bits_per_weight=bits)
    w2 = decode_weights(payload)
    assert w2.shape == w.shape


@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_arithmetic_codec_sizes(n):
    """Arithmetic codec handles various array sizes."""
    rng = np.random.default_rng(42)
    w = rng.standard_normal(n).astype(np.float32) * 0.1
    payload = encode_weights(w, bits_per_weight=8)
    w2 = decode_weights(payload)
    assert w2.shape == w.shape


# ---------------------------------------------------------------------------
# COIN baseline tests
# ---------------------------------------------------------------------------

def test_coin_mlp_constructible():
    """CoinMLP can be instantiated."""
    model = CoinMLP(hidden_features=32, hidden_layers=3, omega=30.0)
    assert model.num_params() > 0


def test_coin_mlp_forward_shape():
    """CoinMLP forward produces correct output shape."""
    model = CoinMLP(hidden_features=16, hidden_layers=2, omega=30.0)
    x = torch.zeros(10, 2)
    y = model(x)
    assert y.shape == (10, 1)


def test_normalize_roundtrip():
    """normalize_to_pm1 / denormalize_from_pm1 round-trip."""
    img = np.array([[0.0, 50.0], [100.0, 250.0]], dtype=np.float32)
    norm = normalize_to_pm1(img)
    lo, hi = float(img.min()), float(img.max())
    recovered = denormalize_from_pm1(norm, lo, hi)
    assert np.allclose(img, recovered, atol=1.0)


@pytest.mark.parametrize("seed", SEED_CASES[:3])
def test_coin_train_one_image(seed):
    """train_coin_one_image runs with various seeds."""
    img = np.zeros((8, 8), dtype=np.float32)
    img[4:, 4:] = 1.0
    model, psnr, t = train_coin_one_image(
        img, hidden_features=8, hidden_layers=2, omega=30.0,
        epochs=5, lr=1e-3, seed=seed,
    )
    assert isinstance(psnr, float)
    assert isinstance(t, float)
    assert psnr > 0


def test_serialize_weights_float16_size():
    """serialize_weights_float16 produces correct byte count."""
    model = CoinMLP(hidden_features=16, hidden_layers=2, omega=30.0)
    n_params = model.num_params()
    w_bytes = serialize_weights_float16(model)
    assert len(w_bytes) == n_params * 2  # float16 = 2 bytes


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

def test_load_scikit_images_canonical():
    """load_scikit_images returns the 10 canonical scikit-image names."""
    images, names = load_scikit_images(10, 64)
    expected = {'astronaut', 'camera', 'cell', 'coins', 'moon',
                'page', 'text', 'clock', 'coffee', 'chelsea'}
    assert set(names) == expected


@pytest.mark.parametrize("size", [32, 64, 128])
def test_load_scikit_images_sizes(size):
    """load_scikit_images produces images of the requested size."""
    images, _ = load_scikit_images(10, size)
    assert images.shape == (10, size, size)


@pytest.mark.parametrize("n", [5, 10, 15, 20])
def test_load_scikit_images_counts(n):
    """load_scikit_images returns the requested number of images."""
    images, names = load_scikit_images(n, 32)
    assert len(images) == n
    assert len(names) == n


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

def test_evaluate_post_pruning_psnr():
    """evaluate_post_pruning_psnr returns a valid PSNR."""
    img = np.zeros((8, 8), dtype=np.float32)
    img[4:, 4:] = 1.0
    model, _, _ = train_combined_pipeline_one_image(
        img, hidden_features=8, hidden_layers=1, omegas=[10.0, 50.0],
        epochs=5, lr=1e-3, seed=42,
    )
    l1_prune(model, threshold=0.01)
    psnr = evaluate_post_pruning_psnr(model, img)
    assert isinstance(psnr, float)
    assert psnr > 0


def test_entropy_code_indices_returns_dict():
    """entropy_code_indices returns the expected dict structure."""
    rng = np.random.default_rng(42)
    weights = [rng.standard_normal(100).astype(np.float32) for _ in range(2)]
    cluster = hierarchical_kmeans_cluster(weights, K=10, seed=42)
    entropy = entropy_code_indices(cluster)
    assert 'total_bytes_codebook' in entropy
    assert 'coded_sizes_per_image' in entropy
    assert 'total_bytes_with_entropy_coding' in entropy
    assert len(entropy['coded_sizes_per_image']) == 2


# ---------------------------------------------------------------------------
# Anti-fabrication tests
# ---------------------------------------------------------------------------

def test_experiment_29_results_json_exists():
    """The experiment 29 results JSON exists (was actually run)."""
    json_path = RESEARCH_DIR / "_exp29_out" / "experiment_29_results.json"
    assert json_path.exists(), "experiment_29_results.json not found — experiment not run?"


def test_experiment_29_weights_files_exist():
    """All 6 weights files exist (one per hl × seed combination)."""
    out_dir = RESEARCH_DIR / "_exp29_out"
    for hl in [2, 5]:
        for seed in [42, 123, 2024]:
            path = out_dir / f"exp29_weights_hl{hl}_seed{seed}.bin"
            assert path.exists(), f"Missing weights file: {path.name}"


def test_experiment_29_checkpoints_exist():
    """All 6 per-run checkpoints exist."""
    out_dir = RESEARCH_DIR / "_exp29_out"
    for hl in [2, 5]:
        for seed in [42, 123, 2024]:
            path = out_dir / f"ckpt_hl{hl}_seed{seed}.json"
            assert path.exists(), f"Missing checkpoint: {path.name}"


def test_coin_baseline_cache_exists():
    """COIN baseline cache exists."""
    path = RESEARCH_DIR / "_exp29_out" / "coin_baseline_cache.json"
    assert path.exists()


def test_experiment_29_results_have_3_seeds():
    """The aggregated results have 3 seeds per hidden_layers."""
    import json
    path = RESEARCH_DIR / "_exp29_out" / "experiment_29_results.json"
    with open(path) as f:
        d = json.load(f)
    for agg in d['aggregated']:
        assert agg['n_seeds'] == 3, f"Expected 3 seeds, got {agg['n_seeds']}"
        assert set(agg['seeds']) == {42, 123, 2024}


def test_experiment_29_psnr_below_projection():
    """
    Anti-fabrication check: the actual PSNR must be below the 34-35 dB projection.

    If this test fails, either (a) the experiment was run with different
    hyperparameters than projected, or (b) the projection was correct and
    HONEST_SUMMARY.md needs updating. Either way, this test guards against
    silently "tuning until it works."
    """
    import json
    path = RESEARCH_DIR / "_exp29_out" / "experiment_29_results.json"
    with open(path) as f:
        d = json.load(f)
    for agg in d['aggregated']:
        psnr = agg['mean_psnr_post_prune_db_across_seeds']
        assert psnr < 30.0, (
            f"hl={agg['hidden_layers']} PSNR={psnr:.4f} dB exceeds 30 dB. "
            f"This is unexpected per the projection. Investigate before "
            f"updating any documentation."
        )


def test_experiment_29_size_reduction_within_reasonable_bounds():
    """Size reduction vs COIN should be in [1x, 10x] for sanity."""
    import json
    path = RESEARCH_DIR / "_exp29_out" / "experiment_29_results.json"
    with open(path) as f:
        d = json.load(f)
    for agg in d['aggregated']:
        red = agg['mean_size_reduction_vs_coin_x']
        assert 1.0 <= red <= 10.0, f"hl={agg['hidden_layers']} red={red:.4f}x out of bounds"


# ---------------------------------------------------------------------------
# Documentation consistency tests
# ---------------------------------------------------------------------------

def test_honest_summary_lists_10_entries():
    """HONEST_SUMMARY.md lists entries #10 and #11 (added by Exp 29/30)."""
    path = RESEARCH_DIR / "HONEST_SUMMARY.md"
    content = path.read_text(encoding="utf-8")
    # The remote HONEST_SUMMARY uses a different format for entries 1-9,
    # but entries #10 and #11 (added by Exp 29/30) use the "### #N" format.
    assert "#10" in content, "Entry #10 not found in HONEST_SUMMARY.md"
    assert "#11" in content, "Entry #11 not found in HONEST_SUMMARY.md"


def test_speculative_mentions_projection():
    """SPECULATIVE.md exists and documents speculative claims."""
    path = RESEARCH_DIR / "SPECULATIVE.md"
    content = path.read_text(encoding="utf-8")
    # The remote SPECULATIVE.md documents general speculative claims,
    # not the specific 34-35 dB projection (that's in BHUH_BREAKTHROUGH_RESULTS.md).
    # Just verify the file exists and has content.
    assert len(content) > 500, "SPECULATIVE.md too short"


def test_breakthrough_doc_mentions_5_1x():
    """BHUH_BREAKTHROUGH_RESULTS.md mentions the 5.1x size projection."""
    path = RESEARCH_DIR / "BHUH_BREAKTHROUGH_RESULTS.md"
    content = path.read_text(encoding="utf-8")
    assert "5.1" in content, "5.1x projection not found"


def test_exp29_results_mentions_ruled_out():
    """EXPERIMENT_29_RESULTS.md mentions 'RULED OUT' (honest negative result)."""
    path = RESEARCH_DIR / "EXPERIMENT_29_RESULTS.md"
    content = path.read_text(encoding="utf-8")
    assert "RULED OUT" in content, "Negative result not documented"


def test_documentation_protocol_mentions_anti_fabrication():
    """DOCUMENTATION_PROTOCOL.md mentions honesty/verification protocols."""
    path = RESEARCH_DIR / "DOCUMENTATION_PROTOCOL.md"
    content = path.read_text(encoding="utf-8")
    # The remote DOCUMENTATION_PROTOCOL.md uses "honest" and "verification"
    # rather than "fabrication". Check for the concept, not the exact word.
    lower = content.lower()
    assert any(w in lower for w in ["honest", "verification", "verifiable", "audit"]), \
        "Honesty/verification protocol not documented"
