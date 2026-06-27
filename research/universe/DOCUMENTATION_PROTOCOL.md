# Documentation Protocol — Preventing Knowledge Loss

> **Purpose**: This document defines the protocol for BHUH research documentation
> to prevent the kind of knowledge loss that occurred previously and to ensure
> all claims are verifiable, honest, and properly caveated.

---

## 1. The Incident (What Happened)

During a previous research session, 14 experimental phases (71-84) were
developed, executed, and committed. However, due to a combination of:
- Infrequent verification that commits landed on `origin/main`
- Long-running sessions without intermediate syncs
- A `git reset` operation that wiped local state

...the work appeared lost from the local working tree. Fortunately, the
commits HAD been pushed to GitHub and were recoverable via `git pull`.

**Lesson**: Always verify `git log origin/main` after each push, not just
rely on the local `git log`.

---

## 2. Verification Checklist (MANDATORY)

After every research session, before considering work "done":

### 2.1 Pre-commit checks
- [ ] All Python files execute without import errors
- [ ] All experiments produce JSON output that matches documented numbers
- [ ] No file uses `from .` relative imports that break when run as script
- [ ] `python -m pytest tests/ -q` shows 159+ passed (production safe)

### 2.2 Post-push checks (CRITICAL)
- [ ] `git log origin/main --oneline | head -5` shows the new commit
- [ ] `git diff origin/main..HEAD --stat` is empty (nothing unpushed)
- [ ] Visit GitHub web UI to confirm files appear in the commit

### 2.3 Documentation consistency
- [ ] Every claim in THEORY.md cites a specific phase number
- [ ] Every phase in EXPERIMENT_LOG.md has matching code in `experiments/`
- [ ] Every "VALIDATED" verdict has a JSON result file backing it
- [ ] Every "❌ INVALID" or "⚠️ PARTIAL" verdict is honestly reported

---

## 3. Theoretical Honesty Rules

### 3.1 Cryptographic claims (CRITICAL)
**FORBIDDEN**: Claiming BHUH is a "one-way function" in the cryptographic
sense. The formal definition requires that NO polynomial-time algorithm
can invert. Gradient descent inverts SIREN in O(P·N·E) — polynomial.

**ALLOWED**: "Computational asymmetry" — the ratio T_inverse / T_genesis
is large (typically 1000-6000×) but FINITE and POLYNOMIAL.

**FORBIDDEN**: Comparing BHUH "security bits" to AES-256 or RSA-2048.
These are fundamentally different primitives:
- AES/RSA: superpolynomial inversion (cryptographic)
- BHUH: polynomial inversion with large constant (computational)

### 3.2 Physics analogies
AdS/CFT, Bekenstein bound, holographic principle are CONCEPTUAL
ANALOGIES, not formal derivations. Every physics connection must include:

> "This is a conceptual analogy. BHUH does not derive from nor prove
> AdS/CFT correspondence. The connection is interpretive."

### 3.3 Theoretical claims
Every theorem/axiom must specify:
- **Status**: Validated (empirically tested) / Theoretical (proven) / Speculative (hypothesis)
- **Scope**: What domain (smooth images? all images? specific sizes?)
- **Limitations**: What cases fail (e.g., "fails for high-frequency fractals")

---

## 4. File Structure

```
research/universe/
├── README.md                    # Overview, status, how to reproduce
├── DOCUMENTATION_PROTOCOL.md    # This file
├── RESEARCH_REPORT.md           # Honest summary of all results
├── SPECULATIVE.md               # Unvalidated hypotheses, clearly marked
├── experiments/
│   ├── phase1_multi_file_siren.py    # Real, working code
│   ├── phase71_quantum_superposition.py
│   └── ... (each phase is one self-contained .py file)
├── docs/
│   ├── THEORY.md                # Mathematical framework with caveats
│   ├── EXPERIMENT_LOG.md        # Real experiment entries
│   ├── BIBLIOGRAPHY.md          # Academic references
│   └── ROADMAP.md               # Future directions
└── papers/
    └── phase1_draft.md          # arXiv-ready paper draft
```

---

## 5. Reproducibility Standard

Every experiment file MUST:
1. Be runnable standalone: `python research/universe/experiments/phaseNN_*.py`
2. Print human-readable summary to stdout
3. Print JSON result to stdout at the end
4. Use only standard scientific Python (numpy, torch, scipy)
5. Complete in <10 minutes on a CPU
6. Have a deterministic seed for reproducibility

---

## 6. Commit Cadence

- **Small commits**: One phase per commit when possible
- **Descriptive messages**: Include verdict (VALIDATED/PARTIAL/INVALID)
- **Verify push**: After every commit, run `git push && git log origin/main --oneline | head -3`
- **Never assume**: If a push reports success but `git log origin/main` doesn't show it, investigate

---

## 7. Audit Trail

Every phase must produce:
- `phaseNN_*.py` — the experiment code
- An entry in `EXPERIMENT_LOG.md` — human-readable summary
- An entry in `RESEARCH_REPORT.md` — results matrix row
- An entry in `THEORY.md` — theoretical context (if applicable)
- An entry in `SPECULATIVE.md` — if any claims are unvalidated

---

*"Documentation is not optional. It is the difference between science and storytelling."*
