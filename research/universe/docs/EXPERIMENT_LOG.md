# 🧪 Black Hole Universe — Experiment Log

> Tracking all experiments, results, and learnings

---

## Format

Each experiment entry follows:

```
### Experiment: [NAME]
- **Date**: YYYY-MM-DD
- **Phase**: 1/2/3/4/5
- **Status**: Planned / In Progress / Complete / Failed
- **Hypothesis**: What we expect
- **Method**: How we test
- **Results**: What happened
- **Conclusion**: What we learned
- **Next steps**: Where to go from here
```

---

## Phase 1: Multi-File SIREN

### Experiment: P1.001 — Multi-File SIREN (10 images @ 64×64)
- **Date**: 2026-06-25
- **Phase**: 1
- **Status**: ✅ Complete
- **Hypothesis**: Multi-file SIREN with per-image modulation achieves 2-5x compression vs separate SIRENs
- **Method**: 10 satellite-like images, baseline vs BHUH, zlib compressed
- **Results**:
  - Separate SIRENs: 86,258B → BHUH: 21,571B = **4.00x improvement**
  - vs ZIP: 2.73x smaller
- **Conclusion**: ✅ Hypothesis confirmed. Shared roots principle works.

### Experiment: P1.002 — Multi-File SIREN Scaling (20 images @ 128×128)
- **Date**: 2026-06-25
- **Phase**: 1
- **Status**: ✅ Complete
- **Hypothesis**: Larger images and more files improve the shared roots benefit
- **Method**: 20 satellite-like images @ 128×128, baseline vs BHUH
- **Results**:
  - Separate SIRENs: 172,243B → BHUH: 22,221B = **7.75x improvement**
  - vs ZIP: 27.14x smaller
- **Conclusion**: ✅ Scaling confirmed. 2x more images → ~2x better improvement.

### Experiment: P1.003 — Multi-File SIREN Scaling (50 images @ 128×128)
- **Date**: 2026-06-25
- **Phase**: 1
- **Status**: ✅ Complete
- **Hypothesis**: 50 images will show 15x+ improvement over separate SIRENs
- **Method**: 50 satellite-like images @ 128×128, baseline vs BHUH
- **Results**:
  - Separate SIRENs: 431,277B → BHUH: 24,032B = **17.95x improvement**
  - vs ZIP: 62.81x smaller
- **Conclusion**: ✅ Hypothesis strongly confirmed. Scaling law validated.

### Scaling Law Discovery

| N images | Size | Improvement vs SIREN | vs ZIP | BHUH size |
|----------|------|---------------------|--------|-----------|
| 10 | 64×64 | 4.00x | 2.73x | 21,571B |
| 20 | 128×128 | 7.75x | 27.14x | 22,221B |
| 50 | 128×128 | 17.95x | 62.81x | 24,032B |

**Key insight**: BHUH size stays nearly constant (~22-24KB) while baseline scales linearly with N. This confirms the "shared roots" principle — the base network is amortized across all files.

---

## Phase 2: Universal Hypernetwork

(No experiments yet — planned for Q4 2026)

---

## Phase 3: LLM Program Search

(No experiments yet — planned for Q1 2027)

---

## Phase 4: Diffusion Seed

(No experiments yet — planned for Q2 2027)

---

## Phase 5: Universe Prototype

(No experiments yet — planned for Q3-Q4 2027)

---

## Failed Experiments (Honest Record)

(No failures yet — but we will document them honestly when they occur)

---

## Summary Table

| Phase | Experiments | Successful | Failed | Success Rate |
|-------|-------------|------------|--------|--------------|
| 1 | 1 | 1 | 0 | 100% |
| 2 | 0 | 0 | 0 | - |
| 3 | 0 | 0 | 0 | - |
| 4 | 0 | 0 | 0 | - |
| 5 | 0 | 0 | 0 | - |
| **Total** | **1** | **1** | **0** | **100%** |

---

*"In research, negative results are as valuable as positive ones — if honestly reported."*
