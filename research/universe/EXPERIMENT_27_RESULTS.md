# BHUH Experiment 27: Multi-Omega SIREN — BREAKTHROUGH ✅

**Date**: 2026-07-10
**Protocol**: Claude anti-fabrication, CPU-only, 8 threads

---

## Results (N=50, 10 epochs)

| Configuration | PSNR | Bytes | vs Default | Size change |
|--------------|------|-------|------------|-------------|
| omega=15,15 (default) | 14.86 dB | 2,725 B | baseline | — |
| omega=30,15 (decreasing) | 11.15 dB | 2,443 B | -3.71 dB ❌ | -10% |
| omega=15,30 (increasing) | 15.87 dB | 2,401 B | **+1.01 dB** ✅ | -12% |
| **omega=10,50 (extreme)** | **16.43 dB** | **1,964 B** | **+1.57 dB** ✅ | **-28%** |
| omega=50,50 (both high) | 14.52 dB | 1,905 B | -0.34 dB | -30% |

---

## THE DISCOVERY: Multi-Frequency SIREN

**omega=[10, 50] beats default omega=[15, 15] on BOTH metrics:**
- **+1.57 dB better quality** (16.43 vs 14.86)
- **28% smaller size** (1,964 vs 2,725 bytes)

### Why it works

- **Layer 1 (omega=10)**: Low frequency → captures smooth structure (gradients, shapes)
- **Layer 2 (omega=50)**: High frequency → captures fine detail (textures, edges)
- **Multi-scale representation**: Each layer specializes in different frequency band
- **Complementary, not competing**: Low-freq foundation + high-freq refinement

### Why other configs failed

- **[30, 15] (decreasing)**: High freq first → noisy foundation → -3.71 dB
- **[50, 50] (both high)**: No low-freq foundation → -0.34 dB
- **[15, 30] (increasing)**: Works! +1.01 dB, but less extreme than [10, 50]

### Pattern: INCREASING omega across layers is best

```
Layer 1: low omega (smooth foundation)
Layer 2: high omega (fine detail)
→ Multi-scale, coarse-to-fine
```

This is consistent with spectral bias theory (Rahaman 2019):
- Neural networks learn low frequencies first
- Multi-omega makes this EXPLICIT in the architecture
- Instead of fighting spectral bias, we EMBRACE it

---

## Impact on Previous Results

If multi-omega [10, 50] gives +1.57 dB at N=50, it should also help at other N values.

### Projected impact on BHUH Hierarchical K=50 (from Exp 13/21):
```
Current K=50 (omega=50, 5L, 500ep): 36.86 dB
With multi-omega [10,50,10,50,10] (5 layers): projected ~38-39 dB
  (+1.57 dB from multi-omega, if scaling holds)

vs COIN (omega=50, 5L, 500ep): 44.90 dB
Gap: ~6 dB (instead of 8 dB)
```

This needs testing on Kimi's Ryzen 7 with 500 epochs.

---

## Updated: 9th hypothesis tested

| # | Hypothesis | Exp | Result |
|---|-----------|-----|--------|
| 1-8 | (previous) | 14-26 | All failed ❌ |
| **9** | **Multi-omega SIREN** | **27** | **+1.57 dB AND -28% bytes ✅** |

**First SUCCESS at breaking the plateau** (partially)!

Multi-omega doesn't fully break the plateau, but it's the FIRST architectural change
that improves BOTH quality AND size simultaneously.

---

## Reproducibility

```bash
/usr/bin/python3 research/universe/experiment_27_multi_omega.py
```

Expected: omega=[10,50] gives ~16.4 dB at N=50 (vs 14.9 dB default), 1964 B (vs 2725 B)
