# BHUH Experiment 21: Fair Comparison — THE MOST IMPORTANT RESULT

**Date**: 2026-06-28
**Status**: CRITICAL — Corrects previous "breakthrough" claim
**Hardware**: AMD Ryzen 7 5700X, CPU-only

---

## The Result That Changes Everything

| Method | PSNR | Size | Config |
|--------|------|------|--------|
| COIN (optimized) | **44.90 dB** | 27,159 B | omega=50, 5L, 500ep, AC, pruning |
| BHUH K=50 (optimized) | 36.86 dB | **14,886 B** | same + K=50 hierarchical |
| **Ratio** | -8.05 dB | **1.82× smaller** | |

---

## What Happened

### The unfair comparison (Exp 13)
- BHUH used: omega=50, 5 layers, 500 epochs
- COIN used: omega=15, 3 layers, 50 epochs (DEFAULT)
- Result: BHUH "won" +3.1 dB
- **This was WRONG** — apples vs oranges

### The fair comparison (Exp 21)
- BHUH: omega=50, 5 layers, 500 epochs, AC, pruning
- COIN: omega=50, 5 layers, 500 epochs, AC, pruning (SAME)
- Result: COIN wins by 8.05 dB, BHUH is 1.82× smaller
- **This is the truth**

### Why COIN wins on quality
When COIN gets the same optimizations:
- omega=50: +3.5 dB (each image gets full frequency range)
- 5 layers: +11.4 dB (each image gets full depth)
- 500 epochs: +5 dB (each image gets full training)
- COIN's PSNR jumps from 28.10 to 44.90 dB

BHUH can't match this because:
- Shared backbone must compromise across 2+ images
- One-hot conditioning adds noise
- Sharing = regularization (helps generalization, hurts peak quality)

---

## What BHUH Actually Is (honest)

BHUH is a **size-quality trade-off mechanism**, NOT a quality improvement:

```
You want maximum quality? → Use COIN (44.90 dB, 27KB)
You want smaller size?    → Use BHUH K=50 (36.86 dB, 15KB) — 1.82× smaller, -8 dB
You want extreme size?   → Use BHUH K=1 (17 dB, 3KB) — 29.7× smaller, -28 dB
```

The K parameter controls the trade-off. This is STILL a valid contribution —
a tunable R-D curve that COIN doesn't offer. But it's NOT a "breakthrough"
that beats COIN on quality.

---

## Correction to Previous Claims

| Previous Claim | Corrected |
|---------------|-----------|
| "BHUH beats COIN by +3.1 dB" | ❌ WRONG — was unfair comparison |
| "SOTA on CPU" | ❌ WRONG — COIN with same opts is better |
| "BHUH is 1.5× smaller AND better quality" | ❌ WRONG — 1.82× smaller but 8 dB worse |
| "BHUH is 1.82× smaller at -8 dB quality" | ✅ CORRECT (fair comparison) |
| "Hierarchical sharing is a new technique" | ✅ CORRECT (not in COIN/COIN++) |
| "Omega=50 gives +3.5 dB" | ✅ CORRECT (applies to BOTH) |
| "AC saves 61.4% vs zlib" | ✅ CORRECT (applies to BOTH) |

---

## What's Still Valid

1. **Hierarchical sharing is novel** — not in COIN/COIN++ literature
2. **Tunable R-D curve** — K controls trade-off (COIN is a single point)
3. **Scaling law** — BHUH advantage grows with N (size, not quality)
4. **Omega=50 optimization** — applies to ALL SIREN compression
5. **Arithmetic coding** — applies to ALL SIREN compression
6. **CPU-only viability** — both COIN and BHUH run on consumer CPU
7. **L1 pruning at 0.01** — safe, 24.3% reduction

## What's NOT Valid

1. ❌ "BHUH beats COIN on quality" — only when COIN is handicapped
2. ❌ "SOTA result" — COIN with same opts has higher quality
3. ❌ "+3.1 dB advantage" — was unfair comparison

---

## The Real Story for the Paper

BHUH provides a **tunable rate-distortion trade-off** for multi-image SIREN compression:

- COIN: single point (one SIREN per image, best quality)
- BHUH: curve (K controls sharing, trades quality for size)
- At K=50: 1.82× smaller, -8 dB (useful for bandwidth-constrained)
- At K=1: 29.7× smaller, -28 dB (useful for thumbnails/previews)

This is a legitimate contribution — just not the "breakthrough" we thought.

---

## Reproducibility

```bash
python research/universe/experiment_21_combined.py
```

Expected: COIN 44.90 dB / 27KB, BHUH 36.86 dB / 15KB, ratio 1.82×
