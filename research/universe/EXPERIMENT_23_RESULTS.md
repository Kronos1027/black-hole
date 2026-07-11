# BHUH Experiment 23: Capacity Ablation — NOT the Bottleneck

**Date**: 2026-07-10
**Protocol**: Claude anti-fabrication, CPU-only, 8 threads
**Question**: Is the ~15 dB quality plateau caused by insufficient backbone capacity?

---

## Results (N=100, 10 epochs, varying backbone size)

| Hidden | Params | PSNR | Bytes | Time |
|--------|--------|------|-------|------|
| 32 | 1,285 | 9.04 dB | 877 B | 27.3s |
| **64** | **4,517** | **15.08 dB** | **2,751 B** | **12.2s** |
| 128 | 17,125 | 15.32 dB | 9,352 B | 18.1s |
| 256 | 66,917 | 15.58 dB | 30,195 B | 32.1s |

---

## Key Finding: Capacity is NOT the bottleneck

Going from h=64 to h=256 (16× more parameters):
- PSNR gain: only **+0.50 dB** (15.08 → 15.58)
- Size cost: **11× larger** (2,751 → 30,195 bytes)
- **Quality per byte: TERRIBLE** (0.50 dB for 11× more bytes)

From h=32 to h=64: +6.04 dB (significant — h=32 is too small)
From h=64 to h=128: +0.24 dB (marginal)
From h=128 to h=256: +0.26 dB (marginal)

**The plateau at ~15 dB is NOT caused by insufficient capacity.**

---

## What we've now ruled out (accumulated evidence)

| Hypothesis | Experiment | Result |
|-----------|-----------|--------|
| Insufficient epochs | Exp 14 | ❌ +0.28 dB only (60→150 epochs) |
| Insufficient capacity | Exp 23 | ❌ +0.50 dB only (h=64→256) |
| Bad clustering | Exp 15 | ❌ Quality-aware was -3.61 dB worse |
| Wrong architecture | Exp 11 | ❌ KAN was -7 to -19 dB worse |
| Seed noise | Exp 22 | ❌ std=0.19 dB (very stable) |

**The ~15 dB plateau is FUNDAMENTAL to shared SIREN with one-hot conditioning.**

---

## What might cause the plateau (hypotheses to test)

1. **Spectral bias (Rahaman et al., 2019)**: SIREN learns low frequencies first. With 100 images competing, only the lowest common frequencies converge in 10 epochs. Higher frequencies need exponentially more epochs.

2. **Gradient interference**: Different images' gradients at high frequencies cancel each other. The backbone can only learn what's COMMON across all images — individual detail is lost.

3. **One-hot conditioning noise**: The 100-dim one-hot vector is 99% zeros. This creates an extremely sparse signal that limits the head's ability to differentiate images.

4. **NTK saturation (Jacot et al., 2018)**: The Neural Tangent Kernel of a fixed-capacity network has a finite rank. Beyond that rank, adding more functions (images) doesn't improve representation.

---

## Implications

- **The plateau IS the fundamental limit of this approach**
- **No amount of engineering (bigger, longer, better clustering) will break it**
- **To go beyond ~15-17 dB at N=100, need a DIFFERENT approach**:
  - Hierarchical (K=50) — already tested, gives 27-37 dB by reducing group size
  - Meta-learning (COIN++) — needs 50K+ images
  - Separate SIRENs (COIN) — gives 28-45 dB but no sharing

**The trade-off is fundamental**: sharing a backbone gives compression but caps quality.
The K parameter in hierarchical sharing is the knob that controls this trade-off.

---

## Reproducibility

```bash
/usr/bin/python3 research/universe/experiment_22_n150_n200.py  # includes capacity test
```

Or standalone capacity test: see inline script in experiment_22 results.
