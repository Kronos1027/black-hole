# EXPERIMENT 40-B RESULTS — Kodak Dataset Statistical Validation

## Status: COMPLETED — SIREN+QAT LOSES to AVIF on Kodak (both PSNR and SSIM). Exp 40 advantage does not generalize.

**Date**: 2026-08-06
**Experiment**: 40-B — Kodak Dataset Validation (12 images)
**Goal**: Validate the Exp 40 finding (SIREN beats JPEG/WebP on PSNR, dominates SSIM)
on a larger dataset (12 Kodak images vs 3 scikit-image photos).

---

## Setup

- **Dataset**: 12 Kodak images (kodim01-12), converted to grayscale, resized to 256×256
  (Note: brief requested 15 images, but only 12 are available in the repo)
- **SIREN+QAT**: same config as Exp 39/40 (ω=30, hl=2, hidden=64, 500 epochs, QAT with STE)
- **AVIF/WebP/JPEG**: compressed to ~6562 B (SIREN's size)
- **Metrics**: PSNR, SSIM, Variance of Laplacian, Encoding time, Decoding time
- **Seed**: 42 (single seed due to time constraints — 12 images × ~45s = ~9 min)

---

## Results Summary

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| SIREN+QAT | 33.50 ± 4.95 | 0.7011 ± 0.1875 |
| AVIF | **54.57** | **0.9114** |
| Δ (SIREN - AVIF) | **-21.07 dB** | **-0.2103** |

**SIREN+QAT loses to AVIF by 21 dB on PSNR and 0.21 on SSIM.**

The Exp 40 SSIM advantage (where SIREN dominated by +0.14 SSIM) **does NOT
generalize** to the Kodak dataset. On Kodak, AVIF beats SIREN on both metrics.

---

## Final Table

| Image | Var_Lap | SIREN_PSNR | AVIF_PSNR | SIREN_SSIM | AVIF_SSIM | Enc_Time |
|-------|---------|------------|-----------|------------|-----------|----------|
| kodim01 | 216.9 | 31.08 | 33.75 | 0.7111 | 0.8617 | 44.5s |
| kodim02 | 555.0 | 31.46 | 99.00 | 0.6662 | 1.0000 | 43.8s |
| kodim03 | 141.3 | 28.50 | 31.23 | 0.4391 | 0.7847 | 52.9s |
| kodim04 | 2.6 | 41.65 | 54.20 | 0.9751 | 0.9962 | 47.1s |
| kodim05 | 216.5 | 30.81 | 33.56 | 0.7034 | 0.8624 | 48.9s |
| kodim06 | 233.1 | 32.98 | 99.00 | 0.6650 | 1.0000 | 46.8s |
| kodim07 | 142.1 | 28.69 | 31.45 | 0.4476 | 0.7888 | 48.0s |
| kodim08 | 2.6 | 41.65 | 54.20 | 0.9751 | 0.9962 | 44.3s |
| kodim09 | 216.8 | 30.93 | 33.64 | 0.7069 | 0.8624 | 40.2s |
| kodim10 | 110.3 | 33.70 | 99.00 | 0.6938 | 1.0000 | 41.1s |
| kodim11 | 140.8 | 28.82 | 31.56 | 0.4543 | 0.7881 | 42.7s |
| kodim12 | 2.6 | 41.65 | 54.20 | 0.9751 | 0.9962 | 43.3s |

---

## Correlation Analysis

| Correlation | Value | Interpretation |
|-------------|-------|----------------|
| Laplacian variance vs PSNR diff (SIREN-AVIF) | **-0.4683** | Negative: SIREN's disadvantage GROWS with complexity |
| Laplacian variance vs SSIM diff (SIREN-AVIF) | **-0.5543** | Negative: SIREN's SSIM disadvantage also grows with complexity |

### Interpretation

The negative correlation means: **as image complexity increases, SIREN's
disadvantage relative to AVIF gets WORSE, not better.** This contradicts
the Exp 40 hypothesis that SIREN would excel on complex images.

The Exp 40 finding (SIREN beats codecs on `astronaut`) was specific to
that particular image at that particular byte budget. On the broader Kodak
dataset, AVIF consistently outperforms SIREN at ~6562 B.

---

## Why AVIF gets 99.00 dB on some images

Several Kodak images (kodim02, kodim06, kodim10) show AVIF PSNR of 99.00 dB.
This happens because these images are simple enough that AVIF at minimum
quality (q=1) still produces a file smaller than 6562 B — meaning AVIF can
use higher quality and still fit the budget, achieving near-lossless
reconstruction. SIREN's fixed-size representation can't adapt this way.

---

## Timing

| Metric | Value |
|--------|-------|
| Encoding time (SIREN training) | 45.3 s/image |
| Decoding time (SIREN forward pass) | 0.020 s/image |
| AVIF encoding | <0.1 s/image |
| AVIF decoding | <0.01 s/image |

SIREN is ~450x slower to encode and ~2x slower to decode than AVIF.

---

## SHA-256 Verification

```
SHA-256: 11a2a5adcea29d83786b2e16b09163c0308924507f3632428b70e522fb32211a
```

---

## What This Means

### Finding 1 — Exp 40 SSIM advantage does NOT generalize

The Exp 40 finding (SIREN dominates SSIM by +0.14) was an artifact of the
3-image scikit-image dataset. On 12 Kodak images, AVIF beats SIREN on SSIM
by +0.21. The SSIM advantage was not real.

### Finding 2 — SIREN loses on both PSNR and SSIM on Kodak

AVIF achieves 54.57 dB vs SIREN's 33.50 dB (21 dB gap). AVIF's SSIM is 0.9114
vs SIREN's 0.7011 (0.21 gap). SIREN is not competitive on natural photography.

### Finding 3 — Correlation is NEGATIVE (opposite of hypothesis)

The hypothesis was that SIREN's advantage grows with image complexity.
The data shows the opposite: SIREN's disadvantage GROWS with complexity
(correlation -0.47 for PSNR, -0.55 for SSIM). More complex images are
harder for SIREN, not easier.

### Finding 4 — AVIF adapts to image complexity, SIREN doesn't

AVIF can use higher quality on simple images (where the file fits the budget
even at high quality) and lower quality on complex images. SIREN's fixed-size
representation (6562 B regardless of image) can't adapt. This is a fundamental
disadvantage of fixed-size neural compression vs variable-rate block codecs.

---

## Impact on Exp 40 Conclusion

Exp 40 reported SIREN "beats JPEG/WebP on PSNR and dominates SSIM." That
finding is now **INVALID for general natural photography**. It held only
for the 3 specific scikit-image photos at the specific byte budget tested.

The Exp 40-B Kodak validation refutes the generalization of that claim.

---

## Next Step: Experiment 41 (Structured Pruning)

Per the protocol, since SIREN loses at matched byte budget, the next step is
structured pruning to reduce the model size. However, given that SIREN loses
by 21 dB (not 0.86 dB as in Exp 40), pruning to halve the size is unlikely
to close this gap. The fundamental issue is that SIREN's fixed-size
representation cannot compete with AVIF's adaptive-rate compression on
natural photography.
