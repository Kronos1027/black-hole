# EXPERIMENT 41 RESULTS — Structured Pruning + Recovery QAT

## Status: COMPLETED — SIREN (pruned) beats AVIF on 3 scikit-image photos, but Kodak generalization untested

**Date**: 2026-08-06
**Experiment**: 41 — Structured Neuron Pruning + Recovery QAT
**Goal**: Remove 50% of hidden layer neurons, recover with QAT, test if SSIM
advantage holds at reduced size (~4.7 KB vs 6.5 KB).

---

## Setup

- Prune 50% of neurons in hidden layer 1 (lowest L2 norm)
- Recovery: 300 epochs of QAT fine-tuning with K=50, STE, reg_weight=0.01
- 3 scikit-image photos (astronaut, camera, cell), 256×256
- 3 seeds (42, 123, 2024)
- Compare against AVIF at matched ~4703 B

---

## Results (3 seeds, 3 photos, 256×256)

| Method | PSNR (dB) | SSIM | Size (B) |
|--------|-----------|------|----------|
| **SIREN (pruned 50%)** | **31.62 ± 0.22** | **0.8424 ± 0.0040** | **4703** |
| AVIF | 30.52 ± 0.00 | 0.6759 ± 0.0000 | ~4703 |

### Comparison

| Metric | Diff (SIREN - AVIF) | Winner |
|--------|---------------------|--------|
| PSNR | **+1.10 dB** | **SIREN** |
| SSIM | **+0.1664** | **SIREN** |
| Size | 4703 B (was 6562 B, -28.3%) | — |

**SIREN (pruned) beats AVIF on BOTH PSNR and SSIM** at ~4703 B on the
3-image scikit-image dataset.

---

## ⚠️ CRITICAL CAVEAT — Kodak Generalization NOT Tested

Exp 40-B showed that SIREN's advantage on 3 scikit-image photos does NOT
generalize to 12 Kodak images (AVIF won by 21 dB on Kodak). This experiment
uses the SAME 3 scikit-image photos. The positive result here may be
specific to these images, just as Exp 40's was.

**To validate, Exp 41-B would need to run on Kodak.** Given the 21 dB gap
on Kodak (Exp 40-B), pruning 28% of the size is unlikely to close that gap.

---

## Per-Image Breakdown (seed 42)

| Image | SIREN PSNR | SIREN SSIM | AVIF PSNR | AVIF SSIM | Size |
|-------|-----------|------------|-----------|-----------|------|
| astronaut | 26.53 dB | 0.8232 | 5.50 dB | 0.0899 | 4703 B |
| camera | 27.66 dB | 0.7626 | 35.73 dB | 0.9423 | 4703 B |
| cell | 40.91 dB | 0.9431 | 50.34 dB | 0.9956 | 4703 B |

Same pattern as Exp 40: SIREN dominates on `astronaut` (complex image where
AVIF can't compress to 4.7 KB) but loses on `camera` and `cell` (simpler
images where AVIF excels). The aggregate average favors SIREN because
`astronaut`'s extreme advantage outweighs the per-image losses.

---

## Pruning Statistics

| Metric | Value |
|--------|-------|
| Original params | 8577 |
| Pruned params | 6433 |
| Param reduction | 2144 (25%) |
| Neurons kept | 32/64 (50%) |
| Size before pruning | 6562 B |
| Size after pruning | 4703 B |
| Size reduction | 28.3% |

Note: 50% neuron pruning only reduces size by 28% because the codebook and
index overhead don't shrink proportionally with neuron count.

---

## SHA-256 Verification

```
SHA-256: 2fd8d7a88d7ec2904e61842a37bf0fb29a67b1bdb5da04821817785063d5a41e
```

---

## What This Means

### Finding 1 — Structured pruning + QAT preserves quality

Removing 50% of neurons and recovering with QAT maintained PSNR (31.62 vs
30.75 in Exp 40 — actually slightly higher) and SSIM (0.8424 vs 0.8231).
The recovery QAT successfully compensated for the pruned neurons.

### Finding 2 — SIREN beats AVIF on scikit-image at 4703 B

At the reduced byte budget, SIREN beats AVIF by +1.10 dB PSNR and +0.17 SSIM
on the 3 scikit-image photos. The SSIM advantage is even larger than in
Exp 40 (+0.17 vs +0.14).

### Finding 3 — BUT Kodak generalization is the real test

The 3-image scikit-image dataset is not representative. Exp 40-B showed
AVIF beats SIREN by 21 dB on Kodak. This experiment doesn't change that
conclusion — it uses the same unrepresentative 3 images.

### Finding 4 — The "astronaut effect" drives the aggregate

SIREN's aggregate advantage comes entirely from `astronaut`, where AVIF
produces 5.50 dB (garbage) because it can't compress a complex image to
4.7 KB. On `camera` and `cell`, AVIF still beats SIREN by 8-10 dB. The
average is misleading because `astronaut`'s 21 dB SIREN advantage masks
the per-image losses.

---

## Honest Assessment

The structured pruning + QAT technique works — it reduces size by 28%
while maintaining quality. But the fundamental problem remains: SIREN's
fixed-size representation cannot compete with AVIF's adaptive-rate
compression on natural photography (as shown in Exp 40-B on Kodak).

The positive result on 3 scikit-image photos is real but not generalizable.
The program's conclusion remains: **SIREN+QAT is not competitive with
production codecs on natural photography.**
