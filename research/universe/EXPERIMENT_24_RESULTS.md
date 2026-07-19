# BHUH Experiment 24: Training Order — No Effect

**Date**: 2026-07-10
**Hypothesis**: Training images in frequency order (low→high or high→low) changes quality
**Based on**: Rahaman et al., "Spectral Bias of Neural Networks" (2019)

---

## Results (N=100, 10 epochs, 3 orderings)

| Ordering | PSNR | Difference |
|----------|------|------------|
| Random | 15.08 dB | baseline |
| Low freq → High freq | 15.08 dB | +0.00 dB |
| High freq → Low freq | 15.09 dB | +0.01 dB |

**Training order has ZERO effect.** All three orderings give identical results (within 0.01 dB).

---

## Why order doesn't matter

Images are trained **simultaneously** (all images in each epoch), not sequentially. The spectral bias of SIREN affects which frequencies are learned first WITHIN each image, but since all images compete equally in every epoch, the order of presentation is irrelevant.

Spectral bias would matter if we trained **sequentially** (one image at a time, freezing previous). But in simultaneous training, all gradients are summed and applied together.

---

## Complete picture: What we've ruled out

| Hypothesis | Exp | Result | Verdict |
|-----------|-----|--------|---------|
| More epochs | 14 | +0.28 dB (60→150 ep) | ❌ Not the cause |
| More capacity | 23 | +0.50 dB (h=64→256) | ❌ Not the cause |
| Better clustering | 15 | -3.61 dB WORSE | ❌ Not the cause |
| Different architecture | 11 | -7 to -19 dB WORSE | ❌ Not the cause |
| Training order | 24 | +0.01 dB (zero) | ❌ Not the cause |
| Seed noise | 22 | std=0.19 dB | ❌ Not the cause |

**The ~15 dB plateau at N=100 is FUNDAMENTAL to simultaneous shared SIREN training.**

The only way to break it: **reduce N per group** (hierarchical K=50 → 27-37 dB).
