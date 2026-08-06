# EXPERIMENT 39 RESULTS — QAT with STE (Quantization-Aware Training)

## Status: COMPLETED — QAT reduces quantization loss by 91%, but byte-parity caveat remains

**Date**: 2026-08-06
**Experiment**: 39 — Neural Weight Manifold Alignment via QAT + STE
**Goal**: Fix the Exp 38 problem (KMeans quantization destroys 13.33 dB) by training
the model AWARE that it will be quantized, using Straight-Through Estimator.

---

## What Changed from Exp 38

| Aspect | Exp 38 (broken) | Exp 39 (QAT) |
|--------|-----------------|--------------|
| Training | Normal float32, quantize AFTER | STE quantization DURING training |
| K config | Global K=50 for all weights | Asymmetric: layer_0=256, hidden=64, output=128 |
| Regularization | None | Quantization-friendly reg (penalize weights far from centroids) |
| Codebook | Fitted once, post-training | Re-fitted every 100 epochs during training |
| PSNR measurement | Pre-quant (bug) | **Post-quant (correct)** |

---

## Raw Results (3 seeds, 256×256, 3 real photos, 500 epochs)

| Method | PSNR (dB) | Size (B) |
|--------|-----------|----------|
| SIREN pre-quant (float32) | 31.95 ± 0.24 | — |
| **SIREN post-quant (QAT, REAL)** | **30.75 ± 0.25** | **6562** |
| PSNR drop from quantization | **1.19 ± 0.01 dB** | — |
| AVIF at 0.1 BPP | 23.62 ± 0.00 | 642 |

---

## QAT Improvement Over Exp 38

| Metric | Exp 38 (no QAT) | Exp 39 (with QAT) | Improvement |
|--------|-----------------|-------------------|-------------|
| PSNR drop from KMeans | 13.33 dB | **1.19 dB** | **12.14 dB (91% reduction)** |
| Post-quant PSNR | 17.29 dB | **30.75 dB** | +13.46 dB |

**QAT reduced the quantization loss by 91%.** The Straight-Through Estimator
allowed the model to learn weights that survive KMeans quantization, and the
quantization-friendly regularization pushed weights toward codebook centroids.

---

## ⚠️ CRITICAL CAVEAT — Not a Byte-Parity Comparison

| Method | PSNR (dB) | Size (B) |
|--------|-----------|----------|
| SIREN+QAT | 30.75 | 6562 |
| AVIF | 23.62 | 642 |

**SIREN+QAT is 10x LARGER than AVIF.** The 7.14 dB PSNR advantage is at
different byte budgets — SIREN uses 6562 B while AVIF uses only 642 B.

This is NOT a valid parity comparison. To determine if SIREN+QAT genuinely
beats AVIF, we need either:
1. Compress AVIF to 6562 B (higher quality) and compare PSNR
2. Compress SIREN to 642 B (lower quality) and compare PSNR
3. Compare both at multiple byte budgets and plot the RD curve

**The positive result is conditional**: QAT dramatically reduces quantization
loss (from 13.33 dB to 1.19 dB), which is a genuine technical achievement.
But whether this makes SIREN competitive with AVIF at matched byte budget
remains untested.

---

## Per-Image Breakdown (seed 42)

| Image | Pre-quant PSNR | Post-quant PSNR | Drop | Size |
|-------|----------------|-----------------|------|------|
| astronaut | 26.62 dB | 25.99 dB | 0.64 dB | 6562 B |
| camera | 27.64 dB | 27.13 dB | 0.51 dB | 6562 B |
| cell | 41.73 dB | 39.32 dB | 2.41 dB | 6562 B |

The `cell` image (highest PSNR) has the largest drop (2.41 dB), consistent
with the "phase shift" theory: high-quality reconstructions are more sensitive
to weight perturbation because the fine details they capture are exactly what
quantization disrupts.

---

## SHA-256 Verification

```
SHA-256: 1f468f3bc5b27e95a5e7425f3476a2463bcb17d61dec7edb6f5438060017fb14
```

---

## What This Means

### Finding 1 — QAT works: 91% reduction in quantization loss

The Straight-Through Estimator + quantization-friendly regularization reduced
the KMeans quantization loss from 13.33 dB (Exp 38) to 1.19 dB. This is a
genuine technical improvement — the model learned to converge to weights
that quantize cleanly.

### Finding 2 — Asymmetric K helps

Using K=256 for the input layer (coordinate encoding) and K=128 for output,
with K=64 for hidden layers, gives more granularity where it matters. The
input layer handles the coordinate-to-frequency mapping and needs finer
quantization; hidden layers can tolerate coarser quantization.

### Finding 3 — The "phase shift" theory is supported

The `cell` image (highest PSNR) has the largest quantization drop (2.41 dB vs
0.51-0.64 dB for lower-PSNR images). This supports the theory that high-
frequency detail is more sensitive to weight perturbation in SIREN's sin
activation. The regularization partially mitigates this but doesn't eliminate it.

### Finding 4 — Byte parity is NOT established

SIREN+QAT at 6562 B beats AVIF at 642 B by 7.14 dB — but this is an unfair
comparison (10x size difference). The real question is: at the SAME byte
budget, does SIREN+QAT still beat AVIF? This requires Exp 40 (byte-parity
test with QAT model).

---

## Recommended Next Step (Exp 40)

Run the Exp 37/38 parity test with the QAT model:
1. Train SIREN with QAT (same as this experiment)
2. Compress JPEG/WebP/AVIF to SIREN's byte budget (6562 B)
3. Compare PSNR at matched size

If SIREN+QAT at 6562 B beats JPEG/WebP/AVIF at 6562 B, that's the first
genuine positive result of the entire program. If not, QAT fixed the
quantization loss but SIREN is still not competitive.

---

## Reproducibility

```bash
cd /home/z/my-project/research/universe
python3.13 experiment_39_qat_ste.py --seeds 42,123,2024 \
    --size 256 --epochs 500 --ste-start-epoch 200
```

Expected runtime: ~30 minutes (3 seeds × ~10 min each).
