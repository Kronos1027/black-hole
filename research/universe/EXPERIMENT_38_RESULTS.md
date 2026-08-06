# EXPERIMENT 38 RESULTS — Quantized PSNR Correction (the real number)

## Status: COMPLETED — Exp 37 victory DISAPPEARS. SIREN+entropy LOSES to JPEG/WebP by 10+ dB.

**Date**: 2026-08-06
**Experiment**: 38 — Methodological Correction: Measure PSNR AFTER KMeans Quantization
**Goal**: Correct the measurement bug identified in all experiments 30-37 where
PSNR was measured on the float32 model BEFORE KMeans clustering, while size was
measured AFTER clustering. This experiment measures the REAL post-quantization PSNR.

---

## The Bug

In every experiment from 30 to 37, the PSNR was measured like this:

```python
# Train model in float32
model = train_siren(img)
psnr = evaluate_psnr(model, img)  # ← measured HERE, on float32 model

# Then cluster weights for size calculation
weights = get_model_weights_flat(model)
cluster_result = hierarchical_kmeans_cluster(weights, K=50)
size = entropy_code_indices(cluster_result)  # ← size from quantized model
```

The PSNR comes from the **float32 model** (before quantization).
The size comes from the **KMeans-quantized model** (after quantization).

**Nobody ever reloaded the quantized weights back into the model to measure
the actual PSNR of the compressed representation.** This inflated every PSNR
number since Exp 30 by an unknown amount.

---

## The Correction

This experiment adds the missing step:

```python
# After clustering, reconstruct quantized weights
quantized_weights = codebook[indices]  # ← dequantize

# Reload into model
dequantize_and_reload(model, cluster_result, image_index)

# NOW measure PSNR on the quantized model
psnr_real = evaluate_psnr(model, img)  # ← the REAL number
```

---

## Raw Results (3 seeds, 256×256, 3 real photos, same config as Exp 37)

| Method | PSNR (dB) | Size (B) | Notes |
|--------|-----------|----------|-------|
| SIREN pre-quant (Exp 37 number) | 30.62 ± 0.23 | 4266 | **INFLATED** — measured on float32 model |
| **SIREN post-quant (REAL)** | **17.29 ± 0.41** | **4266** | **The actual PSNR of the compressed representation** |
| PSNR drop from KMeans K=50 | **13.33 ± 0.31 dB** | — | KMeans quantization destroys 13.33 dB |
| JPEG (matched size) | 27.34 ± 0.38 | ~3086 | |
| WebP (matched size) | 28.56 ± 0.00 | ~2838 | |

---

## Corrected Parity Comparison

| Comparison | PSNR diff | Interpretation |
|------------|-----------|----------------|
| SIREN post-quant - JPEG | **-10.04 dB** | JPEG is 10 dB BETTER |
| SIREN post-quant - WebP | **-11.27 dB** | WebP is 11 dB BETTER |
| **Winner** | **JPEG/WebP** | Both crush SIREN+entropy |
| **Conclusion** | **NEGATIVE** | Exp 37 victory DISAPPEARS |

### The correction in numbers

```
Exp 37 reported:  30.62 dB  (pre-quantization, INFLATED)
Exp 38 measured:  17.29 dB  (post-quantization, REAL)
Correction drop:  13.33 dB  (KMeans K=50 destroys this much PSNR)
```

**SIREN+entropy is 10-11 dB WORSE than JPEG and WebP** at matched byte budget
when PSNR is measured correctly. The "victory" reported in Exp 37 was entirely
an artifact of the measurement bug.

---

## Per-Image Breakdown (seed 42)

| Image | Pre-quant PSNR | Post-quant PSNR | Drop |
|-------|----------------|-----------------|------|
| astronaut | 25.32 dB | 13.77 dB | 11.54 dB |
| camera | 26.43 dB | 16.00 dB | 10.42 dB |
| cell | 40.30 dB | 21.04 dB | 19.26 dB |

The `cell` image (which had the highest pre-quant PSNR) suffers the LARGEST
drop (19.26 dB). High-quality reconstructions are more sensitive to weight
quantization because the fine details they capture are exactly what KMeans
destroys.

---

## What This Means

### Finding 1 — KMeans K=50 quantization destroys 13.33 dB of PSNR

The KMeans K=50 clustering, which was supposed to be the "entropy coding"
innovation, actually destroys the SIREN representation. With only K=50
centroids to represent all weight values, the quantization error is enormous.

### Finding 2 — The entire Exp 30-37 PSNR reporting was inflated

Every experiment from 30 to 37 reported PSNR measured on the float32 model
(before KMeans), not on the quantized model (after KMeans). This means:
- Exp 30's "37-49 dB without pruning" was inflated
- Exp 31's pruning sweep results were inflated
- Exp 32-33's "BHUH beats COIN 8/8" was inflated
- Exp 34's "entropy coding = 94% of advantage" was inflated
- Exp 35's "COIN dominates BHUH" was still valid (both sides had same bug)
- Exp 37's "SIREN beats JPEG/WebP" was **completely false**

### Finding 3 — SIREN+entropy is NOT competitive with production codecs

At matched byte budget (~4 KB), JPEG achieves 27.34 dB and WebP achieves
28.56 dB. SIREN+entropy achieves only 17.29 dB — **10-11 dB worse**.
The entropy coding pipeline does NOT beat production codecs on real photos.

### Finding 4 — The BHUH program ends with a purely negative result

After 10 experiments (29-38) of rigorous self-correction:
- Multi-omega architecture: REFUTED (Exp 35)
- Hierarchical sharing: REFUTED (Exp 36)
- Entropy coding "beats JPEG/WebP": REFUTED (Exp 38)
- L1 pruning: REFUTED (Exp 29)

**No component of BHUH provides a real advantage over existing codecs.**
The program ends with a negative conclusion.

---

## SHA-256 Verification

```
SHA-256: b96e67a1d5dff5f5a3aef1e0369a4cce70ad7be887804e8baf90dcd383c3b281
```

---

## Impact on Prior Experiments

| Experiment | Claimed Result | Corrected Status |
|-----------|----------------|------------------|
| Exp 30 | "No-pruning PSNR 37-49 dB" | INFLATED — post-quant would be ~24-36 dB |
| Exp 31 | "Pruning sweep, sweet spot at 0.001" | INFLATED — all PSNRs were pre-quant |
| Exp 32-33 | "BHUH beats COIN 8/8" | INVALID — both sides had same bug, but comparison may still hold if both were equally inflated |
| Exp 34 | "Entropy coding = 94% of advantage" | DIRECTIONALLY CORRECT — entropy coding is still the dominant factor, but the absolute numbers were inflated |
| Exp 35 | "COIN dominates BHUH on both axes" | STILL VALID — both sides measured pre-quant, so the relative comparison holds |
| Exp 36 | "Hierarchical sharing refuted" | STILL VALID — same reasoning as Exp 35 |
| **Exp 37** | **"SIREN beats JPEG/WebP"** | **COMPLETELY FALSE — SUPERSEDED by Exp 38** |

---

## Reproducibility

```bash
cd /home/z/my-project/research/universe
python3.13 experiment_38_quantized_psnr_correction.py --seeds 42,123,2024 \
    --size 256 --epochs 300
```
