# BLKH Experiment 70 — serialized shared linear image control vs packed-atlas codecs

**Status: CI fully PASSED.** A valid, reproducible **negative** result for *linear* conditional corpus compression on rendered font glyph masks; **no conclusion about nonlinear neural SIREN** should be inferred yet.

**Protocol fixed before experiment:** [BHUH_EXP70_PROTOCOL.md](BHUH_EXP70_PROTOCOL.md)  
**Official GitHub Actions run:** https://github.com/Kronos1027/black-hole/actions/runs/40620199613  
**Code:** `research/conditional/experiment_70_shared_linear_font_atlas.py` and `.github/workflows/exp70.yml`.

## Data, codec and exact evaluation

Rasterize the same 48 uppercase-letter/numeral/punctuation glyphs into antialiased 32×32 grayscale masks from **four already identified real publicly distributed font faces**: DejaVu Sans, DejaVu Serif, Lato Regular, Liberation Sans Regular. Each face supplies one *correlated family* of 48 images; **these are not 192 independent artworks**. The pipeline stores each source font's SHA256 and the rasterized atlas SHA256, without repackaging font files. The algorithm fits and evaluates each family on **the same corpus** (this tests compression, not held-out generalization).

A shared conditional linear coordinate decoder uses one quantized mean image + a shared rank-`k` SVD basis and a distinct int8 latent vector for each glyph, `k=4,8,16,32`. All mean values, shared basis, individual latent coefficients, quantization scales, zlib stream and source metadata are counted. Every PSNR and mean per-glyph SSIM score is computed **only after restoring the saved compressed model bytes**, including all quantization.

Competing coders are *joint atlas* PNG lossless, JPEG2000 lossless, WebP lossless, and lossy WebP quality 1/5/10/20/30/40/50/60/70/80/90/95/100, all receiving the exact same 8×6 48-glyph atlas and serialized manifest. For each model, select the **smallest** conventional atlas with at least the same overall PSNR (MSE) and mean glyph SSIM. All comparison source images and serialized results were independently verified in CI. No train-time float quality is substituted for post-serialization output quality.

## Actual results

| Font | Rank-4 model bytes | Smallest equal/higher-quality atlas | Post-decoding PSNR (dB) | Mean glyph SSIM |
|---|---:|---:|---:|---:|
| DejaVu Sans | 3,085 | **2,807** | 15.42 | 0.540 |
| DejaVu Serif | 3,152 | **3,034** | 15.37 | 0.509 |
| Lato Regular | 3,104 | **2,767** | 15.54 | 0.518 |
| Liberation Sans | 3,113 | **2,807** | 15.75 | 0.532 |

The chosen conventional baseline for rank 4 was WebP quality 1 in every font. These are **low visual-quality** results, not release-worthy image quality.

As rank increased, the quantized shared linear model improved visual quality but rapidly increased its saved basis and latent bytes:

| Rank | Total model bytes across 4 face archives | Total smallest matched conventional atlas bytes | Win count |
|---|---:|---:|---:|
| 4 | 12,454 | **11,415** | 0/4 |
| 8 | 21,016 | **11,415** | 0/4 |
| 16 | 37,918 | **11,415** | 0/4 |
| 32 | 71,144 | **15,343** | 0/4 |

**Overall: 0/16 positive size wins at matched or better decoded PSNR and mean SSIM.** Baseline WebP atlases with a common manifest beat the shared linear representation even at low quality. Most rank-32 reconstructed glyph corpora were still below 30 dB, with average SSIM below 0.90, despite considerably greater byte count.

## Interpretation

1. This is a **real model of shared basis + per-image conditioning**, but **not** a nonlinear neural SIREN, let alone proof about all image compression.
2. A trained SVD basis needs to serialize a 1024-pixel basis pattern for every added rank; amortization over only 48 small images is insufficient in this design.
3. Comparison with individually encoded PNG glyph files alone would be misleading: packing related images into one WebP atlas is a much stronger baseline, and is correctly included here.
4. PSNR and SSIM are basic pixel/structural metrics. Subjective readability and alpha/color support remain unmeasured. The 48 glyphs per font are correlated samples; no p-values or generalization assertions follow from this benchmark.
5. No significant novel compressor was demonstrated; this **negative result is scientifically useful** because it prevents claiming savings from ignoring actual model bytes or atlas packing.

## Next falsifiable experiment

An actually **nonlinear shared conditional coordinate model** with sine activations, learned per-image latents and all trainable weights serialized can test whether a compact network overcomes the linear-control weakness. The precommitted protocol is [BHUH_EXP71_PROTOCOL.md](BHUH_EXP71_PROTOCOL.md). Evaluate **post-float16 reconstructed** pixels against the same atlas baselines and require **PSNR ≥30 dB, mean glyph SSIM ≥0.90** before treating any smaller bytes as potentially useful. This is still only a within-corpus pilot.

**Release and marketing:** HOLD. Keep the protected research branch and draft PR; no main/demo changes or claims of a BLKH breakthrough.
