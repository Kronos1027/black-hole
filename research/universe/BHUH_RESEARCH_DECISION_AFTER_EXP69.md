# BLKH research decision: redirect after Exp68–69 (2026-10-28 UTC)

**Scope:** decision memo for the exploratory research branch, not a product roadmap commitment, paper acceptance or new result.

## What has been experimentally established

- **Exp36 audit:** prior hierarchical SIREN evaluation had an important baseline flaw: default K=50 with 30 images results in `min(K,n_images)=30` clusters and no meaningful sharing. Several earlier measured size-vs-PSNR values used prequantization quality. These tests cannot establish a neural-sharing breakthrough.
- **Exp50 terrain scaling:** a tiny neural recipe versus a full *lossless* PNG is not matched-fidelity evidence. Real elevation-meter RMSE and actual decoded quality are mandatory.
- **Exp54–55:** generic SIREN neural terrain recipes were rate–distortion-dominated by conventional codecs across the tested real DEM patches and two-area contiguous representations.
- **Exp56–62:** a practical reversible predictive Brotli codec was built with true tile indexing, signed-meter lossless decoding, horizontal georeferencing, and bounded decoded-block caching. It demonstrated size advantages over selected PNG and fixed-32px GeoTIFF settings, but **JPEG2000 and GDAL read path optimizations reversed several early broad claims**.
- **Exp63–65:** a fixed 16-bit residual backend and per-block hybrid selection gave a reproducible size/decoding-speed tradeoff on already studied sources. This is useful codec engineering, not a new compression principle.
- **Exp66–67:** a frozen 20% hybrid beat *separately indexed* 32px JPEG2000 on both bytes and uncached Python decode latency in 3/4 genuinely unseen geographic HGT grids. However **one continuous whole-JP2 codestream**—decoded by GDAL directly as real 32px spatial windows—was about 41% smaller overall and was at least as fast in mean ROI latency on 3/4 sources in the same run. The original 3/4 claim must always be paired with this correction.
- **Exp68:** tested 3 stripe heights ×3 compressors ×4 already examined regions, 36 real reconstructed archives. **All 36 stripes were dominated in full-file bytes AND the measured mean 32px-window latency by whole-JP2**. Merely combining tiles into wide strips does not produce a new winning access/storage frontier.
- **Exp69:** enlarged two already examined real HGT grids to 256²/512²/1024² pixels. All **6/6** source/resolution comparisons were smaller AND faster on window means with full-JP2 versus equally georeferenced seekable BH69 (same frozen block selector). At 1024², whole-JP2 saved about 45% bytes. No evidence that JP2 window access slowed with these map sizes.

**Decision:** Do not continue blind local parameter sweeps of 32px independently compressed DEM blocks or plain horizontal striping on the same 2–6 geographical regions and mistake small-file wins for a new result. The original BLKH goal of compact, meaningful *shared/continuous structure* is not proven by conventional modular delta + Brotli codecs.

## Experimental uncertainties that remain

1. Results cover SRTM-derived small/mid-size DEMs with horizontal EPSG:4326 and signed elevation meters, not comprehensive GIS vertical datum/NoData policies, LIDAR clouds, 3DEP or all terrain types.
2. No real cold-storage cache-normalized large-map ROI benchmark across comparable optimized native implementations; J2K and GDAL have internal caches. JavaScript/GUI or field GIS workloads may change the trade-off.
3. Many reported results are *correlated test configurations* of few regions, not thousands of independent input images.
4. Even where BLKH demonstrates a useful domain feature, the techniques are existing and cannot be claimed as a worldwide theoretical breakthrough.

## A better, falsifiable next research track

**Experiment 70 proposal (NOT executed by this memo):** evaluate a *conditional shared implicit neural representation* over real correlated **asset families** (e.g. sprites, glyph families, technical textures, repeatable multi-view icon sets), rather than treating unrelated natural DEM tiles as 64 independent images.

Freeze the scientific protocol before viewing target results:

- Dataset: **real independent source families**, distinct origin/artist/assets with documented permits/licenses; source images and SHA256 fixed. Do not treat flipped or rotated copies as independent examples. Use at least four separate source families, and hold out one family for external confirmation.
- Model: a **single shared coordinate network plus actual per-image conditioning** (latent embedding, FiLM or distinct small residual/head), unlike the defective Exp36 shared unconditioned backbone. Count the shared backbone **ONCE**, and all latents, headers, codebooks, decoders and indices honestly across the corpus.
- Fidelity: compare *post-serialization, post-quantization decoded* image to original image using PSNR, SSIM, and at least one perceptual criterion; preserve actual alpha channels and color profiles when present. Never measure raw, unquantized model quality alongside quantized file size.
- Baselines: AVIF, JPEG XL, WebP and/or JPEG2000 when supported, **including batch/atlas packing for correlated small assets**, with matched perceptual quality and decode/encode-time measurements. Normal PNG only as a supplementary lossless reference.
- Decision gate: repeatedly beat the *strongest actually matched* conventional baseline in real serialized bytes **at the same or higher decoded image fidelity**, with an independently verified meaningful margin (e.g. ≥15% size savings) on at least three independent families, and a useful editing/scaling capability or acceptable decoder cost. This is an intentionally stringent engineering release gate, not a significance-test p-value.
- Test one external held-out family only **after all hyperparameters, data-selection rules and release thresholds are frozen**. A negative result must be published unmodified and should cause a pivot rather than retrospective tuning.
- Keep deployment/demo claims separate from scientific finding. A good user feature (e.g. scale-independent image manipulation) may still be worth shipping without claiming codec-world leadership, subject to owner approval.

## Current action status

- Scientific records for Exp68 and Exp69, scripts, CI workflows and compressed decoded artifacts are committed to `research/exp51-fair-shared-siren`.
- Draft PR #1 remains open. **Do not auto-merge or modify `main` or the public demo.**
- No sufficiently supported universal compression breakthrough has yet emerged. A careful LinkedIn post about reproducible research and negative benchmarks is possible, but the requested **major breakthrough/demonstration update remains on hold**.
