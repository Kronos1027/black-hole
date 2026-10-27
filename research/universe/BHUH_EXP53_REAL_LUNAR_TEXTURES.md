# BLKH Experiment 53 — real lunar image tiles (exploratory, 2026-10-27)

## Motivation

Experiment 51 found a favorable, highly correlated **synthetic sinusoidal family** where shared SIREN clearly beat six independent SIRENs and some WebP rate–distortion operating points. This experiment tests whether the gain persists on **related but real textured patches** from a natural image.

## Dataset / input provenance

- Source: `skimage.data.moon()`, a grayscale image of the lunar surface, **not SRTM elevation data**. Do not report error in meters.
- Six **nonoverlapping** 64×64 patches from row/column origins `[(160,128),(160,192),(160,256),(224,128),(224,192),(224,256)]`.
- Inputs `float32 = uint8 / 255`; all metrics from fully decoded images; all codec/model bytes counted across the six images.

## Exploratory independent local replication

SIREN width 16, depth 1, omega 30, 300 epochs; shared arm K=1, 4-dimensional per-image latent vectors.

| Method | Seed | Bytes, six images | Mean decoded PSNR |
|---|---:|---:|---:|
| Individual SIRENs | 42 | 4,049 | 35.95 dB |
| Individual SIRENs | 123 | 4,060 | 37.27 dB |
| Shared SIREN K=1 | 42 | 1,023 | 34.80 dB |
| Shared SIREN K=1 | 123 | 1,023 | 33.34 dB |
| WebP q=10 | deterministic | **518** | **37.83 dB** |
| WebP q=25 | deterministic | 610 | 38.89 dB |
| WebP q=40 | deterministic | 708 | 40.01 dB |
| WebP q=60 | deterministic | 856 | 41.00 dB |
| WebP q=80 | deterministic | 1,198 | 42.91 dB |
| WebP q=95 | deterministic | 2,910 | 48.16 dB |

**Finding:** WebP q10 dominates the K=1 sharing configuration on both file size and mean fidelity. One should **not** generalize the synthetic-surface success to spatially related real imagery.

**Official GitHub reproduction confirmed:** The exact repository implementation reproduced every neural and WebP row above on 2026-10-27. The Exp 53 workflow job completed successfully: https://github.com/Kronos1027/black-hole/actions/runs/40454246281 . The workflow saved its results and true serialized model files as artifacts.

The independent local replica was used to generate this table. `experiment_53_real_lunar_tiles.py` on the research branch reproduces the setup using the repository's actual Exp51 `one_run` and `codec_baselines`; GitHub CI is configured to run it and save `results.json` and true `.blkh51` files.

## Next falsifiable research steps

1. **Actual DEM terrain tiles**: include elevation RMSE in meters, maximum absolute error, and georeferenced tile metadata, not just visual PSNR.
2. **Spatial cross-tile conditioning**: condition on tile coordinates (and offset/scale), rather than treating each tile as an unrelated image ID.
3. **Analytic frequency / low-rank residual hybrid**: share predictable smooth structure; store an independently decodable lossless or lossy residual for irregular features.
4. **Matched quality and total corpus storage**: compare independent WebP/AVIF/JPEG XL and multiframe codecs; count codebook, model, residual, headers and decoder costs.
5. **Validation on withheld geographic regions** to avoid selecting only easy locations.

No claim of general image compression superiority is supported by these results.
