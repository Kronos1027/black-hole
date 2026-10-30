# BLKH / BHUH Experiment 55 — single INR over contiguous real DEM, fixed geographic coverage

**Date:** 2026-10-27. **Status: reproduced negative result.** This is a *different* representation from Exp54: a single globally coordinated SIREN `f(x,y)` fitted to one 256×256 SRTM-derived source area, downsampled to 64, 128, or 256 pixels. All returned neural reconstructions are decoded from serialized float16+zlib bytes before any quality measurement. No product/deployment claim is warranted.

**Source / origin:** public Mapzen/AWS Skadi HGT `N37W122` and `N37W119`, selected at row/column (1200, 1300), cropped to 256×256 physical samples. Each lower-resolution condition samples **exactly the same geographic footprint**, area-averages the higher-resolution elevation and rounds the result to integer meters. This gives a different pixel count but not a different terrain footprint. Actual source hashes and model bytes are recorded in `results.json`.

**Actual repository run:** https://github.com/Kronos1027/black-hole/actions/runs/40594651403 — **successful**, including self-test, both geographic areas, 2 seeds, 3 sizes, 500 training epochs, hidden width 32, one hidden SIREN layer, neural decode checks, conventional-codec reconstruction, real serialized size checks, and artifact upload. Raw files are in the workflow artifact `exp55-contiguous-dem`.

## Reproduced rate vs error

All sizes are for **one contiguous area**. Physical error is RMSE in meters against the corresponding integer-meter source image.

| Region | Pixels | SIREN seed 42: bytes | SIREN 42: RMSE m | SIREN seed 123: bytes | SIREN 123: RMSE m | WebP at strictly smaller bytes: bytes / RMSE |
|---|---:|---:|---:|---:|---:|---|
| N37W122 | 64×64 | 2,451 | 9.701 | 2,450 | 10.109 | q95: 1,450 / **3.623** |
| N37W122 | 128×128 | 2,448 | 11.518 | 2,443 | 11.869 | q80: 1,436 / **6.794** |
| N37W122 | 256×256 | 2,454 | 12.210 | 2,435 | 12.551 | q40: 2,054 / **7.931** |
| N37W119 | 64×64 | 2,455 | 4.687 | 2,435 | 5.197 | q95: 948 / **2.205** |
| N37W119 | 128×128 | 2,449 | 5.362 | 2,441 | 5.848 | q95: 2,088 / **1.931** |
| N37W119 | 256×256 | 2,454 | 5.577 | 2,442 | 6.069 | q80: 2,106 / **2.679** |

The column labelled WebP is a **single grounded operating point** demonstrating classical dominance for both seeds in that condition, not a cherry-picked claim of neural superiority. In addition, the experiment actually swept WebP qualities 1/10/25/40/60/80/95, delta+zlib quantization 0.5/1/2/5/10/20/50m, and lossless PNG16 and zlib on native signed integer elevations.

**Finding:** **12/12** tested neural configurations were strictly dominated by at least one conventional codec on the serialized-bytes-versus-RMSE plane.

## Context and interpretation

- The **network file size stays near 2.45 KB** as pixel count rises from 64² to 256², as expected for a fixed architecture. However, it does not maintain competitive RMSE compared with WebP at equal/smaller bytes.
- A constant-size recipe alone **does not establish a competitive rate–distortion scaling law**. Exp50's comparison against lossless PNG cannot establish that.
- The model requires training from scratch for each source (roughly **1–21 seconds** in this run depending on image size). WebP has no iterative training phase. Encoding cost needs direct measurement before product claims.
- The experiment does **not** prove all global INR neural codecs are inferior, only these particular architectures, seeds, optimizer, resolutions, durations and tested terrain patches.
- All fixed location/CRS metadata is external and excluded equally across methods. Production must serialize it if stand-alone geospatial files are desired.
- This is source-data compression **in sample**: not held-out geographic generalization.

## Sources and reproduction

```bash
python research/universe/experiment_55_contiguous_dem.py --self-test
python research/universe/experiment_55_contiguous_dem.py --regions N37W122,N37W119 \
  --size-list 64,128,256 --epochs 500 --seeds 42,123 \
  --width 32 --depth 1 --out /tmp/blkh_exp55_real_contiguous
```

On reproducibility: saved workflow results contain the original HGT SHA-256, per-resolution selected-image SHA-256, every saved format's SHA-256, actual byte sizes and post-decode errors. The baseline codec/Pillow implementation and source versions must be pinned before an archival paper release.

## Research decision

**No demo update, no LinkedIn breakthrough claim yet.** The model is approximately fixed-size, but currently not rate–distortion competitive for real elevation. A future useful project may require **hybrid predictive-residual coding**, **multi-scale allocation**, **spatially aligned cross-tile features**, and a target domain with intrinsically smooth parametric textures, instead of relying on a generic coordinate network alone.

**Gate:** repeated real held-out source improvements versus tuned codecs at matched meter RMSE and reasonable decoder/encoder costs. Keep this negative evidence public rather than selectively hiding it.
