# BLKH / BHUH Experiment 54 — real SRTM-derived DEM, meter-error rate-distortion (2026-10-27)

**Status:** fully executed in GitHub Actions; negative result for the specific neural architectures tested. **Do not claim a new compression breakthrough or update the public demo on this basis.**

## Data and reproducibility

- Source: public Mapzen / AWS Terrain Tiles Skadi HGT files `N37W122` and `N37W119` from `https://s3.amazonaws.com/elevation-tiles-prod/skadi/N37/{tile}.hgt.gz`. These are SRTM-derived elevation rasters, **not independently validated USGS 3DEP measurements**. Original elevations in integer meters.
- Within each HGT grid, six fixed nonoverlapping adjacent 64×64 tiles at `(row,col) = (1200,1300), (1200,1364), (1200,1428), (1264,1300), (1264,1364), (1264,1428)`.
- NoData (-32768) causes rejection rather than silent fill; two chosen regions passed.
- Source compressed/uncompressed SHA-256 and precise input SHA-256 are retained in the workflow artifact `results.json`. Full serialized output for every neural and conventional codec is also preserved as an artifact.
- **Original script:** `research/universe/experiment_54_srtm_real_meters.py`.
- **Successful full workflow:** https://github.com/Kronos1027/black-hole/actions/runs/40593801140 ; successful round-trip checks plus full two-region, two-seed runs (300 epochs, width 16).
- An initial run failed on Pillow 13 after `Image.fromarray(...,mode='I;16')` API change; corrected to infer the mode automatically. The corrected complete workflow passed. Failure is retained in GitHub Actions history.

Every PSNR and RMSE measure is computed from independently *decoded* bytes. Neural architecture from Exp 51 uses global min/max scaling encoded as float32 plus the actual float16+zlib recipe; size includes these 8 bytes. Baselines use WebP grayscale mapped to the same global min/max, 16-bit PNG, zlib on raw signed integers, or a horizontal predictive delta codec at seven meter quantization steps. All modes count byte-stream headers/lengths. For all methods the fixed geographic location/CRS is treated as external dataset metadata and excluded from byte counts. This must change before production.

## Scientific results — six real elevation patches per region

| Region | Method | Seed / setting | Serialized bytes | RMSE meters |
|---|---|---|---:|---:|
| N37W122 | Individual SIREN | 42 | 4,073 | 11.608 |
| N37W122 | Individual SIREN | 123 | 4,084 | 12.491 |
| N37W122 | Shared conditional SIREN, K=1 | 42 | 1,031 | 15.653 |
| N37W122 | Shared conditional SIREN, K=1 | 123 | 1,031 | 16.418 |
| N37W122 | WebP grayscale | quality 1 | 750 | 11.195 |
| N37W122 | WebP grayscale | quality 10 | 1,028 | 9.165 |
| N37W122 | WebP grayscale | quality 25 | 1,316 | 7.334 |
| N37W122 | WebP grayscale | quality 40 | 1,502 | 6.064 |
| N37W119 | Individual SIREN | 42 | 4,071 | 5.873 |
| N37W119 | Individual SIREN | 123 | 4,072 | 5.971 |
| N37W119 | Shared conditional SIREN, K=1 | 42 | 1,031 | 7.474 |
| N37W119 | Shared conditional SIREN, K=1 | 123 | 1,031 | 9.480 |
| N37W119 | WebP grayscale | quality 1 | 596 | 5.930 |
| N37W119 | WebP grayscale | quality 10 | 732 | 4.615 |
| N37W119 | WebP grayscale | quality 25 | 876 | 3.945 |
| N37W119 | WebP grayscale | quality 40 | 1,054 | 3.382 |

All sizes are total bytes for the **entire six-tile corpus**, not per image. The neural rows are lossily reconstructed; a lossless baseline must **not** be used as the sole comparator.

Classical predictive + zlib and lossless PNG (native elevation meters):

| Region | Method | Size, bytes | RMSE |
|---|---|---:|---:|
| N37W122 | Delta+zlib with 10m quantization | 7,462 | 2.904m |
| N37W122 | Delta+zlib with 20m quantization | 5,489 | 5.816m |
| N37W122 | PNG16 lossless | 18,919 | 0m |
| N37W122 | Delta+zlib lossless 1m | 18,495 | 0m |
| N37W119 | Delta+zlib with 10m quantization | 4,367 | 2.794m |
| N37W119 | Delta+zlib with 20m quantization | 3,166 | 5.903m |
| N37W119 | PNG16 lossless | 13,284 | 0m |
| N37W119 | Delta+zlib lossless 1m | 12,746 | 0m |

Both geographic test regions yielded **4 of 4 neural configurations strictly dominated** by a conventional codec, defined as **at least one codec with no more bytes and no worse RMSE, with one axis strictly better**. Thus all 8 neural trials are dominated. Examples:

- N37W122 shared SIREN 1,031B/15.65m versus WebP q10 1,028B/9.17m.
- N37W119 shared SIREN 1,031B/7.47m versus WebP q25 876B/3.94m.

A source with 300 training epochs and a width=16 SIREN is not exhaustive hyperparameter search, nor enough to reject all implicit neural representation techniques.

## Conclusions

1. The favorable Exp 51 result for correlated sinusoidal surfaces is **not sufficient** to demonstrate value on natural DEMs.
2. The earlier Exp 50 claim of `98.63x vs PNG` is **not a valid matched-fidelity comparison**, because PNG is lossless while SIREN is lossy. Do not present that ratio alone as evidence of general codec superiority.
3. Physical RMSE matters: a compact elevation representation that misses terrain by 5–16 meters may be unacceptable for surveying, navigation, hydrology or safety-critical elevation use.
4. There is presently **no justification for a public demo claiming superior real DEM compression**.

## Gate for demo / LinkedIn announcement

Do not publicize BLKH as superior on real DEM until ALL are satisfied:

- Multiple real, independent geographical regions, including a genuinely held-out region;
- At least 3 training seeds with uncertainty bars;
- Competitive against tuned WebP and specialized elevation predictors at matched RMSE, plus AVIF/JPEG XL when feasible;
- A useful margin (for example **>=20% lower bytes at the same or lower meter RMSE**) replicated across several regions, not an isolated cherry-picked operating point;
- Saved decodable artifacts with SHA256, accounted metadata and measured training/inference time;
- GPU/CPU decoder available as a reproducible end-user workflow, no hidden cloud/API requirements.

Next avenue: test single global-coordinate INR over *large contiguous DEM tiles*, rather than treating each 64×64 tile as independent. Also test a quantized geometry-aware residual path. All new designs remain hypotheses until measured.
