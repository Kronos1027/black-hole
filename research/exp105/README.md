# Exp105 — stripe-wise predictor selection by actual compressed byte cost

New research-only hypothesis following negative Exp103/Exp104. Instead of heuristic per-row residual magnitude, independently compress 1, 8, 32 or 128-row stripes with four predictors, choosing the minimum **actual compressed payload length** for each stripe. The full file includes a 14-byte header and a 5-byte predictor/length record for each stripe, plus independent zlib streams. Decoder has no source-image dependency. Full 8-bit grayscale exact roundtrips.

## Executed local exploratory benchmark, 2026-11-16

Python 3.13 / Pillow 12.3 / scikit-image 0.26, Linux x86_64; 128x128 top-left crops of public scikit-image examples. Three image identities used as development (camera, coins, moon), four as exploratory holdout (page, astronaut, chelsea, coffee). These are **not** independently sourced external datasets. 28 encode/decode measurements (7 images × 4 stripe heights), all roundtrips exact. The following are **total encoded bytes**, comparing stripe height 8 versus 128 against Pillow WebP lossless (method 6):

| Split | Image | Exp105 8-row | Exp105 128-row | WebP lossless |
|---|---|---:|---:|---:|
| dev | camera | 4478 | 3977 | 3518 |
| dev | coins | 10045 | 9706 | 8072 |
| dev | moon | 3765 | 3052 | 2874 |
| test | page | 12514 | 12180 | 11200 |
| test | astronaut | 9557 | 9176 | 7054 |
| test | chelsea | 9918 | 9612 | 7692 |
| test | coffee | 10122 | 9711 | 7906 |

**Result:** 8-row segmentation loses to 128-row selection in 7/7 and to WebP lossless in 7/7. Stripe signaling and resetting zlib context outweigh benefits on these crops. Negative hypothesis; do not promote this approach. No AVIF/JXL available for these measurements; no fair lossy R-D curves (this is lossless only). Peak RSS recorded in benchmark JSON, not isolated per encode; times are exploratory. No external independent corpus, no Actions run confirmed.

## Reproduce

```bash
python -m pip install numpy pillow scikit-image
cd research/exp105
python -m unittest -v test_exp105.py
python bench.py > results.json
```

The benchmark uses fixed, named image identities and exact bytes. Avoid tuning on holdout results. Next direction: move away from zlib-based grayscale predictors; explore reversible color transforms or a novel entropy coder with meaningful external datasets, or terminate this branch of research if baselines remain dominant.
