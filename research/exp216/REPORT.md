# BLKH Exp216 — edge-aware reversible RGB predictors (2026-11-21)

## Protocol and execution
80 local encode/decode round trips: eight scikit-image images resized to 128x128 RGB, five predictors (left/up/Paeth/median-clamped-gradient) x two entropy codecs (zlib9/LZMA6). All decoded RGB arrays matched source exactly. Header 16 bytes counted. Development: astronaut, coffee, chelsea, rocket. Frozen holdout: camera, coins, moon, page. Selection by total dev bytes: median predictor + LZMA6.

## Holdout
BLKH: **40,340 complete bytes**, WebP lossless Pillow method=6: **33,784 bytes**; BLKH **19.41% larger**. Negative result. Source image SHA-256, per-image bytes and encode/decode durations recorded in local results.json; code in local /mnt/data/blkh_exp216/run.py. These local files are NOT included in this commit; GitHub Actions reproduction is NOT established.

## Limitations / next steps
Builtin scikit-image images are not independent external photo datasets; grayscale sources converted to RGB. No AVIF/JPEG XL available; no isolated RAM, no lossy RD curves. Need upload code and results to this branch and add CI before scientific claims. Do not compare lossy vs PNG lossless. No demo or LinkedIn warranted.
