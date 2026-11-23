# Exp257 — reversible spatial predictors (2026-11-23)

Exploratory local run: 12 deterministic synthetic RGB 128x128 images (seed 257), first 6 development, last 6 holdout. Four predictors (left, up, average, clipped gradient) x zlib level 9 / LZMA preset 6, plus PNG and WebP lossless controls. 96 candidate encode/decode roundtrips plus 24 codec-control roundtrips passed exact RGB comparison. Container BLK7 has a real 14-byte header (magic, width, height, predictor, compressor, payload length); bytes counted from actual serialized blob.

Selection was frozen on development: up + LZMA. Holdout total: BLK7 172140 bytes; PNG 172105; WebP lossless 169272. BLK7 is 1.694% larger than WebP (negative). Python 3.13.5, NumPy 2.3.5, Pillow 12.3.0, Linux x86_64. Process peak RSS 128656 KB, NOT isolated codec memory. Timing recorded individually in local results.json.

Local reproducibility artifacts: /mnt/data/blkh_exp257/bench.py and /mnt/data/blkh_exp257/results.json (not yet committed to repository). This report is NOT a reproducible GitHub Actions result. Limitations: synthetic-only corpus, no external photography, no AVIF/JXL, no RD curves, no isolated RAM. Next: commit script and results; add independent corpus and Actions, verify fair codec settings and lossless reconstruction.

No competitive discovery; no demo or LinkedIn publication warranted. Existing branches and alert automation untouched.
