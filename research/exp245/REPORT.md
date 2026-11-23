# Exp245 — reversible color + byte/bit layout ablation (2026-11-22)

Status: **negative, exploratory, NOT externally validated**.

## Hypotheses
Test 3 reversible RGB transforms (identity, XOR relative to green, modular delta relative to green) crossed with 3 byte layouts (interleaved pixels, planar channels, bit-plane shuffle) and 3 entropy compressors (zlib level 9, LZMA preset 6, Brotli quality 9). Bit-plane shuffle is a new layout ablation relative to Exp244; not a new codec discovery.

## Protocol
Four bundled scikit-image images resized to RGB 128x128; predeclared development: astronaut, coffee; held-out: chelsea, rocket. 27 configurations × 4 images = 108 verified exact RGB encode/decode roundtrips. Complete 16-byte header + compressed payload counted. Baselines Pillow WebP lossless method 6 and optimized PNG on same pixel arrays. Config chosen ONLY on development totals, evaluated once on test. No random seed used. Python 3.13.5, Pillow 12.3.0, NumPy 2.3.5, Brotli available.

## Results
Selected on development: modular green-delta + planar channels + LZMA. On reserved test, **53,364 bytes**, versus **39,614 WebP lossless** and **51,350 PNG**. Experimental format is **34.71% larger than WebP**, and also larger than PNG. All 108 roundtrips exact. Encoding/decoding durations and tracemalloc Python peak measured individually in local results.json (tracemalloc excludes native allocation). No competitive advance.

## Reproducibility blockers
Executed script and full per-image JSON are local artifacts `/mnt/data/blkh_exp245/bench.py` and `/mnt/data/blkh_exp245/results.json`; **not yet committed**. This GitHub report alone is NOT reproducible in Actions. No independent external corpus, no AVIF/JXL binaries, no lossy rate-distortion curves, no true process peak RSS. A later run must first commit the actual executable and data manifest, then run CI; do not represent this report as a passing benchmark.

## Decision
No scientific claim, public demo announcement, or LinkedIn post warranted. Preserve existing research branches and alerts.
