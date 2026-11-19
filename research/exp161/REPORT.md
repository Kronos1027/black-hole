# Exp161 — Tile-local reversible predictors (2026-11-19)

Hypothesis: select raw/left/up/Paeth predictor per tile by actual zlib-9 compressed length, including selector and length metadata. Tested tiles 8/16/32/64 against fixed-up controls, 8 configurations × 6 images = 48 exact RGB encode/decode roundtrips and 96 CRC-corruption rejections. BLKH stream B161 includes header, compressed blocks, per-block metadata and CRC32. WebP lossless method 6 decoded and pixel-compared for all six images.

Data: six 64×64 RGB resized scikit-image examples; first three development, last three held out. Selection on development: adaptive tile=64. Holdout BLKH 22,561 bytes vs WebP 18,128 bytes (+24.45% BLKH). Negative result; no competitive gain.

Limitations: no independent external photos, AVIF/JXL binaries unavailable, no lossy R-D comparison, no Actions run. Source and JSON are local artifacts from the run and not yet committed. This report alone is NOT sufficient to reproduce; do not claim CI validation or publish performance claims. Previous Exp160 had 22,040 vs WebP 18,128 bytes on a comparable split, but implementations differ, so avoid interpreting the delta as a controlled head-to-head.

Next: commit full experiment.py and results.json, add pinned dependency workflow and SHA256 fixture manifest; run Actions; expand independent photographic holdout; fair matched-quality WebP/AVIF/JXL R-D.
