# Exp233 — fixed predictor × zlib level ablation (2026-11-22)

Status: **negative competitive result**, exploratory, not externally validated.

Executed 90 BLKH lossless encode/decode roundtrips (5 predictors × 3 zlib levels × 6 images), plus 12 PNG/WebP lossless roundtrips. Every decoded RGB image matched exactly. Predetermined development: astronaut, coffee, chelsea; held-out test: rocket, hubble_deep_field, immunohistochemistry. All cropped to first 96×96 RGB pixels; scikit-image datasets, NOT independent external data. Results count full custom header (14 bytes) plus zlib stream, and complete PNG/WebP files.

Predictors: none, left, up, floor average, MED. zlib levels: 1,6,9. Development chose MED + zlib 9 at 41,910 bytes versus WebP lossless 32,188. Held-out MED+9 47,666 bytes, PNG 47,246, WebP lossless 37,582; **26.83% larger than WebP**. Held-out up+9 48,384, avg+9 49,282, left+9 51,444, none+9 55,202. This is a negative result. Timing was captured per run, but not isolated process RAM. Dataset too small and reused to claim generalization.

Method: Python numpy residuals modulo 256, sequential pixelwise inverse; PIL WebP lossless quality=100 method=6, PNG compress_level=9; zlib compressed raw interleaved RGB residuals. No external AVIF/JXL codecs, no rate-distortion curves, no independent corpus, no CI reproduction. Code and detailed JSON generated in ephemeral run environment at /mnt/data/exp233/bench.py and /mnt/data/exp233/results.json, **not yet committed**; do not interpret this report as full reproducibility.

Next: commit full executable script, external corpus manifest with SHA256, CI, isolate peak RSS, compare AVIF/JXL lossless. No demo or LinkedIn publication justified.
