# BLKH Exp258 — adaptive row predictor after reversible green decorrelation (2026-11-23)

Status: NEGATIVE. Local benchmark executed; GitHub Actions reproduction NOT established.

## Protocol
RGB uint8 128x128 from scikit-image (Pillow LANCZOS resize). Fixed development: astronaut, coffee, chelsea, rocket. Fixed holdout: hubble_deep_field, immunohistochemistry, colorwheel, logo. No holdout-based parameter selection. Seven candidates: RGB Sub, RGB Up, reversible green differences + Sub/Up, and row-wise Sub/Up selected by sum of signed residual magnitudes (including RGB row-wise variant); LZMA preset 6, with one zlib level-9 control. Binary BLK8 header is 11 bytes, followed by actual compressed payload and per-row flags when applicable. Decoder verifies exact RGB reconstruction.

## Measured
72 total encode/decode or baseline image runs: 56 BLK8 cycles plus 16 PNG/WebP controls. Selected by minimum development bytes: **green_row_select + LZMA**. On four held-out images, BLK8 **87,464 bytes**, WebP lossless **73,876 bytes**, PNG **101,272 bytes**. BLK8 is **18.39% larger than WebP** (negative result). These are actual serialized byte lengths, not estimated header costs.

## Reproducibility gap
Executed local artifacts: /mnt/data/blkh_exp258/bench.py and /mnt/data/blkh_exp258/results.json (not in GitHub, paths may be ephemeral). No Actions workflow and no externally independent corpus. AVIF/JXL and isolated RAM benchmarks unavailable. No lossy RD curves. Therefore this report is **not** a competitive claim. Next run should prioritize committing script + result JSON and CI before interpreting any apparent gain. Do not publish demo or LinkedIn claims.
