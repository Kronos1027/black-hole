# Exp212 — Adaptive per-image reversible RGB representation (2026-11-21)

Local execution: 8 RGB 128x128 scikit-image samples, four dev / four held-out. Nine configurations: RGB, green subtraction, YCoCg-R modular, crossed with none/up/left causal predictor; zlib level 9. Header 10 bytes included. 72 roundtrips exact, truncated streams rejected. Fixed configuration chosen on development only: green/up.

| Split | WebP lossless | BLKH fixed | BLKH per-image oracle | PNG |
|---|---:|---:|---:|---:|
| Dev | 82,792 | 107,483 | 107,483 | 111,071 |
| Test | 68,006 | 87,565 | 84,686 | 92,873 |

BLKH fixed is 28.76% larger than WebP on held-out images; per-image oracle 24.53% larger. Per-image oracle is an upper-bound selection result, not evidence of competitive improvement. The RGB modular YCoCg-R implementation is reversible under 8-bit arithmetic; predictor uses row/column causal reconstruction. Image SHA256s, per-configuration timings and bytes are in local /mnt/data/blkh_exp212/results.json; script /mnt/data/blkh_exp212/run.py (not committed). Shared process maxRSS, not codec-isolated. Pillow WebP method 6 used; WebP decoder RGB verified.

Limitations: scikit-image images not external independent corpus; no AVIF/JPEG XL binaries; no lossy RD curves; no Actions reproduction. Do not claim discovery or publish demo/LinkedIn. Next: commit executable and results after transferring local artifacts; run Actions, add external photographic dataset and codecs.
