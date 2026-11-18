# Exp137 — multiscale reversible spatial differences (2026-11-18)

## Protocol and result
Ten lossless transforms: raw, vertical/horizontal modular differences at offsets 1, 2, 4, 8, and checkerboard hybrid; all encoded with Python LZMA preset 6. Container counted 10-byte fixed header plus LZMA payload. RGB images resized to 128×128. Development: astronaut, coffee, chelsea, rocket. Evaluation: camera, coins, moon, hubble_deep_field (skimage built-ins). Selection frozen on development before test scoring.

80 encode/decode roundtrips validated pixel-exact. Selected v1 (vertical offset 1). Held-out aggregate BLKH 61,064 bytes versus WebP lossless method 6 48,516 bytes: BLKH **25.86% larger**. Negative result, no competitive discovery. This split is not truly independent of prior experiments and cannot support generalization claims.

Environment: Python 3.13.5, Pillow 12.3.0, NumPy 2.3.5, libwebp 1.6.0. Peak process RSS 146,940 KiB (not codec-isolated). Encoding and decoding times measured per image/configuration in local JSON; local script and full results at /mnt/data/exp137/experiment.py and results.json (not committed). No external AVIF/JPEG XL executable was available, and GitHub Actions reproduction has not been performed. No R-D curves because this experiment is strictly lossless.

## Next steps
Commit executable script and full JSON when local artifact transport is available; run CI; use truly unseen external photographs with original dimensions; run fair WebP/AVIF/JXL benchmarks and isolated RSS/timing. Do not advertise superiority based on this result.
