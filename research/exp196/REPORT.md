# Exp196 — scan ordering and Gray coding (negative)

2026-11-20. Compared 24 combinations of raster, serpentine, column, Morton traversal; reversible Gray byte mapping on/off; zlib, LZMA, BZ2. Each stream includes a 15-byte header, compressed payload, and CRC32. 8 fixed 96x96 RGB scikit-image samples, 4 development and 4 held-out test. 192 exact encode/decode round trips and 192 corrupted-stream rejection checks executed locally.

Configuration selected on development: `column/False/lzma` (66,048 bytes). Held-out test: **61,864 bytes BLKH vs 48,748 bytes WebP lossless** (+26.91% BLKH). Pillow WebP lossless method=6 quality=100. SHA256 reconstruction checked by array equality; results JSON and executable code packaged locally in `BLKH_exp196_bundle.zip`, **not committed** to repository. Local environment Python 3.13.5, Pillow 12.3.0, NumPy 2.3.5, Linux; shared peak RSS 157,180 KiB.

Limitations: resized built-in images, not external independent photo dataset; no AVIF/JXL; no fair RD curves; RAM not codec-isolated; GitHub Actions not run. Reproduction requires transferring bundled code and workflow into branch, installing pinned dependencies, and executing. No competitive discovery; no demo or LinkedIn publication.
