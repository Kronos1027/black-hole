# Exp251 — Exact RGB palette serialization (2026-11-23)

Follow-up to Exp250: test lossless reconstruction of **original** RGB, not merely quantized inputs. Tested 3 index layouts (raster, column, horizontal modular delta) × 2 compressors (LZMA preset 6, zlib level 9) on 12 deterministic 128×128 synthetic images (seed 251). Six development images and six held-out images; 72 encode/decode round trips, all byte-exact. PNG (level 9) and WebP lossless (quality 100, method 6) also decoded and checked against source.

Chosen on development: raster + LZMA (172691 bytes across six images; column + LZMA 172783). Held-out: **180799 bytes BLKH versus 119874 bytes WebP**, ratio **1.50824**, or **50.82% larger**. Negative result. Full container header (12 bytes), RGB palette and compressed index stream counted. No claim of improvement versus modern codecs.

Environment: Linux 6.18.57 x86_64, Python 3.13.5, Pillow 12.3.0, NumPy 2.3.5. Process max RSS 128036 KiB, not isolated per codec. Per-image timings/hashes in local results.json.

Reproduction artifact generated in execution environment: /mnt/data/blkh_exp251/bench.py and results.json. These files are not committed; independent Actions reproduction **not established**. Important limitation: synthetic corpus only, no independent natural-image dataset, no AVIF/JPEG XL, no rate-distortion curves (lossless only). Next: commit executable and pinned dependencies; CI, external corpora and comparable AVIF/JXL baselines. No demo or LinkedIn publication.
