# BLKH Exp184 — reversible palette and tile-palette trial

Local run 2026-11-20, Python/Pillow/NumPy/scikit-image. Hypotheses: global exact RGB palette (<=256 colors), 16x16 tile-local exact palettes, horizontal modular delta, and raw RGB, each with zlib-9 or LZMA-6. Complete BLKH header, payload, and CRC counted. 192x128 RGB; dev: astronaut, coffee, chelsea, rocket; held-out test: immunohistochemistry, hubble_deep_field, colorwheel, cat. Select one variant by dev-only aggregate bytes.

**Selected delta/lzma. Held-out: BLKH 154,308 bytes vs WebP lossless 121,474 bytes; BLKH +27.03%**. Encode/decode and CRC corruption checks passed for evaluated configurations. Shared-process peak RSS 150,648 KiB (not isolated). Negative result; no discovery.

Limitations: skimage bundled imagery is not an independent external dataset; no AVIF/JPEG XL executables, no rate-distortion curves; results were executed locally, NOT in GitHub Actions. The complete local script and JSON were generated as BLKH_exp184. This commit is a report only; reproducibility in Actions pending uploading the runnable files. No demo or LinkedIn warranted.
