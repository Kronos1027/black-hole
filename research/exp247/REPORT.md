# Exp247 — reversible spatial ordering ablation (2026-11-23)

Hypothesis: a lossless pixel permutation (checkerboard parity, Morton Z-order, 8x8 tiles, or column-major) can improve downstream entropy compression versus raster order without changing RGB values.

Protocol: eight scikit-image bundled images resized to 128x128 RGB; locked development split astronaut/camera/colorwheel/coffee, reserved test chelsea/rocket/coins/hubble. Five permutations x two backends (LZMA preset 6, zlib level 9) x eight images = 80 full encode/decode roundtrips, all exact (numpy array equality). Custom B247 container includes 12-byte header and compressed payload. Baselines PNG and WebP lossless method 6 use Pillow and exact RGB roundtrip. Winner selected exclusively on development aggregate size.

Selected: column-major + LZMA, development total 104308 bytes. Reserved test: selected 102360 bytes; WebP lossless 74544 bytes; PNG 107224 bytes. Thus selected is 37.31% larger than WebP, despite being 4.54% smaller than PNG. Negative result, no competitive gain.

Environment: Python 3.13.5; Pillow 12.3.0, libwebp 1.6.0. Process peak RSS 147888 KiB (not isolated codec RAM). Script and full per-image results are local-only at /mnt/data/blkh_exp247/bench.py and results.json, **not yet committed**; this report alone is not reproducible in Actions. No AVIF/JPEG XL; no lossy R-D; bundled images are not an external independent corpus. Future run should integrate script, pin dependencies and dataset hashes, add GitHub Actions, then evaluate an independent holdout with matched lossy quality.

No demo/LinkedIn claims justified. Preserve all existing work.
