# Exp179 — reversible channel decorrelation × spatial predictors × byte layout

Date: 2026-11-19 (America/Sao_Paulo). Follow-up to Exp178. New test of **24** combinations: RGB vs green-subtraction, Left/Up/Paeth, interleaved vs planar, zlib-9 vs LZMA-6.

## Reproducible local run (code pending repository integration)
Python 3, numpy, Pillow 12.3.0, scikit-image; 8 built-in images resized to 128×128 RGB. Development: astronaut, coffee, chelsea, rocket. Held-out: camera, coins, moon, page. No random seeds. Selection by sum of full container bytes on development only.

192/192 encode-decode round trips reproduced RGB exactly. 192/192 mutations of CRC header rejected. Each container includes 17-byte header with dimensions, transform, predictor, layout, compressor and CRC. WebP lossless encoded via Pillow (quality 100, method 6), decoded and pixel-verified.

Winner on development: green/up/planar/lzma. Held-out totals: BLKH **38,584 bytes**, WebP lossless **35,870 bytes** (+7.566% BLKH overhead). Peak shared process RSS 149,772 KiB; not per-codec memory. CPU times and per-image SHA256 recorded in local results.json.

**Negative outcome:** no competitive superiority. This experiment does not support a public scientific breakthrough, demo claims, or LinkedIn announcement.

## Outstanding limitations / continuation
The executable exp179.py and results.json are local artifacts and have NOT been uploaded to this branch. External independent photographic datasets, AVIF and JPEG XL lossless, lossy rate-distortion curves, isolated per-codec RAM, independent GitHub Actions reproducibility, and broader image sizes remain unverified. Avoid claiming CI success. Next: upload runnable experiment and pin dependencies, add Actions workflow, test real external datasets and modern codec binaries before any conclusion.
