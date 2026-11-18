# BLKH Exp150 — DCT zigzag and 8×8 blockwise (negative)

## Novel hypotheses
Following Exp149 (one global low-frequency DCT square), Exp150 tested (a) global zigzag frequency ordering and (b) local 8×8 blockwise DCT with zigzag coefficients. All files counted a 12-byte BLKH header plus the complete zlib-9 compressed coefficient payload.

## Reproducible local experiment
Deterministic bundled scikit-image grayscale 128×128 images. Development: camera, moon, coins; untouched evaluation: page, grass, gravel. Grid: square/zigzag/block × retained coefficient fractions 0.10/0.25/0.50 × quantization steps 8/24 = 18 candidates per image. Rule fixed before evaluation: minimum development bytes with PSNR >=26 dB on **each** development image. Selected block/0.25/24. Lossy decoded reconstruction, not pixel-identical. All 108 candidate bitstreams decoded twice; 324 corruption checks rejected truncated streams, bad magic and invalid mode.

## Holdout actual files
| Image | BLKH bytes / PSNR | WebP bytes / PSNR | AVIF bytes / PSNR | JXL bytes / PSNR |
|---|---|---|---|---|
| page | 2049 / 24.39 | 1622 / 28.92 | 1038 / 25.99 | 2735 / 30.82 |
| grass | 2296 / 25.14 | 2394 / 27.15 | 3417 / 30.83 | 1749 / 26.25 |
| gravel | 2435 / 23.95 | 2826 / 27.08 | 3623 / 30.58 | 2084 / 26.25 |
| **Total** | **6780** | **6842** | **8078** | **6568** |

Baseline: smallest **actually encoded** file in a sparse three-quality-point grid that met/exceeded BLKH PSNR on each image. Baselines often overshoot BLKH quality by several dB. These totals do **not** establish RD superiority. JPEG XL was smaller in aggregate; BLKH failed the 26 dB development threshold on all evaluation images, exposing generalization weakness.

## Environment and limits
Python/scipy/numpy/scikit-image/Pillow and FFmpeg libwebp/libaom-av1/libjxl. Process peak RSS 132700 KiB, **not** codec-specific. FFmpeg startup dominates timings. No external independent dataset. Source script, raw results JSON, tests, and CI workflow generated locally in /mnt/data/exp150; Git local cannot push because DNS lookup for github.com fails. **This GitHub report is not evidence that the scripts or CI were uploaded or Actions executed.**

Next: publish runnable source and machine-readable results, reproduce in Actions, expand external corpus and RD grids, measure isolated memory. No discovery, no demo/LinkedIn.