# BLKH Exp177 — reversible median and clamped-gradient predictors

2026-11-19. Local experiment, **not CI-reproduced**.

36 configurations = 3 channel transforms (raw, green subtraction modulo 256, chained channel differences) × 6 spatial predictors (left, up, average, Paeth, median gradient, clamped gradient) × 2 entropy backends (zlib-9, LZMA-6). 8 RGB 96×96 images bundled with scikit-image, first four dev, last four held-out. 288 full encode/decode exact RGB round trips. Full container byte counts include 15-byte header, CRC32, and payload. WebP lossless Pillow method=6 verified by decoding and exact RGB comparison.

Selected on development only: green/median/LZMA-6. Held-out four images: BLKH **54,452 B** versus WebP lossless **47,244 B**, BLKH **15.26% larger**. No competitive gain. Experiment ran locally with Python 3.13.5, NumPy 2.3.5, Pillow 12.3.0. Local script and JSON results saved separately in `blkh_exp177` working artifacts, **not yet uploaded to repository**. No GitHub Actions run, no external independent photographic corpus, no AVIF/JXL baseline, no rate-distortion curve, no isolated peak RAM. Do not claim codec superiority or publish a demo/LinkedIn finding.

Next: commit executable script/results and Actions workflow; external Kodak/Tecnick or licensed photo corpus; lossless JXL and AVIF decoding verification; independent process RSS and timing. Preserve split and do not tune on test.