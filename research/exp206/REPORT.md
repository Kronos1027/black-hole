# BLKH Exp206 — grayscale row residuals and bitplanes (2026-11-21)

Local reproducible prototype (script and JSON are in the run's artifact bundle; **not yet uploaded to this branch**).

## Method
Eight built-in scikit-image grayscale images resized to 128x128 with Pillow BOX. Development: camera, moon, coins, page. Held-out test: text, clock, grass, gravel. Fixed split before choosing the winner. 12 candidates: raw, row-wise XOR, row-wise modular subtraction, and bitplane-transposed versions of each, each compressed with zlib level 9 or LZMA preset 6. Custom container includes a 16-byte header. Each candidate underwent encode/decode and pixel-identical reconstruction on all eight images (96 total). WebP lossless encoded with Pillow method 6 and decoded for comparison. Lossless PSNR is infinite for both methods.

## Result
Selected exclusively on development: **row-wise modular subtraction + LZMA6**. Held-out test: **44,088 bytes complete** versus **42,142 bytes WebP lossless**. Ratio **1.046177**, i.e. **4.62% larger**, a negative result. Process-wide peak RSS **137,196 KiB** (not isolated). Encode/decode times recorded per image/candidate in local JSON.

## Limitations / next action
Small built-in dataset; no independent external photographic holdout; no AVIF/JXL; no lossy rate-distortion; no GitHub Actions run; script/results not committed yet. No evidence of a competitive compression breakthrough. Do not promote demo or LinkedIn.
