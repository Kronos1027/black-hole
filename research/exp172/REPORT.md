# BLKH Exp172 — reversible per-row filter selection (2026-11-19)

New experiment following Exp171, no prior experiment modified.

Hypotheses: (1) select row filter by sum of absolute signed residuals; (2) select by row entropy; (3) fixed Paeth vs fixed Up. Modes RGB and green-subtracted, compressors zlib-9 and LZMA-6. 16 configurations x 8 images = 128 exact RGB roundtrips and 128 rejected CRC corruptions. Complete BK72 header, filter selectors, compressed payload, CRC32 counted.

Dataset: 8 bundled skimage sample images resized to 64x64 RGB. Development (4): astronaut, coffee, chelsea, rocket. Held-out (4): hubble_deep_field, immunohistochemistry, colorwheel, brick. Dataset is NOT externally independent. Selection on development ONLY: green subtraction + minimum absolute signed row residual + LZMA-6.

Development: BLKH 28,044 bytes vs lossless WebP 25,076 (+11.84%); PNG 32,320.
Held-out: BLKH 22,896 bytes vs lossless WebP 19,170 (+19.44%); PNG 26,968.

Python 3.13.5, Pillow 12.3.0, NumPy 2.3.5, Linux 6.18.57. Shared process peak RSS 139,508 KiB (NOT isolated per codec). Encode/decode durations recorded per image/config in local JSON.

Negative result. No competitive compression advantage demonstrated. Missing: external independent photos, lossless AVIF/JXL, rate-distortion curves, per-codec isolated memory and GitHub Actions reproduction. Do NOT claim scientific breakthrough or publish demo/LinkedIn.

The full runnable script and JSON are in the automation execution artifact BLKH_exp172_bundle.zip; **not yet committed**. Next run: incorporate script and JSON into this branch and add CI before claiming reproducibility.