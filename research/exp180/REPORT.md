# BLKH Exp180 — reversible RGB predictors (local run, 2026-11-20)

Local experimental package: `BLKH_exp180_bundle.zip` (script, JSON per-image measurements, README, proposed Actions workflow). **The script and JSON are not yet in this GitHub branch.** Do not mark this result Actions-reproduced.

Tested 4 reversible preprocessing variants (RGB/green-difference × Up/Left) with zlib-9, LZMA-6, bzip2-9, on 8 bundled scikit-image images, four for development and four held out. Total **96 successful encode/decode exact RGB roundtrips**; WebP lossless decoded exactly on all 8.

Development-only winner: `green_up/lzma6`. Held-out full BLKH files: **314,184 bytes**, WebP lossless: **275,040 bytes**; BLKH **14.23% larger**. Process peak shared RSS: 162,872 KiB, not per-codec. Python 3.13.5 on Linux, Pillow WebP available.

Limitations: test data bundled with scikit-image, not external independent photographic dataset; grayscale inputs replicated to RGB; no AVIF/JPEG XL; no lossy RD curves; no Actions execution; no per-codec memory isolation. This is a negative result, not a breakthrough. Do not prepare demo/LinkedIn on this evidence.

Next: commit script and JSON into this research branch; enable reproducibility workflow; collect external test photos with pinned hashes; benchmark AVIF/JXL and full rate-distortion under controlled settings.