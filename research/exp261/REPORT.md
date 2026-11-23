# BLKH Exp261 — reversible residual bit/nibble plane experiment (2026-11-23)

## Hypothesis and protocol
Test whether splitting high/low nibbles or 8 bitplanes of reversible spatial prediction residuals improves lossless compression relative to raw or channel-planar layouts. Full 17-byte BL61 header, no estimated overhead. Predictor: none/left/up; layout: raw/planar/nibble/bitplane; backend: zlib/LZMA/bz2. 36 combinations per image, 8 images = 288 BLKH encode/decode verified exact; 16 PNG/WebP lossless control roundtrips. Fixed split before selection: development astronaut, coffee, chelsea, rocket; reserved test hubble_deep_field, immunohistochemistry, colorwheel, cat. Images from bundled scikit-image, RGB resized 96x96 using Pillow LANCZOS. Not an independent external corpus.

## Observations
Chosen solely on development aggregate: **up + raw + LZMA** (65,092 bytes development). On reserved test: **61,000 bytes** BL61 vs **48,982 bytes** WebP lossless and **66,232 bytes** PNG. BL61 is **24.54% larger than WebP**. No competitive advantage; novel bitplane/nibble rearrangement did not win selection. All BL61 and baseline decodes exact. Python 3.13.5, NumPy 2.3.5, Pillow 12.3.0, libwebp 1.6.0, Linux x86_64. Whole-process max RSS 142,332 KB; this is NOT per-codec isolated RAM. No AVIF/JPEG XL, no lossy RD curves, no external photographic holdout, no Actions reproduction.

## Reproducibility and blockers
Local script: `/mnt/data/blkh_exp261/bench.py`; detailed JSON: `/mnt/data/blkh_exp261/results.json`. These are local artifacts, not currently committed to this branch. Thus this report alone does not constitute independently reproducible GitHub Actions evidence. Next run should commit executable script and JSON, add pinned dependency workflow, and run independent Kodak/CLIC samples with codec versions and equivalent settings. Preserve all existing branches.

**Decision:** Negative result. Do not publish demo or LinkedIn claims.
