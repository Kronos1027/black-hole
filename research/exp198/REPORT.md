# Exp198 — codec portfolio and spatial residuals (2026-11-20)

Eight bundled scikit-image images resized to 128×128 RGB; first four development, last four held-out test. Eight configurations: PNG, WebP lossless, left/up/average residuals each with zlib9/LZMA6. Every encoded candidate was decoded and checked for exact RGB equality. Byte counts include BLKH framing. Selection by aggregate development bytes only.

Selected: WebP lossless wrapped in five-byte BLKH header. Development: 86,510 bytes versus raw WebP 86,490 (+0.023%). Held-out: 25,368 bytes versus WebP 25,348 (+0.079%). Peak shared process RSS 148,168 KiB, not isolated. The winner merely wraps WebP and is NOT a novel compressor. No competitive improvement, no demo or LinkedIn.

Limitations: no external photographic dataset, AVIF/JXL comparison, independent memory measurement, or GitHub Actions reproduction. Local executable script/results are in BLKH_exp198_bundle.zip; script has NOT been committed to GitHub, Actions NOT run. Do not interpret this report as independently reproduced.

Next: move executable into branch and CI, test real independent photos, measure per-process resource use, compare lossless WebP/AVIF/JXL at equal settings, then consider lossy rate-distortion curves.
