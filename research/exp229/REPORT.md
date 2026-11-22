# Exp229 — Reversible color lifting sanity check (2026-11-22)

This run evaluated a new reversible modulo-256 color lifting variant (RCT) against RGB and green-relative color differences, with no prediction or MED prediction, followed by zlib level 9 or LZMA preset 6. Six scikit-image source photographs were resized to 96x96 RGB. Three were development and three were held out within this run. The test images are **not independent of previous BLKH experiments**.

**Exact round-trip:** all 72 experimental encode/decode cycles and 12 PNG/WebP baseline cycles passed byte-for-byte RGB reconstruction. Container sizes include a 9-byte experimental header. Codec controls used complete files.

**Held-out test totals, three images (bytes):**

| Method | Bytes |
|---|---:|
| WebP lossless (Pillow quality 100, method 6) | 40,872 |
| Green-relative + MED + LZMA | 46,031 |
| New reversible lifting RCT + MED + LZMA | 47,263 |
| RGB + MED + LZMA | 49,923 |
| PNG | 52,457 |
| Green-relative + MED + zlib | 49,196 |
| RCT + MED + zlib | 50,125 |

RCT is 2.68% larger than green-relative + MED + LZMA. Green-relative remains 12.62% larger than WebP lossless. **Negative result, not a competitive breakthrough.**

Environment: Python 3.13.5, Pillow 12.3.0, Linux 6.18.57 x86_64. No random seeds (deterministic fixed corpus). Experimental format is a prototype, not a stable self-describing format: algorithm parameters are supplied externally. Timing measurements are in the local JSON artifact, not yet uploaded. RAM was not isolated. No JPEG XL or AVIF equivalent comparison or rate-distortion sweep was completed. No GitHub Actions reproduction.

**Follow-up:** persist the full script and JSON in the repository; implement a self-describing format and independent test corpus; add fair JPEG XL/AVIF lossless and lossy rate-distortion comparisons, process-isolated memory profiling, and Actions. Do not interpret PNG wins as codec superiority. No demo or LinkedIn release warranted.
