# Exp118 — locked family selector (2026-11-17)

Independent local experiment designed to correct Exp117's per-test-image oracle bias. The selector minimizes aggregate encoded bytes over **dev seeds 101–103** separately for each synthetic family, then freezes the predictor and entropy backend for **test seeds 9101–9103**. Five predictors (raw, xor-left, subtract-left, subtract-up, modular Paeth) × two entropy backends (zlib-9, LZMA-6). Each BLKH file contains an 11-byte header, 48-byte RGB palette and compressed payload. Exact BLKH decode and independently decoded Pillow WebP lossless RGB verified for each of 15 test cases.

| Held-out family | Locked choice | BLKH mean bytes | WebP mean bytes | BLKH wins |
|---|---|---:|---:|---:|
| 16-color random | raw/LZMA | 8680.3 | 8342.7 | 0/3 |
| stripes | Paeth/zlib | 110.3 | 174.7 | 3/3 |
| diagonal | raw/zlib | 196.7 | 287.3 | 3/3 |
| blocks | raw/LZMA | 435.0 | 498.7 | 3/3 |
| repeated rows | subtract-up/zlib | 182.0 | 212.7 | 3/3 |

These values differ from Exp117 because this implementation creates a new seeded palette and explicitly measures fully encoded files. The selector uses known synthetic family labels, which are **not available for arbitrary user images**; hence the wins cannot be claimed as operational codec superiority. Test families share generators with dev, and there is no external photographic holdout. No AVIF/JXL or lossy rate-distortion testing. Measurements include encoding/decoding timing in local JSON and process RSS, but isolated codec memory was not measured. A Python script and JSON were generated and executed in the local sandbox; **they are not committed to GitHub** because the GitHub connector cannot directly read local files. The branch report alone is not a reproducible Actions artifact. Next run must port the executable and JSON to this branch and add pinned CI, then test a deployable selector and external corpora. No demo or LinkedIn publication.
