# Exp115 — indexed palette lossless ablation (2026-11-17)

Continuation of Exp114. Local Python 3 + NumPy 2.3.5 + Pillow 12.3.0 experiment tested two reversible representations: raw RGB and indexed palette (only when <=256 unique RGB colors), each encoded with zlib-9 or LZMA-6. File format uses a 14-byte header, 3 bytes per palette color, and complete entropy stream. Decoder reconstructs exact RGB. WebP lossless Pillow method=6 served as baseline; all WebP files decoded exactly. This is a **synthetic exploratory holdout**, not an independent photographic corpus.

Generated 5 families (flat, random 4-color palette, random 32-color palette, 8x8 palette blocks, random RGB noise), 128x128 RGB, with development seeds 101,102,103 and untouched test seeds 9101,9102,9103. 30 images; 108 exact BLKH encode/decode roundtrips and 30 WebP checks; malformed headers rejected. Per-image best of up to four BLKH variants is an **oracle**, not a deployable selector.

| Split | Family | Oracle wins over WebP | Mean BLKH bytes | Mean WebP bytes |
|---|---|---:|---:|---:|
| dev | flat | 0/3 | 56 | 44 |
| dev | palette4 | 0/3 | 4941 | 4193 |
| dev | palette32 | 0/3 | 10745 | 10419 |
| dev | blocks | 0/3 | 411 | 373 |
| dev | noise | 3/3 | 49187 | 49244 |
| test | flat | 0/3 | 56 | 44 |
| test | palette4 | 0/3 | 4934 | 4192 |
| test | palette32 | 0/3 | 10747 | 10421 |
| test | blocks | 0/3 | 410 | 386 |
| test | noise | 3/3 | 49187 | 49241 |

**Interpretation:** noise-only savings (~54 bytes on test) are likely container/metadata overhead, not improved predictive modeling. Palette coding fails to outperform WebP on every palette family. No competitive advance.

**Execution and publication limitation:** local script and complete per-image JSON were generated in the execution container, but the GitHub connector cannot directly read those local files, and outbound git networking is unavailable. This branch contains the report only; code/results/Actions have **not** been committed, and no GitHub CI run has been verified. Do not treat the above as independently reproducible until the exact code, dataset generator and results are uploaded and rerun. Follow-up: publish code and JSON from the run artifacts, pin Pillow/libwebp and LZMA versions, run CI, then compare JPEG XL/AVIF lossless on external photographic holdout. No demo or LinkedIn claim.
