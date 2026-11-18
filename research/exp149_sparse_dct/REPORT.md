# BLKH Exp149 — sparse DCT grayscale lossy compression

## New experiment
A new codec stores the quantized low-frequency k×k 2D-DCT coefficients of a 128×128 grayscale image (16-byte B149 header plus zlib-9 payload). Decoder reconstructs pixels via inverse DCT and measures actual decoded PSNR. 4 coefficient budgets (k=8,16,32,64) × 3 quantizer steps (2,8,32) × 6 images = **72 real BLKH encodes and decodes**. FFmpeg additionally encoded and decoded **72 real baseline files** (WebP, AVIF, JPEG XL). All 72 baseline commands succeeded.

Data: scikit-image grayscale 128×128. Development: camera, moon, coins. Held-out: page, grass, gravel. Development-only selection rule: minimize total bytes with >=27 dB PSNR on *each* development image. Selected k=64, quantizer step=32.

## Held-out results
| Image | BLKH bytes | BLKH PSNR | WebP bytes* | AVIF bytes* | JPEG XL bytes* |
|---|---:|---:|---:|---:|---:|
| page | 1738 | 23.81 | 1622 | 1499 | 2308 |
| grass | 1971 | 25.48 | 2394 | 1538 | 1507 |
| gravel | 2186 | 24.83 | 2826 | 1837 | 1773 |
| **Total** | **5895** | — | **6842** | **4874** | **5588** |

* Smallest actual encoded baseline file among four quality settings that achieves **at least** the BLKH image's PSNR. Baseline PSNR is frequently higher, so this is not an equal-PSNR rate-distortion comparison and the BLKH/WebP total advantage does **not** demonstrate a breakthrough. AVIF and JPEG XL totals are lower than BLKH even under this baseline constraint.

## Limitations and reproducibility
- Original image identities differ between development and holdout, but the corpus is bundled with scikit-image and is not an external unseen photographic dataset.
- Lossy reconstruction was tested by decoding real B149 files; repeated decoding was deterministic. The B149 format includes full headers and compressed payload.
- This is **not** lossless. No claim of exact original RGB recovery.
- Encoding/decoding times, SHA-256 hashes, per-configuration sizes/PSNR, and process-wide peak RSS (106636 KiB) are in the locally generated results.json.
- **GitHub Actions not executed** for Exp149; the full local experiment.py and results.json have not been uploaded in this commit. Do not interpret this report commit as reproducible CI evidence.
- Need denser real-bitstream quality grids, independent external images, isolated RAM measurements and a published executable test harness.

## Next
Upload the complete tested script and JSON, then run an isolated GitHub Actions job with pinned dependencies. Explore blockwise DCT with zigzag/entropy allocation, selecting hyperparameters only on development data. Recheck all comparisons against modern codecs at genuinely matched quality.

**Verdict:** No independently validated competitive discovery. No demo or LinkedIn publication.