# Exp120 — palette order / predictor holdout (2026-11-17)

## Objective
Test three deterministic palette orderings (first occurrence, frequency descending with RGB tie-break, RGB lexicographic), five reversible predictors (raw, XOR-left, subtract-left, subtract-up, Paeth), and two entropy backends (zlib-9, LZMA-6). Freeze a **single global configuration** using development images only, then evaluate on disjoint test images.

## Executed locally
Python 3.13.5; NumPy 2.3.5; Pillow 12.3.0; scikit-image 0.26.0. Peak process RSS 157,988 KiB. Image size 128x128. All sources converted to 16-level grayscale RGB with the **same preprocessing before all codecs**. Dev: camera, coins, moon and seeded random images (101–103). Test: astronaut, coffee, rocket, chelsea, brick and seeded random images (9101–9103). This is NOT an original-photography lossless benchmark: grayscale quantization changes the source; losslessness is measured only relative to the common preprocessed input.

Global development-selected configuration: **RGB-lexicographic palette, raw indices, LZMA-6**. Complete BLKH stream includes 12-byte header, exact palette bytes, and compressed payload. Decode checked RGB pixel-for-pixel. WebP lossless method=6 and optimized PNG decoded and verified against identical RGB input.

| Held-out image | BLKH bytes | WebP lossless bytes | BLKH-WebP |
|---|---:|---:|---:|
| astronaut | 4416 | 4658 | -242 |
| coffee | 3828 | 4040 | -212 |
| rocket | 1857 | 1764 | +93 |
| chelsea | 3804 | 3888 | -84 |
| brick | 2245 | 2334 | -89 |
| noise_9101 | 8688 | 8266 | +422 |
| noise_9102 | 8680 | 8264 | +416 |
| noise_9103 | 8676 | 8264 | +412 |

This exploratory advantage on four quantized natural-image derivatives is **not** a competitive codec discovery: 128px 16-gray palette sources are a constrained niche, WebP beats BLKH on all held-out random sources, JPEG XL/AVIF not measured, and no independent GitHub Actions reproduction has yet been confirmed. The global choice is locked but shares quantization protocol across dev/test. Timing measurements and PNG sizes exist in the local JSON but are not published in this report to avoid transcription error.

## Reproducibility and blocker
The full local Python executable and JSON are attached in the automation execution environment under `/mnt/data/exp120/experiment.py` and `/mnt/data/exp120/results.json`; the GitHub connector cannot read that local filesystem and the container has no GitHub DNS access. **These files have not been uploaded to this branch.** Do not claim Actions reproduction or treat this report as a complete release. Next run should transfer exact bytes via a shared environment or reimplement and cross-validate hashes/results before enabling CI.

## Decision
No public demo or LinkedIn publication. Continue with independent real images, exact original RGB and equal-quality rate-distortion comparisons against AVIF/JPEG XL, with confirmed Actions reproducibility.
