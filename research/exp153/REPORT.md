# BLKH Exp153 — Paeth / spatial prediction, lossless

Date: 2026-11-18. Continuation of Exp152, distinct from bitplane and DCT experiments.

## Experiment
Local executable: `/mnt/data/exp153/experiment.py`; machine-readable output: `/mnt/data/exp153/results.json`. Python 3, NumPy, scikit-image, Pillow. Eight 128x128 RGB images from scikit-image, fixed split: dev astronaut/coffee/chelsea/rocket; holdout camera/coins/moon/page. Ten configurations: predictors none/left/up/avg/Paeth x zlib-9/LZMA-6. Custom complete container B153 with 14-byte header, lossless residual modulo 256, decoded pixel equality asserted for all 80 combinations. WebP lossless method 6 encoded and decoded for all 8 images, exact RGB equality asserted. All timings and SHA256 reconstruction hashes in local JSON. No external independent photo dataset.

## Predeclared selection and results
Choose lowest total bytes on four development images only. Winner: UP + LZMA-6.

| Split | BLKH bytes | WebP lossless bytes | BLKH overhead |
|---|---:|---:|---:|
| Dev (4) | 103024 | 82792 | +24.44% |
| Holdout (4) | 40008 | 33784 | +18.42% |

**Negative result**: spatial Paeth does not win the development selection; even selected UP+LZMA loses against WebP. No superiority claim. AVIF/JXL and independent external datasets not executed in this run. GitHub Actions reproduction NOT confirmed. Python source and JSON not yet pushed: this report is NOT reproducibility evidence by itself. No demo or LinkedIn. Preserve existing branches and alerts.
