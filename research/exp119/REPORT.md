# Exp119 — operational selection, no generator-family oracle

Starting point: Exp118 selected a predictor separately for each known synthetic family. Exp119 removes that privileged family label. Ten predictor/backend configurations are trained on 15 development images (seeds 101–103, five families). The global configuration is frozen before evaluation on 15 test images (seeds 9101–9103). A second, deployable method compresses each test image with all ten configurations and chooses the smallest complete self-describing bitstream. This is not a retrospective uncharged oracle: selection is performed by the encoder, its time is measured, and the mode is stored in the 59-byte header.

## Local exploratory results, 2026-11-17
Python 3.13.5, NumPy 2.3.5, Pillow 12.3.0; peak process RSS 118296 KiB including interpreter/libraries. Sizes are mean complete bytes per 128x128 RGB image, three test seeds per family:

| Family | WebP lossless method=6 | Global BLKH (raw/LZMA) | Encoder-selected BLKH |
|---|---:|---:|---:|
| random 16 colors | 8342.7 | 8680.3 | 8679.0 |
| stripes | 174.7 | 207.0 | 110.3 |
| diagonal | 287.3 | 243.0 | 196.7 |
| blocks | 498.7 | 435.0 | 435.0 |
| repeated rows | 212.7 | 272.3 | 182.0 |

All 30 BLKH decodes (global and selected) and 15 WebP decodes reconstructed their source exactly in the local run. Encode and decode timings per image are recorded by the script. Global selection fails to beat WebP on stripes, repeated rows, and random color patterns; encoder-side adaptive selection beats WebP on four highly artificial structured families but loses on random-color images. This does **not** establish superiority on real images, and the family overlap between dev/test is a limitation. The palette is always exactly 16 colors and its full 48 bytes are counted. Results apply only to this synthetic domain.

## Reproduction and controls
`python -m pip install numpy==2.3.5 Pillow==12.3.0`; `python research/exp119/experiment.py > exp119-results.json`. Validate all decodes, archive the JSON and environment, check the Actions run independently. The repository script is a standalone transcription of the locally executed experiment; matching CI output is **not yet verified**. This experiment does not benchmark AVIF, JPEG XL, lossy rate-distortion, photographic holdout, or cross-machine memory. Next: real external lossless corpus and operational selector with generic palettes, compare WebP/JXL/AVIF with equal RGB fidelity, and include actual per-encoder peak RAM. No demo or LinkedIn claim warranted.
