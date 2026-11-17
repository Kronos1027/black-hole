# Exp121 — global locked selector / row-RLE ablation (2026-11-17)

**Status:** exploratory negative/mixed; NOT a competitive codec claim. The executable and raw JSON were generated and run in the separate local runtime, but are not yet copied into this branch. Consequently GitHub Actions reproduction is NOT established.

## Design
- Four candidate transforms: raw palette indices, row RLE, XOR-left, subtraction-left; zlib-9 and LZMA-6 for each.
- One GLOBAL configuration selected by summed complete-file bytes on 12 development synthetic images (seeds 11,12,13; four families). No family oracle at evaluation.
- Frozen choice: **raw / LZMA-6**.
- Holdout: 12 synthetic images (seeds 9101,9102,9103) plus 5 natural scikit-image samples converted to **96x96 16-level grayscale before BOTH encoders**. This is not compression of original RGB photos and not a rate-distortion curve.
- Decode validates exact indices, palette, and reconstructed RGB. WebP lossless decoded and compared pixel-for-pixel. Complete output bytes measured; encode/decode wall time recorded in local JSON.

## Holdout full-file bytes (BLKH vs WebP lossless)
- noise: 4955/4766, 4955/4762, 4955/4762 (three losses)
- stripes: 199/184, 199/174, 195/170 (three losses)
- blocks: 323/348, 327/346, 331/350 (three wins)
- diagonal: 267/358, 227/276, 255/348 (three wins)
- astronaut: 2899/2990; coffee: 2435/2636; rocket: 1199/1142; chelsea: 2527/2528; camera: 1807/1832.

## Environment and limitations
Python 3.13.5; NumPy 2.3.5; Pillow 12.3.0; peak process RSS 165568 KiB (includes dependencies). Three invalid inputs rejected. No AVIF/JXL executables; no external independent photographic corpus, no CI confirmation, no curves of rate-distortion. No demo or LinkedIn publication justified.

## Next required action
Transfer local `/mnt/data/exp121/experiment.py` and `results.json` into this branch, run GitHub Actions, compare outputs and investigate discrepancies before any claim. Preserve other branches and scientific alerts.
