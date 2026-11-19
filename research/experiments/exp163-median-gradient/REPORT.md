# BLKH Exp163 — reversible gradient/median predictors (2026-11-19)

Experimental local execution, not yet reproduced in GitHub Actions. Five predictors (up, left, clipped gradient, median edge detector, average), each with zlib-9 and LZMA-6. Eight scikit-image images resized to RGB 128x128, first four development and last four held-out test. 80 BLKH full-file encode/decode roundtrips passed, 80 CRC-corruption tests rejected, WebP lossless via Pillow quality=100 method=6 roundtrip validated for eight images. Winner selected exclusively on development: median + LZMA-6.

| Split | BLKH bytes | WebP bytes | BLKH overhead |
| --- | ---: | ---: | ---: |
| Development (4 images) | 101688 | 82660 | 23.02% |
| Test (4 images) | 29948 | 24362 | 22.93% |

Full BLKH serialized bytes include 14-byte header and 4-byte CRC32. Peak whole-process RSS: 158524 KiB (not codec-isolated). Local artifacts: /mnt/data/exp163/experiment.py and results.json. **These artifacts are not yet committed**, and there is **no CI reproduction**. No independent external photographic dataset, AVIF/JPEG XL comparisons, lossy RD curves, or statistical timing runs. Thus this is a negative result, not a codec breakthrough. No demo or LinkedIn publication warranted. Next: commit executable script and JSON, add Actions job, validate independent external dataset, measure codec comparisons fairly.
