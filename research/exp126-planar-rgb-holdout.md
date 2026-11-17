# Exp126 — planar RGB residual layout (2026-11-17)

Local execution: 8 RGB images at 128x128, 4 development and 4 holdout (camera, hubble_deep_field, immunohistochemistry, colorwheel). Compared 6 reversible representations (raw, left, up, green, green+left, green+up), 2 byte layouts (interleaved and planar), 2 compressors (zlib-9, LZMA-6): 24 configurations, 192 exact encode/decode checks. Container includes 15-byte header. Selection by summed development bytes, frozen before holdout.

Locked configuration: green_up / planar / lzma6. Holdout sizes BLKH vs WebP lossless (Pillow method=6): camera 9267 vs 8596; hubble 26371 vs 22604; immunohistochemistry 31651 vs 29998; colorwheel 11307 vs 7826. Totals 78596 vs 69024 bytes: BLKH 13.87% larger. All four lose. Negative result. Peak process RSS and encoding times are recorded in local JSON, not isolated per codec. Input images resized from skimage sources to 128x128; no claim for originals.

**Reproducibility blocker:** script and results generated in isolated execution environment at /mnt/data/exp126/experiment.py and results.json, not yet uploaded to GitHub. No GitHub Actions validation, AVIF/JXL or RD curves. Do not advertise competitive superiority. Next run: commit executable and results, add Actions, run independent external corpus and proper RD benchmarks. Preserve all previous branches.
