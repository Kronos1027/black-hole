# Exp234: causal row-adaptive color transform (2026-11-22)

Experiment executed locally in Python 3 with Pillow 12.3 and scikit-image 0.26.0. Six 128x128 RGB images from scikit-image: dev astronaut/coffee/chelsea; held-out rocket/hubble_deep_field/immunohistochemistry. This corpus is **not externally independent**. Lossless reconstruction asserted for every codec/image pair (36 experimental and 12 reference round trips).

Hypotheses: RGB, fixed reversible green-relative color differences, and a causal adaptive reference-channel choice based only on the previous reconstructed row (decoder can reproduce the choice without a side map); entropy compressors zlib level 9 and LZMA preset 6. Transform modulo 256. The reported experimental sizes add a **provisional 32-byte header allowance** to compressed bytes; no actual BLKH container was serialized. Thus not a finalized full-container benchmark.

| Method | Dev bytes | Test bytes |
|---|---:|---:|
| RGB zlib |129752|104540|
| RGB LZMA |113108|91172|
| Green zlib |125982|102794|
| Green LZMA |103596|89168|
| Adaptive zlib |126178|104161|
| Adaptive LZMA |103832|90544|
| PNG |96374|90042|
| WebP lossless |71434|70320|

Selected on dev: fixed green + LZMA. Test: 89,168 bytes vs WebP 70,320 (+26.80%). Adaptive LZMA was 90,544 bytes (+1.54% versus fixed green), **negative hypothesis**. Not evidence of superiority.

Reproducibility limitations: executed bench.py and results.json remain in the ephemeral experiment environment (/mnt/data/exp234); not yet committed. No independent external images, AVIF/JXL, RAM peak, R-D curve or GitHub Actions run. Timing data exists in local JSON but not archived here. Next: upload exact script and JSON, create reproducible CI, use an actual self-describing container and independent external dataset. Preserve all prior research branches.
