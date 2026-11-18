# BLKH Exp148 — lossless codec switching by tile (local run, 2026-11-18)

## Hypothesis
For heterogeneous images, independent WebP/JPEG XL lossless tile streams might outperform a whole-image codec. Tested 128x128 tiles and 256x256 whole-image baselines, WebP-only, JPEG XL-only and per-tile minimum-byte switching. Self-contained B148 framing includes dimensions, tile size, codec IDs, payload lengths, CRC32 and payload bytes. Candidate encoding time includes all attempted codecs.

## Dataset and separation
8 bundled scikit-image samples, all converted to 256x256 RGB with Pillow LANCZOS. Development: camera, coins, clock, grass. Holdout: gravel, moon, retina, cell. Some images reused historically: **not an independent external validation corpus**. No tuning on holdout.

## Results
48 exact RGB roundtrips across 8 images x 6 configurations. All reconstructed arrays matched source bytes exactly. Selected on development: **256_webp** (143,130 bytes), vs 256_jxl 150,211 bytes and 128_mixed 144,924 bytes.

Held-out totals (4 images):
| Configuration | Full-file bytes |
| --- | ---: |
| 128_webp | 166,206 |
| 128_jxl | 174,972 |
| 128_mixed | 166,188 |
| 256_webp (selected) | **162,362** |
| 256_jxl | 171,939 |
| 256_mixed | **162,362** |

At 128x128, per-tile switching saved only **18 bytes** over tiled WebP and remained **3,826 bytes worse** than whole-image WebP. Whole-image WebP was 5.57% smaller than whole-image JXL in this holdout. Thus **negative**: spatial tile switching did not beat whole-image WebP. The result is limited to these eight 256px images and tested codec parameters.

## Reproducibility and blockers
Source script and machine-readable per-image results generated locally at /mnt/data/exp148/experiment.py and /mnt/data/exp148/results.json. The local git CLI cannot resolve github.com (DNS), so these files are not yet committed; this report was submitted through the GitHub connector. No GitHub Actions run for Exp148, no independent external dataset, no lossy RD curves, and no AVIF lossless benchmark. Local ffmpeg libjxl used distance=0 and effort=1; Pillow WebP lossless method=6. Peak RSS is process-wide and in local JSON, not isolated by codec. Avoid comparing this result to Exp147 as though identical codec settings were used.

## Next
Publish executable script + JSON and CI workflow via connected API, verify full decode on Actions, obtain external license-permitted photographic corpus, compare image size stratified by content, then test tile sizes only on development before holdout. Do not claim discovery, publish demo or LinkedIn.