# BLKH Exp167 — reversible channel delta + spatial predictors (2026-11-19)

## Provenance and design
Reviewed prior Exp157–166 summaries and the research branches. Exp166 explored RGB decorrelation; this experiment is an **independent implementation / replication plus predictor interaction**, not a claim of a wholly novel transformation. Eight fixed scikit-image samples resized to RGB 128×128. Development: astronaut, coffee, chelsea, rocket. Held out: hubble_deep_field, immunohistochemistry, colorwheel, logo. No external independent photographic test set.

Candidate matrix: raw, up, left, xorup, green-channel delta, delta+up, delta+left, delta+xorup; each compressed with zlib-9 or LZMA-6. 8 images × 16 variants = **128** BLKH round trips, all pixel-exact; 128 header-corruption rejection checks. Complete BLKH format: 4-byte magic, 2-byte H, 2-byte W, 1-byte transform, 1-byte compressor, 4-byte CRC32, compressed payload. Selection solely on development: **deltaup + LZMA-6**.

## Held-out aggregate (four images)
| Codec | Complete bytes | Exact reconstruction | Encode seconds (one run) | Decode seconds (one run) |
|---|---:|---|---:|---:|
| BLKH selected | 84556 | yes | 0.1567 | 0.0133 |
| WebP lossless (Pillow method=6) | 70652 | yes | 5.3587 | 0.0042 |
| PNG optimize | 96151 | yes | 0.0425 | 0.0030 |
| AVIF quality=100 | 48282 | **NO** | 0.1077 | 0.0103 |

BLKH is **19.68% larger than WebP lossless** on held-out images. AVIF quality=100 is lossy in this run and cannot serve as a lossless baseline. Process peak RSS 153132 KiB, not isolated by codec. Python 3.13.5, Pillow 12.3.0. Times are single-run measurements without uncertainty bounds. No rate-distortion curves or JPEG XL benchmark. No independent external corpus. Results **negative**, no scientific discovery, no demo/LinkedIn.

## Reproduction status
Executed locally in the automation container. Full runnable Python and JSON results preserved in the run artifact `exp167` (not yet uploaded to GitHub). **No GitHub Actions run**; CI reproducibility not established. Next: commit executable code/results, add pinned-dependency Actions, obtain external photo dataset, validate AVIF lossless mode or explicitly label lossy, and test JPEG XL under matched settings. Preserve this branch; do not merge based on this result.
