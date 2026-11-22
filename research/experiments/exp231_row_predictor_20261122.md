# Exp231 — row-wise reversible predictor selection (2026-11-22)

## Hypothesis
For green-difference transformed RGB, choosing among None, Left, Up, Average and Paeth/MED predictors per row and channel may outperform fixed MED under LZMA, even after counting selector bytes.

## Execution
Python 3, Pillow 12.3.0, scikit-image 0.26.0. Six bundled scikit-image photographs resized to RGB 96x96 with Pillow LANCZOS. Development: astronaut, coffee, chelsea. Reserved test: rocket, immunohistochemistry, hubble_deep_field. All five variants decoded and matched input exactly (30 roundtrips). File size includes 9-byte BLKH experimental header, predictor selection bytes where applicable, and LZMA container. PNG uses compress_level=9; WebP uses lossless=True, method=6. Tests are not independent external validation.

| Split | RGB MED LZMA | Green MED LZMA | Green row-adaptive LZMA | PNG | WebP lossless |
|---|---:|---:|---:|---:|---:|
| Development (3) | 56603 | 48459 | 48275 | 57022 | 42724 |
| Reserved test (3) | 53035 | 45663 | 45103 | 52362 | 40872 |

Row-adaptive improves on green MED by 560 bytes (1.23%) on reserved test, but is **10.35% larger than WebP lossless**. Selection uses a deterministic sum-of-absolute-modulo-residual proxy, not actual entropy-optimized mode search. No codec competitiveness demonstrated.

## Reproduction and blockers
Executed script and JSON locally at /mnt/data/exp231/bench.py and /mnt/data/exp231/results.json. **These are not yet committed** and this report is not a reproducible GitHub Actions workflow. Need to commit code and dataset manifest, measure isolated peak RSS, test AVIF/JPEG XL, evaluate full R-D curves with equivalent distortion and independent external corpus, and run Actions. Do not cite this as a scientific breakthrough or publish demo/LinkedIn.

## Next hypothesis
Use actual encoded block length or entropy estimates for per-tile predictor selection; compare overhead and speed, and validate held-out image sets before selecting a winner.
