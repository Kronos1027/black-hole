# Exp227 — causal gradient-context zlib ablation (2026-11-22)

Status: negative exploratory result; not a scientific breakthrough.

Prior work reviewed: Exp220/221 branches, historical Exp223–226 reports from automation history. This experiment tests **gradient-context residual stream splitting**, which differs from Exp225's reported threshold split by using four gradation bins per channel; it is still a related ablation and does not constitute an independent codec.

Dataset: scikit-image 0.26, images resized to RGB 96x96 using Pillow Lanczos. Development: astronaut, coffee, chelsea. Locked holdout: rocket, hubble_deep_field, immunohistochemistry.

Methods: raw RGB zlib9 streams per channel; MED residual zlib9; MED shifted residual zlib9; MED residual with causal gradient contexts (sum of absolute differences to upper-left neighbor, four bins per channel) and 12 separate zlib9 streams. Header includes magic, dimensions, method, and complete compressed-stream lengths. Every method decoded byte-exact on all six images. Pillow PNG compress_level=9 and WebP lossless method=6 were also round-trip checked.

| Codec | Dev bytes | Test bytes |
|---|---:|---:|
| Raw channel zlib9 | 73651 | 61393 |
| MED zlib9 | 59382 | 55324 |
| MED shifted zlib9 | 59390 | 55324 |
| MED gradient-context zlib9 | 59536 | 55421 |
| PNG | 57022 | 52362 |
| WebP lossless | 42724 | 40872 |

Gradient contexts *worsened* MED on holdout by 97 bytes (0.175%) and are 35.6% larger than WebP lossless. Do not promote.

Execution performed locally; no Actions run. Full script and machine-readable measurements exist as local execution artifacts `/mnt/data/exp227/bench.py` and `/mnt/data/exp227/results.json` in the execution environment; these are NOT yet checked into this branch. This report alone is not a reproducible GitHub Actions experiment. Memory data are Python tracemalloc allocations only, not isolated process RSS. JPEG XL, AVIF, external corpus, and rate-distortion remain pending.

Next: commit full benchmark script plus frozen external test manifest and CI; compare arithmetic coder under matched datasets and compute full process peak RSS. No demo or LinkedIn publication.
