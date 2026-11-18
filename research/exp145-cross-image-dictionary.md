# Exp145 — cross-image preset dictionaries (2026-11-18)

Hypothesis: zlib preset dictionaries trained on development images improve residual coding on unseen images. No evidence of a competitive improvement.

Local execution: 72 exact encode/decode roundtrips, 4 development and 4 holdout synthetic RGB images (192×192), seeds 1452026..1452029 and 1452126..1452129. Each of 3 residual modes (raw, vertical modular difference, horizontal modular difference) tested with no dictionary, shared dictionary and standalone dictionary. Dictionary trained ONLY from development images using sampled residual bytes; 32768 bytes per mode. Full 16-byte envelope included per image; standalone dictionary includes 32768 bytes per image. Shared dictionary numbers **exclude** the once-per-collection 32768-byte dictionary and are not a standalone fair comparison.

Holdout totals (4 images):
| Mode | No dictionary | Shared dictionary (dictionary excluded) | Standalone dictionary included | WebP lossless method 6 |
|---|---:|---:|---:|---:|
| raw | 301963 | 301991 | 433063 | 192916 |
| vertical | 219987 | 220068 | 351140 | 192916 |
| horizontal | 218827 | 218905 | 349977 | 192916 |

Conclusion: all dictionary variants were worse than the corresponding no-dictionary variant, even before shared dictionary amortization. Best internal horizontal variant 218827 bytes is 13.43% larger than WebP lossless. Negative result; do not claim superiority.

Local artifacts: `/mnt/data/exp145/experiment.py`, `/mnt/data/exp145/results.json`. The local script and JSON are **not** in this GitHub commit. Execution environment recorded in JSON, including hashes and individual timings. Process-wide peak RSS 122716 KiB (Linux ru_maxrss units); not codec-isolated. No independent real photographs, AVIF/JXL, lossy RD curves or Actions validation. Dataset synthetic only; not evidence of photographic generalization. No demo or LinkedIn publication.

Next: add executable script and deterministic CI to repository, run on external natural images and evaluate appropriate lossless codecs with full dictionary amortization.