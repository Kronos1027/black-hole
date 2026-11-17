# Exp104 — row-adaptive lossless predictors: negative result

Research-only experiment following Exp103. Each 8-bit grayscale row chooses among none, left, up, and Paeth via minimum sum of absolute signed residuals. Row choices are packed at 2 bits/row and included in the complete stream size (12-byte header + ceil(height/4) mode bytes + zlib residual). Exact decoder does not require source image.

## Local measurements, 2026-11-16

Five 128×128 **top-left crops** of bundled scikit-image images. These are exploratory, **not independently held-out datasets**. All decoded pixels matched the original exactly.

| Image | Exp104 | Exp103 | PNG | WebP lossless | Exp104 encode s | decode s |
|---|---:|---:|---:|---:|---:|---:|
| camera | 4311 | 3971 | 4348 | 3518 | .1613 | .0977 |
| coins | 9708 | 9700 | 9397 | 8070 | .1595 | .1016 |
| moon | 3077 | 3046 | 2991 | 2874 | .1458 | .0947 |
| page | 12160 | 12174 | 11846 | 11204 | .1678 | .1026 |
| astronaut_gray | 9190 | 9164 | 8319 | 7042 | .1571 | .1002 |

**Finding:** Exp104 loses to lossless WebP in all 5 cases, and to Exp103 in 4/5. Per-row predictor selection with this heuristic is not justified by current evidence. Memory measured with Python tracemalloc was ~319 KB peak per image (does not include all native allocations); do not interpret as peak RSS. Environment: Python 3.13.5, Pillow 12.3.0, Linux x86_64. Local tests: 15 varied exact roundtrips passed. Timing is exploratory, not standardized.

No AVIF/JPEG XL baseline, matched-quality lossy R-D curves, independent held-out corpora, or verified GitHub Actions run. Do not claim superiority. Next: consider block-size sweeps or per-row selection with actual entropy cost, but require a dev/test split and include signaling bytes. Do not optimize on these five crops.
