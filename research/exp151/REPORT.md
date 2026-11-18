# BLKH Exp151 — DC-delta and AC coefficient ordering (2026-11-18)

Research after Exp149/150: 8×8 DCT coefficient entropy layout with four hypotheses (raw zigzag; DC differential; AC column-major; combined), 3 retained coefficient counts (10/20/32), 3 quantization steps (12/24/48), zlib-9/Brotli-9.

Executed locally: 6 grayscale scikit-image images resized to 128×128, 3 development (camera/moon/coins) and 3 held-out evaluation (page/grass/gravel); 432 BLKH encode/decode checks, 1728 malformed-stream rejection checks, and 72 actual WebP/AVIF/JXL encode/decode runs. All file bytes include 12-byte header. PSNR and SHA-256 recorded. Decoder is deterministic, not lossless because experiment is intentionally lossy.

**Predeclared selection:** minimize total development bytes with PSNR >=26 dB on *every* development image. Selected: dc_delta, 20 coefficients, step=48, zlib.

| Holdout | BLKH bytes | BLKH PSNR | WebP bytes >= BLKH PSNR | AVIF | JXL |
|---|---:|---:|---:|---:|---:|
| page | 1488 | 24.25 | 1622 | 1499 | 2735 |
| grass | 1863 | 24.97 | 2394 | 1538 | 1749 |
| gravel | 2117 | 24.41 | 2826 | 1837 | 2084 |
| total | 5468 | — | 6842 | 4874 | 6568 |

**Negative conclusion:** all three holdout images fell below the 26 dB quality constraint. The sparse conventional codec settings often yield materially *higher* quality; no fair BD-rate superiority claim. AVIF beats BLKH in total bytes at higher PSNR. Combined dc_delta_ac_column/zlib yielded 5115 bytes on holdout at fixed quantization, but is post-hoc and cannot be substituted for the development-selected configuration.

**Reproducibility limitations:** scikit-image sample images, not independent external corpus; 3+3 small grayscale images, no BD-rate; FFmpeg startup included in baseline times; peak RSS whole-process only. The Git CLI failed DNS resolving github.com, but GitHub connector could create this isolated branch. Local tested source, full JSON and CI workflow exist in /mnt/data/exp151 and bundle BLKH_exp151_bundle.zip; they are **not yet uploaded to this branch**. No GitHub Actions execution is claimed. Next step: upload those files, trigger CI and verify matching input hashes and exact decode before any scientific announcement.

No demo, LinkedIn, or modification of scientific-alert automation.