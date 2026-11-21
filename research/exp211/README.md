# Exp211 — Row-adaptive PNG-like predictors (2026-11-21)
Prior research checked: Exp201 band-adaptive predictors; Exp209 residual compressors; Exp210 palettes. This tests **per-row** minimum signed residual score, not band selection.

Executed locally in Python 3 with Pillow 12.3.0 and scikit-image 0.26.0, RGB 128x128, WebP lossless method 6, zlib level 9, complete B211 header and payload bytes. Every candidate was decoded and compared pixel-exact. Development: astronaut, coffee, chelsea, rocket. Holdout: camera, coins, moon, horse. Selected adaptive on development without holdout tuning.

| Holdout | Adaptive bytes | WebP lossless bytes |
|---|---:|---:|
| camera | 14057 | 8314 |
| coins | 16861 | 10294 |
| moon | 10740 | 6376 |
| horse | 745 | 364 |
| TOTAL | **42403** | **25348** |

BLKH is **67.28% larger** than WebP lossless. Negative result. Data are scikit-image bundled examples, not independent external datasets. No AVIF/JXL available. Shared process peak RSS, not codec-isolated. No lossy rate-distortion comparison. Code is in exp211.py; CI workflow still pending. Do not claim codec competitiveness or publish demo/LinkedIn.
