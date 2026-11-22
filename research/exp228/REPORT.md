# Exp228 — green-difference + MED entropy ablation (2026-11-22)

Exploratory negative result. Parent: Exp227. **Do not promote as competitive codec.**

Local experiment on six scikit-image images resized to 96×96 RGB with Pillow Lanczos. Development: astronaut, coffee, chelsea. Locked holdout: rocket, hubble_deep_field, immunohistochemistry. 2 reversible channel transforms (RGB, modular green differences) × 3 predictors (none, left, MED) × 3 entropy backends (zlib9, bz2-9, LZMA6) × 6 images = **108 exact encode/decode roundtrips**. Complete custom header: 4-byte magic, three method bytes, 16-bit width and height; 11-byte total header. Output checked for exact RGB equality and SHA256 logged. YCoCg experiment was rejected after an 8-bit signedness bug caused a failed roundtrip; excluded from 108 passing cycles.

| Method | Dev complete bytes | Holdout complete bytes |
|---|---:|---:|
| Green differences + MED + LZMA6 | 48533 | 46037 |
| Green differences + MED + zlib9 | 52284 | 49202 |
| RGB + MED + LZMA6 | 52909 | 49929 |
| PNG lossless | 57022 | 52362 |
| WebP lossless | 42662 | 40904 |

Green/MED/LZMA reduces holdout size by 7.80% relative to RGB/MED/LZMA, but is **12.55% larger than WebP lossless**. WebP still wins. The model is not an arithmetic coder. AVIF Pillow lossless flag failed byte-exact validation and was excluded; JPEG XL not run. No independent external corpus, isolated RSS, RD curve, or Actions run. Not a breakthrough.

The executable local script and JSON were generated at /mnt/data/exp228/bench.py and /mnt/data/exp228/results.json; **not committed**. This report is not CI reproducibility. Next: commit script, introduce external corpus with fixed checksums, measure RSS and run JXL/AVIF correctly. Never publish demo or LinkedIn automatically.
