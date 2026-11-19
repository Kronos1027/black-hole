# BLKH Exp169 — reversible modular channel lifting + Paeth (2026-11-19)

Hypotheses: green-centered modular channel differences; chained modular RGB differences; causal Paeth versus left predictor, with zlib-9 and LZMA-6.

Local Python 3.13.5 / Pillow 12.3.0, Linux, seed 169. Six 64×64 RGB images (scikit-image and synthetic), 3 development / 3 held-out test. All 72 encode/decode round trips exactly reproduced RGB; 72 single-byte payload mutations rejected. Full serialized bytes include 19-byte header, payload and CRC32. WebP lossless (Pillow method 6) was independently decoded and verified.

Development selected **gdiff + Paeth + LZMA-6** (16,501 bytes). Held-out test: **BLKH 25,165 bytes; WebP 23,932 bytes**, BLKH **5.15% larger**. No demonstrated advantage.

Important limits: tiny non-external dataset; no AVIF/JPEG XL, R-D curves, independent memory measurement or GitHub Actions reproduction. This report is not proof of competitive compression. The runnable script and full JSON results were generated locally, but have not been committed; CI must not be claimed green. Next: upload code and JSON, add pinned CI with independently sourced photographic dataset and full codec matrix. No demo or LinkedIn publication recommended.
