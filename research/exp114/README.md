# Exp114 — reversible RGB transforms and independent-family exploratory evaluation (2026-11-17)

Continues Exp113 without changing its code or earlier branches. Tests **raw RGB**, **subtract-green modulo 256**, and **YCoCg-R-like reversible modulo-256 transform**, each with vertical residual differencing and zlib-9 or LZMA-6. The six variants are **ablations, not an operational per-image selector**. The 14-byte header and entire payload are included in byte counts. All variants are exactly decoded and checked against original pixels; WebP lossless and PNG also undergo byte-exact checks.

Local environment: Python 3, NumPy, Pillow 12.3.0, scikit-image 0.26.0. Run `python -m pip install numpy pillow scikit-image && python research/exp114/exp114.py > exp114-results.json`. Unit tests include 18 random RGB roundtrips (3 dimensions × 3 transforms × 2 coders), plus three malformed-input rejection cases. Two test methods passed locally.

Five **new grayscale image families** (brick, grass, gravel, page, text) from scikit-image, converted to RGB and cropped to 192×192, were used as exploratory holdout against prior Exp113 synthetic families. This is **not an external independent photographic corpus**; the same scikit-image package has been used in earlier experiments, and the five images are grayscale duplicated to RGB. They were not used to tune a winner. Exact original byte SHA256 and timings are emitted by the benchmark.

| Image | Best of 6 BLKH bytes (oracle) | WebP lossless bytes | PNG bytes | Best BLKH variant |
|---|---:|---:|---:|---|
| brick | 18162 | 13770 | 22746 | ycocg_lzma |
| grass | 31254 | 28874 | 53933 | raw_lzma |
| gravel | 28406 | 26322 | 45583 | raw_lzma |
| page | 26186 | 23320 | 44516 | raw_lzma |
| text | 19874 | 18224 | 29615 | raw_lzma |

**Negative result:** even the retrospectively selected best BLKH variant loses to WebP in all five cases. A fixed variant would do no better. Winning over PNG in these cases does not imply competitive compression. The image families are not independent external validation. Peak RSS is emitted for the Python process and includes dependencies and image loading, not only the codec. The Actions workflow is provided but its success must be confirmed from an actual run. No AVIF/JPEG XL comparison and no lossy rate-distortion curves are claimed; the code is lossless only. Future work: externally sourced RGB corpus, immutable split manifest, codec version pinning, JXL/AVIF lossless where supported, and lossy comparisons only after implementing a lossy encoder. No demo/LinkedIn announcement warranted.
