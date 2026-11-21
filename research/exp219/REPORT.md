# Exp219: reversible modular YCoCg-R + Paeth (2026-11-21)

Executed 144 encode/decode round trips on eight 128x128 RGB crops from scikit-image (4 development, 4 holdout). Tested RGB, green-XOR and modulo-256 YCoCg-R, each with raw or Paeth residuals and zlib9/LZMA6/Brotli9. All candidate reconstructions exact. Complete 11-byte header included.

Selected on development: ycocg_paeth_lzma (97,920 bytes). Holdout: BLKH 101,848 bytes, WebP lossless 81,542 bytes, PNG 105,872 bytes. BLKH **24.90% larger** than WebP. AVIF quality=100 failed exactness and cannot serve as a lossless comparator. Negative result; PNG advantage alone is insufficient.

Limitations: only built-in scikit-image samples, not independent external photos; no JPEG XL encoder, no controlled lossy R-D curve, no per-codec isolated RAM; GitHub Actions not yet executed. Local executable bundle generated separately, with exp219.py, results.json and proposed workflow. No demo or LinkedIn publication justified.
