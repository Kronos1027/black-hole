# BLKH Exp185 — reversible 2D Haar lifting (2026-11-20)

Status: NEGATIVE; not suitable for demo or LinkedIn.

## Hypotheses
Four invertible transforms (RGB identity, green-channel decorrelation, integer Haar lifting, green+Haar) crossed with zlib-9 and LZMA-6 (8 configurations). Each stores 16-bit signed coefficients and a 14-byte header (magic, dimensions, transform/backend indices, CRC32). All tested encode/decode paths reconstructed RGB byte-for-byte; corruption probes were rejected. No claim of production-grade parser security.

## Protocol and outcome
Pillow 12.3.0, numpy 2.3.5, Python 3.13.5, scikit-image sample images resampled to 128x128 RGB. Development (astronaut, coffee, chelsea, rocket), holdout (immunohistochemistry, hubble_deep_field, colorwheel, cat). Configuration selected only by total development bytes: green+LZMA. Holdout: BLKH 125,272 bytes versus WebP lossless 83,542 bytes, **BLKH +49.95%**. AVIF quality=100 yielded 53,581 bytes but did NOT decode losslessly; this is a lossy, non-rate-distortion-matched reference and MUST NOT be interpreted as a fair lossless comparison. No JPEG XL encoder available. Exactness of WebP verified in code.

## Limitations
Small bundled sample dataset, not independent external photographs. No lossless AVIF configuration validated, no JXL, no matched-PSNR rate-distortion curves, no isolated peak RSS, no GitHub Actions run. The script and per-image JSON were run locally but are not yet uploaded to this branch. Local artifact: BLKH_exp185_bundle.zip. This commit records results, not reproducibility. Next execution should port script into repository, add Actions, collect external photographic dataset with pinned hashes, and benchmark equivalent distortion settings.
