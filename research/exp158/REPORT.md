# Exp158 — row-adaptive reversible predictors (2026-11-19)

## Hypothesis and protocol
Compared 7 lossless PNG-style spatial prediction modes: none, left, up, average, Paeth, adaptive per-row SAD, and adaptive per-row nonzero-count. Each used zlib level 9 or LZMA preset 6. The complete BLKH container is serialized (18-byte header, compressed payload, 4-byte CRC32); adaptive row selectors are included in the compressed payload. All decoded RGB arrays were checked for exact equality. Truncation and header corruption rejection were checked per encoded candidate.

Six bundled scikit-image images were resized to 64x64 RGB. Development: astronaut, coffee, chelsea. Held-out evaluation: rocket, hubble_deep_field, immunohistochemistry. No random seed. All 84 BLKH configurations (6 images x 7 filters x 2 compressors) decoded exactly; 168 malformed-stream checks were performed. Six WebP lossless files were generated and decoded exactly with Pillow/libwebp.

## Selection and result
Development-only winner: adaptive_sad + LZMA-6, total 24,878 bytes on three development images. On held-out evaluation: BLKH 22,134 bytes vs WebP lossless 18,120 bytes (+22.15% BLKH size). Aggregate selected BLKH encode time 0.01349s; decode time 0.09038s (single-run, non-comparable to codec baselines). This is a negative result. Selection may not generalize.

## Limitations / reproducibility blockers
Local script and JSON results are in the run's /mnt/data/exp158/ workspace, **not committed to this branch**. The test set is tiny and drawn from bundled scikit-image samples, not an independent external dataset. AVIF/JPEG XL not evaluated. No lossy rate-distortion curves, no isolated peak memory per codec. GitHub Actions reproduction is **not established**. Do not claim competitive superiority or prepare public demo/LinkedIn scientific claims.

## Next required step
Commit executable script and JSON data, add pinned-dependency Actions workflow and validate run URL, then evaluate external photos with matched PSNR/MS-SSIM rate-distortion curves and complete container byte counts.
