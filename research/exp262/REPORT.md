# Exp262 — reversible channel deltas (2026-11-23)

96 successful RGB lossless BLKH encode/decode cycles: 3 channel orders × 2 cross-channel transforms × 2 spatial predictors × zlib/LZMA, on four scikit-image 128×128 photographs; 8 PNG/WebP lossless controls. Development images: astronaut, coffee. Locked test images: chelsea, rocket. Selected on development: RGB order + difference to green + vertical predictor + LZMA.

Development BLKH 56,260 bytes. Independent holdout BLKH 45,776 bytes, WebP lossless 39,316 bytes, PNG 51,363 bytes. BLKH **16.43% larger** than WebP. Negative result. Actual 16-byte BLKH container header included; reconstruction exact for all tested images. Timings and SHA-256 logged locally. No AVIF/JXL, external corpus, isolated RSS or lossy rate-distortion comparison. Do not claim superiority from PNG comparison.

Local artifacts: /mnt/data/blkh_exp262/bench.py, results.json, REPORT.md, exp262.yml. These paths are local to experiment runtime; benchmark script/results not yet uploaded to repository and CI cannot reproduce until committed. Reproduction requires numpy, pillow, scikit-image. No demo or LinkedIn publication warranted.
