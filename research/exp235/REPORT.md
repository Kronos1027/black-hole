# Exp235 — reversible channel/vertical transforms, codec ablation (2026-11-22)

Status: **negative, exploratory**. Local executed benchmark: `/mnt/data/exp235/bench.py`, raw output `/mnt/data/exp235/results.json`. **These files are not yet committed and local paths are not durable**; reimplement or recover before claiming CI reproducibility.

## Design
Six 128×128 RGB inputs from scikit-image built-ins (astronaut, coffee, gradient = development; chelsea, rocket, camera = held-out test). No external independent corpus. 4 transforms (identity, green-difference modulo 256, above-row difference modulo 256, combined) × 3 entropy backends (zlib-9, LZMA-6, bzip2-9). Each bitstream includes a real 14-byte header: 4 magic, 8 dimensions, 2 transform/codec selectors. Decoder consumes bitstream and reconstructs RGB exactly; 72/72 experimental roundtrips pass, plus 12/12 PNG/WebP lossless controls.

Winner selected **only on development**: green_up_lzma. Held-out totals: **56,410 B** vs **47,652 B** WebP lossless and **67,502 B** PNG; **18.38% larger than WebP**. No superiority claim.

Measured wall-clock encode/decode per sample and Python tracemalloc peak in local JSON, but tracemalloc is **not** isolated process RSS. Pillow WebP configured lossless quality=100 method=6; PNG compress_level=9. Tests used 128×128 resampled source images; no lossy rate-distortion curve, no AVIF/JPEG XL, no independent dataset or Actions run.

## Next steps
1. Commit actual benchmark script and JSON (not done in this commit); add Actions workflow to run deterministic roundtrip and byte checks.
2. Freeze larger external photographic and synthetic corpora with licenses/hashes and split IDs before tuning.
3. Add isolated RSS, codec version capture, AVIF/JXL lossless, and fair lossy RD comparisons with decoded image PSNR/SSIM.
4. Investigate entropy context coding only after above reproducibility gate.

No demo or LinkedIn release warranted.