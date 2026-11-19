# BLKH Exp159 — reversible residual layout (2026-11-19)

Hypotheses: channel-major reordering and bit-plane shuffle of spatial residuals before entropy coding. Experiment locally executed on 6 scikit-image RGB images resized to 64x64, split before selection into 3 development and 3 held-out evaluation images.

24 configurations = 4 predictors (none/up/Paeth/adaptive_sad) x 3 layouts (identity/channel planes/bitshuffle) x 2 compressors (zlib-9/LZMA-6). All 144 complete BLKH files decoded bit-exact, with 288 corruption rejections; 6 WebP lossless baselines decoded bit-exact.

Development selection: adaptive_sad + identity + LZMA-6, 24,872 bytes. Held-out evaluation: BLKH 22,128 bytes versus WebP lossless 18,120 bytes (+22.12% larger). Both new layout hypotheses failed to win selection. Full BLKH file bytes include header and CRC32. Single-run timing and process-wide RSS are not controlled benchmarks.

LIMITATIONS: No independent external dataset, no AVIF/JPEG XL, no lossy rate-distortion curves, no confirmed GitHub Actions reproduction. Executable experiment and results were generated locally, not yet committed. Experiment imports helpers from Exp158, so CI must vendor/restore that dependency before execution.

Verdict: negative; no validated competitive advance. Do not announce a scientific breakthrough, public demo claim or LinkedIn performance claim.