# BLKH Exp188 — Independent-tile adaptive spatial predictors (2026-11-20)

Status: **negative; no competitive breakthrough**.

## Hypothesis
Independently compressing 16/32/64/128-pixel tiles with a locally selected predictor (none, left, up, Paeth) might outperform fixed Paeth after paying full side-information and stream overhead.

## Execution
Executed locally with Python, Pillow, NumPy, scikit-image; 8 independent source images from scikit-image (astronaut, coffee, chelsea, rocket, camera, coins, moon, page), resized to RGB 128x128. First four assigned to development, remaining four held out. Each of 16 variants per image (4 tile sizes × adaptive/fixed × zlib/LZMA) was encoded and decoded, with byte-exact reconstruction asserted. Each tile carries a 5-byte predictor/length header, with a 10-byte file header. Reference: Pillow WebP lossless method=6, decoded byte-exact. Timing collected per variant, but RAM not isolated.

## Selection and holdout
Selected exclusively by total bytes on development: `128_adaptive_lzma`. On holdout: BLKH **51,208 bytes**, WebP lossless **36,014 bytes**, BLKH **42.19% larger**. Note that tile=128 means the selected model effectively uses one whole-image tile; this is *not* evidence of a local adaptation advantage.

## Limits / next work
The executable script and machine-readable JSON were generated locally but are not yet committed here; Actions reproduction is **not validated**. No independent external photographic test corpus, JPEG XL, AVIF lossless, isolated memory accounting, or matched lossy rate-distortion curves. Do not claim codec superiority, demo readiness, or a LinkedIn result.

Follow-up: commit executable harness and fixture hashes, implement CI workflow, compare against JPEG XL and AVIF lossless, isolate encode/decode RSS, add untouched external corpus, then investigate shared entropy coding across tile boundaries to reduce header/stream overhead.
