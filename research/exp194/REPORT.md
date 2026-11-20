# BLKH Exp194 — reversible modular color lifting / predictor / entropy sweep

Date: 2026-11-20. Local execution; 216 lossless RGB encode/decode cycles, 216 deliberate CRC-header corruptions rejected. Eight scikit-image sources resized to 128x128; four dev and four holdout. 27 configurations: RGB, green decorrelation, modular YCoCg; none/horizontal/vertical difference; zlib9/LZMA6/BZ2. Selection locked on development images only.

Chosen: green/vertical/lzma6. Held-out complete file sizes: BLKH 81,108 bytes, WebP lossless method=6 71,990 bytes, PNG optimized 95,244 bytes. BLKH is **12.6656% larger than WebP**. Result negative. Note earlier Exp187 investigated similar green/adaptive combinations; this experiment adds modular YCoCg and BZ2, but is not a new breakthrough.

Local script/results: exp194.py and results.json produced in execution environment; not yet committed to GitHub, hence no reproducible Actions workflow. Blockers: externally sourced corpus, AVIF/JXL equivalent lossless comparisons, rate-distortion curves, isolated RSS, CI. Do not claim codec superiority from PNG comparison. No demo/LinkedIn warranted.
