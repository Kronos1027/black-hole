# BLKH Exp165 — nibble-plane residual packing

Hypotheses: reversible separation of high/low nibbles of RGB bytes, with/without vertical byte predictor. Compared raw/up/nibble/up_nibble × zlib-9/LZMA-6 on 3 development and 4 holdout images, all RGB 128×128. Holdout includes deterministic random noise (seed 165), not an independent external photographic dataset.

All 56 BLKH encode/decode cycles reconstructed exactly; 56 header corruption probes rejected. Seven WebP lossless cycles decoded exactly. BLKH measurements include the entire serialized file: 20-byte header, compressed payload and CRC32.

Selected exclusively on development: up+LZMA6, 87,784 bytes versus WebP 69,052. Holdout: BLKH 131,244 bytes versus WebP 116,746 (+12.42%). Nibble-plane hypotheses did not win development selection. Negative result.

Python 3.13.5; Pillow WebP available; peak process RSS 143,212 KiB, not isolated per codec. Timing exploratory, not statistically repeated. No independent external photographs, AVIF/JXL, rate-distortion curves or GitHub Actions verification. Code and JSON results remain in local Exp165 bundle, not in this report-only commit. Reproduce with experiment.py (numpy, pillow, scikit-image), seed 165.

No competitive discovery; no scientific demo or LinkedIn publication.