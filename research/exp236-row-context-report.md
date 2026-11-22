# Exp236 — row/channel adaptive predictors (2026-11-22)

Executed local experiment (script and machine-readable results generated locally, not yet committed): 6 skimage RGB images, deterministic 128x128 center crops. Development: astronaut, coffee, chelsea. Held-out: rocket, camera, colorwheel. 6 methods/image = 36 BLKH encode/decode cycles, plus 12 PNG/WebP lossless encode/decode controls. Every RGB reconstruction exact.

Tested RGB and reversible green-difference color transforms, with fixed vertical, fixed horizontal, or adaptive per-row/channel predictors selected by absolute modular residual proxy; LZMA preset 6. Full serialized experimental stream: 11-byte header, adaptive selector bytes, compressed residual. Adaptive selection is not entropy-optimal and selector overhead is included.

Winner selected strictly on development: green_adaptive. Held-out test totals: 38,193 bytes BLKH, 31,000 bytes WebP lossless, 45,041 bytes PNG. BLKH is 23.20% larger than WebP. Negative competitive result. No independent external corpus or JPEG XL/AVIF comparison; no GitHub Actions reproduction; no process peak RSS measured. Test images include grayscale replicated into RGB; corpus is tiny and not representative.

Local artifacts: /mnt/data/exp236/bench.py and /mnt/data/exp236/results.json. Reproduction requires Python 3, numpy, Pillow, scikit-image. Execute: python bench.py. The code and JSON must be uploaded to this branch before claiming CI reproducibility. No publication/demo justified.
