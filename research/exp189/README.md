# Exp189 — modular lifting × MED (2026-11-20)

Local experiment: 144 RGB lossless roundtrips and corruption checks on 8 scikit-image sample images resized to 96×96. First four development, last four held-out. Compared RGB, green decorrelation, modular YCoCg-inspired lifting × Left/Paeth/MED × zlib/LZMA. Development-selected green/MED/LZMA: held-out **28,228 complete bytes**, WebP lossless **21,200 bytes**, BLKH **33.151% larger**. Negative result. No claim of superiority.

Reproducible local bundle generated in this run: `BLKH_exp189_bundle.zip` (experiment.py, results.json, workflow proposal). **Bundle code and results are not in this GitHub branch**; Actions not run. No AVIF/JXL, external photographic dataset, rate-distortion curves or per-codec isolated RAM. Shared process peak 145,728 KiB. Decoder SHA-256 and corruption checks passed locally. The lifting is modular and is not standard signed YCoCg-R. Do not publish demo or LinkedIn.
