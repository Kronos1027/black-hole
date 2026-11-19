# BLKH Exp155 — reversible YCoCg-R versus green-delta (2026-11-18)

Experiment conducted locally with 8 packaged scikit-image images resized to RGB 128x128. Fixed 4 development and 4 holdout images. Tested 3 reversible channel transforms (RGB, green-delta, integer YCoCg-R), 3 spatial predictors (none, left, up), and 2 entropy codecs (zlib-9, LZMA-6): 18 configurations, 144 encode/decode cycles, all reconstructed RGB exactly. The YCoCg-R Co/Cg channels were stored as signed int16; this larger representation is part of the measured file size. Full payload and metadata counted.

**Chosen on development only:** green/up/LZMA6.
Development: BLKH 98,872 bytes; WebP lossless 82,660 bytes.
Holdout: BLKH 90,820 bytes; WebP lossless 76,672 bytes; PNG 103,652 bytes. BLKH is **18.45% larger than WebP**.
YCoCg-R/up/LZMA6 holdout: 104,220 bytes. Hypothesis failed.

Limitations: images are packaged scikit-image samples, not external independent dataset. No AVIF/JPEG XL in this run; no lossy rate-distortion; no measured peak RSS; no GitHub Actions verification. This is a negative local result, not a discovery.

Local reproducible artifacts:
- experiment.py SHA256 c0020c89d1e993ec32cd24fad63bee78f09b2cb212ad25c159aa31445f355c4f
- results.json SHA256 1e27fa484740c80581a96acf5885c2d41a9d229651fa2d4cfdfdff40269db4a3

The executable code and JSON remain local pending upload. Do not claim CI reproducibility, prepare demo or publish LinkedIn. Next: commit scripts/results, verify CI, use external dataset and fair AVIF/JXL comparisons.