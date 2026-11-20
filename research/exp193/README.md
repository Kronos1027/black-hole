# BLKH Exp193 — reversible layout and Brotli

Experimented locally on 2026-11-20 with 12 reversible transforms (RGB/green decorrelation, horizontal/vertical delta, planar/interleaved) and four compressors (Brotli q5/q9, zlib9, LZMA6). 384 exact encode/decode checks on eight 128x128 scikit-image samples, with CRC corruption checks. Four development and four reserved test images; selected green_vdelta_planar/LZMA6 only on development.

Holdout complete bytes: BLKH 28,504 vs WebP lossless 25,898 (+10.06% BLKH). Negative result. Full executable code and JSON records were produced locally but are NOT in this commit. No external independent photographic dataset, AVIF/JXL, isolated RAM or GitHub Actions run. Do not interpret this as competitive compression or a scientific breakthrough.

Reproduction package: BLKH_exp193_bundle.zip generated in the automation working environment; it contains experiment.py, results.json, README.md and proposed ci.yml. The workflow is only proposed, not executed. No demo/LinkedIn publication.