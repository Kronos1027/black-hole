# BLKH Exp152 — reversible bitplane transforms (2026-11-18)

## Hypothesis and prior art
Following Exp146 (spatial/color predictors), Exp147 (bounded error), and Exp149–151 (DCT), test an orthogonal **lossless** representation: bitplane transposition and adjacent-plane XOR before entropy coding. This is not a claim of novel codec invention.

## Executed local protocol
- 8-bit grayscale, 128×128, scikit-image bundled images.
- Development: camera, coins, moon. Locked holdout: page, grass, gravel.
- Six candidate configurations: raw/plane/xor × zlib-9/LZMA-6.
- Full ten-byte BLKH header included; each candidate encoded and decoded, byte-for-byte equality asserted.
- Independent real WebP lossless and JPEG XL lossless files created with FFmpeg, decoded and compared pixelwise.
- Selection solely by aggregate bytes on development images.

## Results
Selected `raw_lzma` (the **untransformed baseline**, not either proposed transform). On the holdout:
- BLKH: **38,430 bytes**
- WebP lossless: **36,176 bytes** (BLKH +6.23%)
- JPEG XL lossless: **35,479 bytes** (BLKH +8.32%)

All 36 BLKH encode/decode roundtrips were exact. No claimed improvement over modern codecs. Results are a negative test of both bitplane hypotheses.

## Limitations and reproducibility
These six scikit-image images are bundled reference samples, not an externally sourced independent dataset. Comparisons are lossless only, so no lossy rate-distortion curve is claimed. RSS is whole-process, not isolated codec peak. Local DNS resolution for github.com failed, so the local executable `/mnt/data/exp152/experiment.py` and JSON `/mnt/data/exp152/results.json` were not uploaded to this branch. No GitHub Actions run exists for Exp152. A subsequent run should publish the script and lock dependency versions before triggering CI. Avoid using this report as evidence of scientific superiority.

Decision: **negative result, no demo, no LinkedIn**.
