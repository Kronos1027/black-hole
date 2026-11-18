# Exp139 — per-image RGB anchor selection (2026-11-18)

Status: NEGATIVE; no competitive compression claim.

Hypotheses: select anchor channel from R/G/B, and choose no predictor vs vertical predictor, using reversible modulo-256 channel differences, planar packing and LZMA-6. Each candidate has a 14-byte header (magic, width, height, anchor, predictor). All 48 encode/decode roundtrips reconstructed the RGB arrays exactly. WebP lossless files were independently decoded and checked.

Dataset: skimage astronaut, coffee, chelsea, rocket (development); camera, coins, grass, gravel (test). Original resolutions. **These image identities were used previously, so this is NOT a genuinely independent external evaluation.**

| Split | Adaptive BLKH | Fixed green+vertical | WebP lossless | PNG |
|---|---:|---:|---:|---:|
| Development | 1,245,304 | 1,245,304 | 1,072,790 | 1,394,553 |
| Test | 629,308 | 629,308 | 591,730 | 1,052,197 |

Adaptive selection chose green+vertical for every image, hence no gain over fixed. Test BLKH is 6.35% larger than WebP lossless. This is a negative result, and PNG comparison is not evidence of competitiveness.

Environment: Python 3, NumPy 2.3.5, Pillow 12.3.0, scikit-image 0.26.0. Encoder timings were collected per candidate in local JSON, but RAM not isolated. AVIF/JPEG XL unavailable; no rate-distortion curves. GitHub Actions has not reproduced this run. Local script/results: /mnt/data/exp139/experiment.py and results.json (local runtime only, not uploaded).

Next: upload script and JSON to this branch; add CI; external untouched photo datasets; AVIF/JXL lossless comparison; separately test lossy rate-distortion. No demo/LinkedIn.
