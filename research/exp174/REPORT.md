# BLKH Exp174 — row/channel adaptive predictor, one global entropy stream (2026-11-19)

Continuation of Exp172/173. Experiment run locally using Python 3.13.5, numpy 2.3.5, Pillow 12.3.0; 8 bundled skimage 64x64 RGB samples (not external independent data).

Hypotheses: adaptive filter per channel/row versus row-level absolute-residual or entropy selector versus fixed Paeth; RGB vs green-subtracted channels; zlib-9 vs LZMA-6. One global compressed stream per image (avoids independent block compression overhead). 32 configs/image; 256 encode/decode cycles with pixel-exact reconstruction, 256 CRC tamper rejection checks. BLKH byte counts include 14-byte header, payload, 4-byte CRC and embedded selector bytes.

Dev samples: astronaut, coffee, chelsea, rocket. Selected **gsub / row_abs / LZMA** exclusively on dev: 28072 bytes vs WebP lossless 25076 (+11.95%). Heldout samples: hubble_deep_field, immunohistochemistry, colorwheel, brick: BLKH **22932 bytes** vs verified WebP lossless **19170 bytes** (+19.62%). PNG lossless heldout 26968 bytes, not a competitive baseline by itself.

Peak shared process RSS 147948 KiB; no isolated codec RSS. No external photographic datasets, no AVIF/JPEG XL lossless benchmarks, no rate-distortion curves, no Actions reproduction. **Negative finding**: row/channel selection does not beat WebP lossless. No scientific breakthrough or basis for demo/LinkedIn.

Reproduction artifacts (experiment.py, results.json, README and workflow proposal) generated in local BLKH_exp174_bundle.zip. GitHub report is committed but code/results have not been uploaded; CI not run. Next step: upload reproducible artifacts and run independent datasets with exact lossless codec comparisons. No prior branches or data modified.
