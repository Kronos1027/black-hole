# Exp112 — Context-conditioned Paeth residual streams (2026-11-17)

Hypothesis: split reversible RGB Paeth residuals into 1, 2 or 4 entropy streams based on the absolute difference between already decoded left and upper samples. No context map is stored: the decoder reproduces context decisions from reconstructed neighbors. Both zlib-9 and LZMA-6 tested. Total encoded bytes include 14-byte header and 4 bytes per stream length.

## Local execution

Python 3, Pillow 12.3.0, scikit-image 0.26.0, NumPy 2.3.5. Two unittest methods passed: 90 roundtrips (5 dimensions × 3 input patterns × 3 context counts × 2 backends), plus malformed inputs. Seven scikit-image RGB samples scaled to <=96 pixels on either side, all byte-exact reconstructed. Local runs, not independent CI.

| Sample | 1 context LZMA | 2 contexts LZMA | 4 contexts LZMA | WebP lossless |
|---|---:|---:|---:|---:|
| astronaut | 18110 | 18294 | 18582 | 14726 |
| chelsea | 12006 | 12130 | 12338 | 9434 |
| coffee | 12510 | 12650 | 12794 | 10110 |
| rocket | 8234 | 8294 | 8458 | 7150 |
| hubble_deep_field | 15754 | 15814 | 15934 | 11248 |
| immunohistochemistry | 20810 | 20946 | 21162 | 17648 |
| colorwheel | 6578 | 6614 | 6686 | 4700 |

**Negative finding:** context splitting with LZMA loses to one stream on 7/7 images and to WebP on 7/7. zlib results also included in the benchmark script output. This is an exploratory, previously reused image set, not independent validation. PNG and WebP were verified byte-exact. Lossy rate-distortion is inapplicable to this lossless-only experiment; AVIF and JPEG XL not measured. Peak RSS and single-run timings are emitted by the script, but process RSS includes Python and dependencies and single-run timing is noisy.

Run:
```sh
python -m pip install pillow numpy scikit-image
cd research/exp112
python -m unittest -v test_exp112.py
python bench.py > results.json
```

CI reproducibility requires observing a successful Actions run. Next prioritize independent datasets and fundamentally different entropy models rather than further tuning context thresholds on reused images. No competitive discovery; do not announce a breakthrough.
