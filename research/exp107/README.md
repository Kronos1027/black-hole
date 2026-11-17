# Exp107 — adaptive per-plane reversible filters (2026-11-17)

## Hypothesis
Unlike Exp106 (one filter choice shared across all RGB planes), choose among none/left/up/Paeth independently per plane, with a reversible RGB decorrelation (identity or G,R−G,B−G). For each plane, select the filter with minimum zlib(level 6) compressed bytes, then concatenate all filtered planes into one zlib(level 9) stream. Format overhead is 16 bytes (magic, dimensions, color mode, 3 filter selectors). Decoder is self-contained.

## Actual local exploratory execution
Python stdlib + Pillow + NumPy + scikit-image. Seven 96-pixel-wide RGB examples, downscaled with Pillow thumbnail. Two development examples, five exploratory holdouts from the **same library**, therefore NOT independently sourced. All seven encode/decode roundtrips byte-exact; PNG and WebP lossless roundtrips verified. Two unittest methods passed, covering 15 roundtrips and five corrupt-stream cases.

| Image | Exp107 bytes | PNG bytes | WebP lossless bytes | Encode s | Decode s |
|---|---:|---:|---:|---:|---:|
| astronaut (dev) | 16966 | 18886 | 14210 | 1.2154 | .0165 |
| chelsea (dev) | 10864 | 12367 | 9094 | .7515 | .0105 |
| coffee (holdout) | 12292 | 12716 | 9742 | .8142 | .0160 |
| rocket (holdout) | 8481 | 8360 | 6988 | .7844 | .0119 |
| hubble_deep_field (holdout) | 12857 | 15308 | 10702 | 1.0239 | .0142 |
| immunohistochemistry (holdout) | 19156 | 20859 | 17046 | 1.2842 | .0176 |
| colorwheel (holdout) | 6210 | 7862 | 4578 | 1.2108 | .0187 |

**Negative result:** Exp107 loses to WebP lossless in all 7 cases, despite beating PNG in 6/7. It is not a competitive codec. Time is wall-clock, not rigorously controlled. Python tracemalloc allocation peaks 365883–405835 bytes, **not process RSS**. AVIF/JPEG XL unavailable locally; no fair lossy RD curve can be claimed from a lossless-only prototype.

## Reproducibility / limitations
The prototype and test script were executed in a separate local container, not yet uploaded to this branch; this report records the measurements and methodology but **does not establish GitHub Actions reproducibility**. The local artifact source must be committed before CI can validate the numbers. Image data is publicly bundled in scikit-image and not a truly independent external benchmark. The comparison is not controlled for encode speed; WebP method=6 quality=100 lossless, PNG compress_level=9. Next run should commit code and CI, obtain independent images and compare with JPEG XL/AVIF lossless where supported. Do not publish demo or LinkedIn claims.
