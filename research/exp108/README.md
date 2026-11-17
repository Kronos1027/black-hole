# Exp108 — RGB reversible row-filter ablation (2026-11-17)

**Hypothesis:** PNG-style adaptive per-row residual filters (None, Sub, Up, Average, Paeth), optionally preceded by a reversible G/R-G/B-G transform, might outperform a global Paeth filter or improve Exp107. Each row filter selector is explicitly stored before its residuals; the entire stream is zlib(level 9) compressed. A 13-byte magic/dimensions/color-mode header is counted. No pretrained weights, side information, or original pixels are needed for decoding.

## Actual local execution
Python 3, Pillow 12.3, scikit-image 0.26; 7 public images resized to at most 96×96 RGB8. 2 unittest methods passed: 60 lossless roundtrips (5 sizes × 3 patterns × 4 configurations) plus 5 malformed-input checks. 28 additional image encode/decode benchmarks (7 × 4 variants) passed exact byte equality and baseline PNG/WebP lossless decode checks. An initial implementation failed a roundtrip because the inverse color transform used the wrong channel; it was fixed **before** measurements, and the passing tests rerun. The same-library holdouts are exploratory and not an independent external corpus.

| Image | PNG bytes | WebP lossless bytes | Fixed Paeth + decorrelation | Adaptive row + decorrelation |
|---|---:|---:|---:|---:|
| astronaut (dev) | 18886 | 14210 | 16997 | 16956 |
| chelsea (dev) | 12367 | 9094 | 11301 | 11292 |
| coffee (exploratory) | 12716 | 9742 | 12295 | 12230 |
| rocket (exploratory) | 8360 | 6988 | 8362 | 8354 |
| hubble_deep_field (exploratory) | 15308 | 10702 | 14069 | 13790 |
| immunohistochemistry (exploratory) | 20859 | 17046 | 19353 | 19325 |
| colorwheel (exploratory) | 7862 | 4578 | 7936 | 7885 |

Adaptive filters improve fixed Paeth modestly in all 7 examples (8–279 bytes), but **lose to WebP lossless in all 7**. This is a negative result for codec competitiveness, not a breakthrough. The 28 measurements are ablations, not 28 independent datasets. The benchmark process reported peak RSS of 129292 KiB on Linux; this includes Python, NumPy, Pillow, scikit-image and image-loading allocations, and is not per-codec incremental memory. Timing values are printed by bench.py; no statistically robust timing claims are made.

## Reproduce
```sh
python -m pip install pillow==12.3.0 scikit-image==0.26.0 numpy
cd research/exp108
python -m unittest -v test_exp108.py
python bench.py > results.json
```

CI workflow: `.github/workflows/exp108.yml` (manual dispatch and branch push). A passing Actions run must be verified independently; a green unit test alone does not establish a codec advantage.

## Limitations and next falsifiable hypotheses
- Only 8-bit RGB, tiny images and same-library exploratory examples. No independent external holdout; do not infer generalization.
- Neither AVIF nor JPEG XL was available in this local environment. The experiment is lossless; comparing to lossy codecs at unmatched quality is invalid.
- Compare against JPEG XL lossless and AVIF lossless (where supported), with exact reconstruction and all bytes included; run a separate matched-quality RD study for lossy methods.
- A row-wise heuristic based on absolute signed residual is not entropy-optimal; explore learned entropy contexts only if justified by independent benchmarks, not repeated tuning on the same seven images.
- The decoder bounds output pixels to 100 million but is a research prototype, not hardened for hostile input.
- No demo or LinkedIn superiority claims are warranted.
