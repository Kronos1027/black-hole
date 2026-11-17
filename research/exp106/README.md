# Exp106 — Reversible RGB decorrelation + Paeth (2026-11-17)
Following Exp103–105 negative grayscale results, test a **different hypothesis**: RGB channel decorrelation (identity, G/R-G/B-G, R/G-R/B-G) followed by optional per-plane Paeth and zlib. The decoder is standalone and lossless. The full 14-byte header and payload are counted. Every candidate is evaluated by actual compressed bytes.

Local Linux Python 3.11+ run with Pillow 12.3.0 and scikit-image 0.26.0. Three unittest cases passed (15 roundtrips plus transform and corruption tests). Five public scikit-image images, resized proportionally to fit 128×128, RGB8, with two development images and three exploratory holdouts (NOT independently sourced external corpus). Decoding matched source bytes exactly; baseline PNG and WebP lossless also decoded exactly.

| Image | Exp106 bytes | PNG bytes | WebP lossless bytes |
|---|---:|---:|---:|
| astronaut (dev) | 28190 | 31342 | 23230 |
| chelsea (dev) | 18694 | 21043 | 15266 |
| coffee (holdout) | 21024 | 21634 | 16400 |
| rocket (holdout) | 14602 | 14277 | 11776 |
| hubble_deep_field (holdout) | 22693 | 26448 | 18982 |

Exp106 beats PNG on 4/5 but **loses to WebP lossless on 5/5**. No competitive compression discovery. No AVIF or JPEG XL encoders installed in local environment; no lossy rate-distortion curves because prototype is strictly lossless. No independently sourced test corpus. Timing and process-wide peak RSS captured by bench.py but not a controlled benchmark; no CI run confirmed yet. The local test code was copied to this branch and needs independent CI validation.

Reproduce from repo root:
```bash
python -m pip install numpy pillow scikit-image
cd research/exp106
python -m unittest -v test_exp106.py
python bench.py > results.json
```
Next: evaluate independent image datasets and JXL/AVIF lossless when available, add genuine end-to-end process RSS and matched-speed comparisons, then explore context entropy coding only if a robust win is found on independent images. Do not promote demo or LinkedIn.
