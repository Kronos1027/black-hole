# Exp111 — Independent channel-stream entropy ablation (2026-11-17)

Exp110's planar byte layout occasionally improved LZMA, but still lost to WebP lossless on every sample. Exp111 tests a distinct hypothesis: whether **separate entropy streams for each Paeth residual channel** overcome the overhead of three compressed streams. Four alternatives: one global zlib-9, one global LZMA-6, three channel-wise zlib-9, and three channel-wise LZMA-6. Every stream length and the 14-byte header count toward total size. RGB decorrelation is reversible; decoder requires no original image.

## Executed locally
Two unittest methods passed, covering **60 explicit mode/backend exact roundtrips**, 15 best-choice roundtrips, and five invalid-input cases. Seven 8-bit RGB examples from scikit-image were resized to maximum 96 pixels on either axis. PNG and WebP lossless roundtrips were checked using Pillow. Numbers below include complete files, in bytes.

| Image | Exp111 best | PNG | WebP lossless | selected |
|---|---:|---:|---:|---|
| astronaut | 15318 | 18294 | 13652 | global LZMA |
| chelsea | 9726 | 11900 | 8648 | channel LZMA |
| coffee | 11018 | 12142 | 9166 | global LZMA |
| rocket | 7006 | 7687 | 6416 | global LZMA |
| hubble_deep_field | 11730 | 14151 | 9908 | global LZMA |
| immunohistochemistry | 17390 | 19964 | 16068 | channel LZMA |
| colorwheel | 6894 | 7649 | 4402 | global LZMA |

Channel separation won only 2/7; **WebP lossless won all seven**. This is a negative competitive result. Samples overlap previous experiments, so the holdout designation in bench.py means only *not used to choose this experiment's hypothesis*, not independent external evaluation. Reported timings are single-run and not statistically stable. Local peak RSS was measured in the benchmark JSON, but includes Python, Pillow, numpy and scikit-image.

## Reproduce
```sh
python -m pip install pillow numpy scikit-image
cd research/exp111
python -m unittest -v test_exp111.py
python bench.py > results.json
```
Do not confuse better-than-PNG with state-of-the-art performance. AVIF/JPEG XL lossless, independent external images, repeated timings, and fair matched-distortion lossy comparisons remain untested. CI workflow needs a confirmed successful run before claiming independent reproducibility. No breakthrough, demo, or public superiority announcement is warranted.
