# Exp113 — independent synthetic holdout (2026-11-17)
Built from Exp112 without changing prior experiments. Hypotheses: left, upper, Paeth, and mean spatial predictors; zlib-9 vs LZMA-6. File sizes include 18-byte BLKH header. RGB8 lossless only.

Local Python unittest: 2 test methods passed, including 72 roundtrips (3 sizes × 3 seeds × 4 predictors × 2 entropy backends) and malformed headers. Benchmarks: 16 deterministic synthetic 96×96 RGB images, split by seeds into development (1,2) and holdout (991,992), four image families (gradient, blocks, noise, texture). All encoded BLKH, PNG and WebP files were decoded and checked byte-exact.

**Development** best-of-eight BLKH versus WebP lossless bytes:
gradient 219/62, 217/62; blocks 132/114, 132/114; noise 27682/27754, 27682/27752; texture 15914/14934, 15898/15108.
**Holdout** gradient 234/66, 230/66; blocks 132/114, 132/114; noise 27682/27750, 27682/27750; texture 15870/14940, 15894/14918.
BLKH best-of-eight wins 2/8 in both splits, only for incompressible noise, by ~68 bytes, likely format/header overhead rather than meaningful modeling advantage. **Important** best-of-eight is oracle per-image selection; an actual deployable encoder would need to encode all variants and include selection overhead; results do not demonstrate competitive generalization. The two splits share image-generating families and are not external photographic datasets. This is not a scientific breakthrough.

Reproduce:
```sh
python -m pip install pillow
cd research/exp113
python -m unittest -v test_exp113.py
python bench.py > results.json
```
Timing, image SHA256 and process peak RSS are emitted in JSON; RSS includes Python/Pillow. No AVIF/JPEG XL or fair lossy rate-distortion yet. Successful GitHub Actions execution not yet confirmed. Do not announce demo/LinkedIn.
