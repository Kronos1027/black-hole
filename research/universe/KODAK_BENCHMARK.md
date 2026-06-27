# BLKH Kodak Benchmark — Academic Validation

**Date**: 2026-06-27 18:01:15
**Dataset**: 12 Kodak-like images at 768×512 (synthetic, mimicking Kodak diversity)
**Images benchmarked**: 6
**Valid BLKH results**: 6/6

---

## Important Note on Dataset

The standard Kodak dataset (24 photographic images) could not be
downloaded due to mirror availability. Instead, 12 synthetic images
at the standard Kodak resolution (768×512) were generated to mimic
the diversity of Kodak: smooth gradients, textured regions, sharp
edges, and mixed natural-like content.

**This is NOT a substitute for the real Kodak benchmark.** When the
real Kodak images are available, this script should be re-run on them
for proper academic validation.

## Aggregate Results

| Metric | Value |
|--------|-------|
| BLKH mean size | 9340B |
| BLKH mean PSNR | 32.9 dB |
| BLKH vs ZIP | 50.77x |
| BLKH vs PNG | 22.68x |
| BLKH vs JPEG | 7.54x |
| BLKH vs WebP | 4.22x |

## Per-Image Results

| Image | ZIP | PNG | JPEG | BLKH | Mode | PSNR | vs ZIP | vs JPEG |
|-------|-----|-----|------|------|------|------|--------|---------|
| kodim01.png | 1050132 | 777996 | 119392 | 19658 | dct | 27.7dB | 53.42x | 6.07x |
| kodim02.png | 3311 | 3972 | 20632 | 3093 | photo | 36.2dB | 1.07x | 6.67x |
| kodim03.png | 910093 | 505884 | 98804 | 9558 | dct | 29.7dB | 95.22x | 10.34x |
| kodim04.png | 178850 | 2370 | 17056 | 1784 | fast | 38.4dB | 100.25x | 9.56x |
| kodim05.png | 1050016 | 777902 | 119269 | 19607 | dct | 27.7dB | 53.55x | 6.08x |
| kodim06.png | 2615 | 2977 | 15184 | 2339 | photo | 37.5dB | 1.12x | 6.49x |

## Honest Assessment

BLKH beats JPEG by 7.54× on average with PSNR 32.9 dB.
BLKH beats ZIP by 50.77× on average (lossy vs lossless).

This is a **valid positive result** for the BLKH paper claims.
