# Exp123 — Original RGB reversible transforms (2026-11-17)

Status: negative exploratory local benchmark; NOT independently reproduced on GitHub Actions.

## Motivation
Exp120–122 quantized images to 16 grayscale levels, so comparisons did not represent lossless compression of original photographs. Exp123 avoids quantization and works on exact original RGB pixels. This synthetic experiment is not yet an external photographic benchmark.

## Protocol
Four byte-exact reversible transforms: raw RGB, channel differences R-G/G/B-G modulo 256, horizontal delta modulo 256, vertical delta modulo 256. Each is compressed with zlib level 9 or LZMA preset 6. Full BLKH container: four-byte magic, height, width, transform ID, coder ID, four-byte compressed length and compressed payload (14 bytes header). 96x96 RGB, deterministic NumPy PCG64 RNG. Families: noise, gradient, 12x12 random-color blocks upsampled nearest-neighbor, 12x12 random RGB upsampled bilinearly using Pillow. Dev seeds 111,112; holdout 9111,9112. Select one configuration globally on dev aggregate bytes; no test-time optimization. WebP lossless method 6 and PNG compress_level 9 via Pillow, exact RGB roundtrip validated.

## Local observed results
Locked dev winner: vertical delta + LZMA-6. 128 exact BLKH roundtrips (8 configs x 16 samples), WebP and PNG roundtrip on all 16 samples, three malformed containers rejected. Holdout eight samples: BLKH 71,868 bytes; WebP 71,518 bytes; PNG 85,292 bytes. BLKH beat WebP on 5/8 images, but lost in aggregate by 350 bytes (0.49%). Family totals (BLKH vs WebP): noise 55,444 vs 55,512; gradient 844 vs 136; blocks 1,388 vs 1,442; bilinear smooth 14,192 vs 14,428. The tiny wins on noise and some synthetic families are not evidence of better compression of photographs. Header and compressed payload included.

## Reproducibility blockers
Experiment script and JSON were executed and generated locally at /mnt/data/exp123/experiment.py and /mnt/data/exp123/results.json, but have NOT been transferred to GitHub. Container GitHub network DNS unavailable; connector cannot directly access local container files. Therefore no CI workflow or independent replication is claimed. Timings are recorded per configuration in local JSON, but no isolated RSS measurements or AVIF/JPEG XL baselines were obtained. These deficiencies prohibit any competitive claim.

## Next
Transfer exact source/JSON, add GitHub Actions with pinned dependencies and SHA256 assertions, test an independent photographic corpus at native resolution, add lossless AVIF/JPEG XL and rate-distortion curves for lossy modes, and measure codec wall-time and peak RSS separately. No demo or LinkedIn post.
