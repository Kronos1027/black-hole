# BLKH Exp147 — Bounded-error quantization vs modern codecs

## Experimental hypotheses
Uniform 8-bit RGB scalar quantization (step 1/3/7/15/31) vs 16x16 adaptive edge-aware step selection; six reversible predictors+entropy backends.
Full on-disk B147 header, mode, quantizer, side-map, and compressed payload counted. Reconstruction is bounded to <= floor(step/2) absolute RGB error; step=1 exact.

## Separation
Development: camera, coins, clock, grass. Holdout: gravel, moon, retina, cell, cat, horse. All original skimage data resized to 192x192 LANCZOS RGB.
Mode and quantizer selected only from development images at 40 dB (interpolated rate); holdout untouched for selection.

## Selection and outcome
Selected: `('green_left', 'brotli')` adaptive=False. Self-tests: {'valid_roundtrips': 162, 'malformed_rejected': 6}.
Holdout lossless full-file totals: {'BLKH': 210333, 'WEBP': 150116, 'JPEGXL': 146460}.

## Rate-distortion (interpolated bytes at equal PSNR)
| Target PSNR | Codec | Coverage | BLKH bytes | Baseline bytes | BLKH excess |
|---:|---|---:|---:|---:|---:|
| 35.0 dB | WEBP | 3/6 | 60399.8 | 21362.4 | 182.74% |
| 35.0 dB | AVIF | 4/6 | 63110.1 | 21940.9 | 187.64% |
| 35.0 dB | JPEGXL | 3/6 | 60399.8 | 24204.8 | 149.54% |
| 40.0 dB | WEBP | 5/6 | 77679.4 | 33066.7 | 134.92% |
| 40.0 dB | AVIF | 5/6 | 77679.4 | 33118.3 | 134.55% |
| 40.0 dB | JPEGXL | 6/6 | 97719.1 | 51792.9 | 88.67% |
| 45.0 dB | WEBP | 4/6 | 55873.0 | 31123.7 | 79.52% |
| 45.0 dB | AVIF | 4/6 | 55873.0 | 31443.1 | 77.7% |
| 45.0 dB | JPEGXL | 6/6 | 121580.7 | 67334.8 | 80.56% |

## Integrity
162 randomized/edge self-test roundtrips plus 540 full-image encode/decode validations; 6 malformed files rejected.
All candidate RGB images decoded and max-error checked. WebP/AVIF/JXL decoded and evaluated on actual RGB outputs; JXL distance=0 pixel-exact.

## Known limits
- Bundled scikit-image images and resized 192x192; not external unseen corpus
- Independent image identities between dev/test but some may appear in historical BLKH experiments
- AVIF and WebP lossy encode settings are not quality-equivalent by number; PSNR matching is used
- RD interpolation is an estimate, not a bitstream actually encoded at target PSNR
- Process RSS includes Python runtime and all codec dependencies
- Local run only unless exact GitHub Actions run explicitly confirmed

## Reproduce
`python research/exp147_rd/experiment.py --out research/exp147_rd/outputs`
Requires Python packages numpy, pillow[AVIF support], scikit-image, brotli and ffmpeg compiled with libjxl.
Environment: {'python': '3.13.5', 'numpy': '2.3.5', 'pillow': '12.3.0', 'libwebp': '1.6.0', 'brotli': '1.2.0', 'ffmpeg': '/usr/bin/ffmpeg', 'skimage_version': '0.26.0', 'system': 'Linux-6.18.57-x86_64-with-glibc2.41', 'size': 192, 'resampler': 'Pillow.LANCZOS', 'ffmpeg_libjxl_available': True, 'AVIF_encoder': True, 'max_process_rss_kib': 177452, 'data_source': 'scikit-image bundled images; no external download'}