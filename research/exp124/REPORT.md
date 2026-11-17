# Exp124 — photographic RGB predictor holdout (2026-11-17)

**Negative exploratory result.** Local experiment; no Actions reproduction yet. This is NOT a competitive compression claim.

## Why
Exp123 used only synthetic RGB. Exp124 uses eight distinct public scikit-image source images, separated by image identity into four development and four holdout images, not merely different seeds of the same generator. All inputs are RGB 128×128 derived with Pillow LANCZOS resize; the exact resized RGB input is reconstructed byte-for-byte, but results are NOT native-resolution source-file lossless.

## Tested
Five reversible byte transforms: raw RGB, horizontal delta, vertical delta, per-channel Paeth, and reversible green channel difference; each with zlib-9 and LZMA-6. Ten configs; one global choice based only on aggregate development bytes. Complete BLKH container includes 14-byte header and payload length. Every one of the 80 encode/decode roundtrips was exact. WebP lossless method 6 and PNG level 9 were independently decoded and compared to RGB inputs on all eight images. Three invalid containers rejected. Source images: dev astronaut, coffee, chelsea, rocket; holdout camera, hubble_deep_field, immunohistochemistry, colorwheel.

## Results
Locked choice: vertical delta + LZMA-6. All numbers are complete file bytes.

| Split | Image | BLKH | WebP lossless | PNG |
|---|---|---:|---:|---:|
| dev | astronaut | 29582 | 24062 | 32494 |
| dev | coffee | 31046 | 24392 | 32709 |
| dev | chelsea | 29758 | 23114 | 31406 |
| dev | rocket | 19966 | 18178 | 21607 |
| holdout | camera | 9898 | 8596 | 16933 |
| holdout | hubble_deep_field | 29678 | 22604 | 31832 |
| holdout | immunohistochemistry | 35790 | 29998 | 37667 |
| holdout | colorwheel | 8918 | 7826 | 12131 |
| **holdout total** | | **84284** | **69024** | **98563** |

BLKH is 22.1% larger than WebP lossless on the holdout aggregate. Beat PNG in all four holdout images, which is not sufficient evidence of competitive superiority. No favorable result in any of the four holdout images versus WebP. Image resize and mixed photographic/nonphotographic sources limit external validity.

## Reproducibility and environment
Local Python 3.13.5, NumPy 2.3.5, Pillow 12.3.0, scikit-image 0.26.0. Process-wide peak RSS 147764 KiB; not isolated codec RAM. Per-config encode times, full RGB SHA256, all ten compressed sizes and selected mode recorded in local results JSON. Local files: /mnt/data/exp124/experiment.py and /mnt/data/exp124/results.json. **These files are not yet committed to GitHub**, so Actions validation is NOT claimed. The GitHub connector cannot directly read local filesystem files. The full script and JSON must be transferred into this branch, then CI with pinned dependencies and checksum assertions added. No AVIF/JPEG XL executables were available; no rate-distortion curves, no native-resolution external photo corpus.

## Next
Transfer executable + JSON to repository, add Actions; evaluate original-resolution externally licensed corpus; benchmark AVIF/JXL lossless and lossy RD at matched color handling and quality; isolate encode/decode wall-time and RSS; test a genuinely new context model against WebP instead of only reversible transforms + general-purpose compressors.

No demo or LinkedIn publication justified or performed.
