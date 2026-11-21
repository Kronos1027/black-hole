# BLKH Exp220 — lossless tile-wise WebP/PNG hybrid (2026-11-21)

Hypothesis: tile-specific choice between WebP lossless and PNG may beat whole-image WebP on heterogeneous RGB content. This differs from earlier pixel predictor/transform experiments Exp210–219.

Executed locally: 8 bundled skimage RGB sources, center-cropped 256x256; 4 development (astronaut, coffee, chelsea, rocket), 4 held out (hubble, immuno, colorwheel, cathedral). Six variants: tiles 64,128,256 and WebP-only or WebP/PNG selection by smallest complete encoded tile. Full container includes 11-byte header plus 5 bytes/tile. 48 hybrid encode/decode cycles verified exact pixel equality. Baselines WebP lossless Pillow method 6, PNG level 9, JPEG2000 reversible. JPEG2000 baseline also roundtripped exactly. Selection used dev only.

Selected: tile256_webp (i.e. effectively full-image WebP plus container overhead), dev total 311282 bytes. Holdout 311860 bytes vs WebP 311796 (+0.021%), PNG 416782, JPEG2000 388713. Negative: no gain over WebP. This strongly suggests tiled mixtures did not overcome tile/container overhead on these images. Peak shared process RSS 138844 KiB (not codec-isolated). Per-sample encoding/decoding wall times and source hashes recorded in local results.json.

Limitations: skimage bundled samples are not independent external photographs; no AVIF/JPEG XL or full rate-distortion curves; no Actions reproduction. Local executable and JSON retained in BLKH_exp220_bundle.zip; source not yet committed due container network isolation. Do not claim competitive superiority. No demo/LinkedIn.

Reproduction: Python 3, numpy, Pillow with WebP and JPEG2000 support, scikit-image. Run python exp220.py. Confirm results.json SHA256 source hashes and exact reconstruction. Next: upload script/results and workflow, use external immutable image corpus, test AVIF/JXL and isolated memory, verify GitHub Actions before scientific claim.
