# Exp260 — median/gradient reversible predictors (2026-11-23)

## Scope and provenance
New local experiment following Exp259. Compared five reversible RGB spatial predictors (left, average, clipped gradient, median of left/up/gradient, Paeth), each with zlib level 9 or LZMA preset 6. Dataset: eight 96×96 RGB images from scikit-image (astronaut, coffee, chelsea, rocket = development; camera, coins, moon, page = held-out). Resize with anti-aliasing and round-to-uint8 BEFORE all codecs; these are lossless comparisons on the resized inputs, not the originals.

## Results
80 actual BLK8 encode/decode round trips; all exact against resized RGB arrays. Actual 14-byte binary header (magic, dimensions, predictor, compressor, payload length) included. Development selection by aggregate bytes: median + LZMA, 59,348 bytes. Held-out: 23,588 bytes vs Pillow/libwebp WebP lossless method=6 at 19,452 bytes (+21.26% vs WebP). Negative result. All other variants were tested but not chosen on held-out data. Per-image hashes, bytes and encode/decode timings recorded locally in results.json.

## Limitations
Small bundled dataset; grayscale originals replicated to RGB. No external photographic corpus, AVIF, JPEG XL, RD curves, isolated RSS, or GitHub Actions reproduction. Container uses Python 3.13, numpy 2.3.5, Pillow 12.3.0 and scikit-image 0.26.0; record exact version via results.json. WebP codec version is in local results. Do not claim competitiveness. Local executable artifacts: /mnt/data/blkh_exp260/bench.py, results.json and run.log. These are not included in this commit and the experiment is NOT reproducible through GitHub Actions yet.

## Next actions
Commit the actual benchmark script and dataset manifest in a subsequent commit, add Actions matrix and third-party corpus with hashes, compare AVIF/JXL and PSNR-matched RD, then measure subprocess RSS. Preserve held-out split. No demo or LinkedIn publication warranted.
