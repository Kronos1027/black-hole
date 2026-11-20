# BLKH Exp181 — XOR and second-order residuals (local, negative)

2026-11-20. Recovered Exp172–Exp180 from prior research context; tested a distinct hypothesis: spatial XOR-up versus modular first/second-order residuals, with RGB versus green-difference transforms and zlib9 versus LZMA3. The fixed 4-development/4-test split uses bundled scikit-image samples resized by Pillow BOX to at most 192×192. **This is not an external independent dataset.**

96 encode/decode roundtrips reconstructed exact RGB; 96 CRC tampering checks rejected. Selection on development only: `green/delta_up/zlib9`. Held-out test: **BLKH 114,491 complete bytes, WebP lossless (Pillow method=6) 75,464 bytes; BLKH +51.72%**. Shared process peak RSS 149,588 KiB, not per-codec. Python 3.13.5, numpy 2.3.5, Linux 6.18.57. Full per-image hashes, timings, experiment script, results JSON and a proposed Actions workflow exist in the local Exp181 bundle, but are **not committed in this branch**; no Actions run has been performed. Reproduction requires uploading those files first.

Limitations: no external image set, no AVIF/JXL, no lossy rate-distortion curves, no isolated memory. Negative finding; no competitive claim, demo or LinkedIn recommendation. Do not compare aggregate bytes to other Exp runs using different images/resolutions.
