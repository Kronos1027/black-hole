# BLKH Exp192 — XOR spatial residuals and bit-plane layouts

Experiment performed locally 2026-11-20. The full executable script and JSON are in the local Exp192 bundle, **not yet committed to this branch**.

- 8 scikit-image samples downscaled to 32x32 RGB uint8; 4 development, 4 held-out.
- 24 configurations: four causal XOR predictors (left, up, left XOR up, left XOR up XOR upper-left), three byte layouts (packed, planar, bit planes), zlib9 and LZMA6.
- 192 encode/decode exact roundtrips; 192 deliberately corrupted streams rejected. Header and checksum included in byte counts.
- Selected solely on development: up / packed / LZMA6.
- Held-out: 4228 BLKH bytes vs 2732 WebP lossless bytes (+54.76%). Encode 0.01616 s, decode 0.01156 s summed across four held-out images, not isolated CPU timing.
- Python 3.13.5; NumPy 2.3.5; libwebp 1.6.0.
- Important limitation: only tiny resized samples, not independent photographic datasets; no AVIF/JXL or CI runs; no isolated RAM measurement. No codec superiority demonstrated.

Next: commit script and JSON from bundle; execute Actions; expand to independent 512px+ photographic dataset, compare WebP, AVIF lossless and JXL lossless with exact reconstruction, and measure memory per process. No demo or LinkedIn justified.
