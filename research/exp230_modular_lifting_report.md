# Exp230 — modular YCoCg lifting validation (2026-11-22)

Local executable: /mnt/data/exp230/bench.py; detailed JSON: /mnt/data/exp230/results.json (NOT committed; ephemeral). The benchmark was run locally with Python, NumPy 2.3.5 and Pillow 12.3.0. Six 96x96 RGB resized scikit-image samples: development astronaut/coffee/chelsea; held-out rocket/hubble_deep_field/immunohistochemistry. **Not an independent external corpus.**

18 combinations per image: color RGB, green-relative, modular 8-bit YCoCg lifting; predictor none, left, MED; entropy zlib-9 or LZMA-6. Each stored stream includes a 14-byte header and compressed payload. 108 encode/decode trials; 108 exact RGB reconstructions after fixing an initial incorrect signed YCoCg inverse (initial invalid trials excluded and rerun). The modulo-256 lifting implementation is reversible; it differs from conventional signed YCoCg-R.

Development best green+MED+LZMA: 48,598 bytes vs WebP lossless 42,662 bytes (+13.91%). Held-out green+MED+LZMA: 45,974 bytes vs WebP lossless 40,904 bytes (+12.39%). Held-out RGB+MED+LZMA 49,938 bytes; modular YCoCg did not improve over green. PNG and WebP encoded via Pillow; only WebP used as competitive reference here. All image bytes are complete files; candidate is an experimental container, not an independently released codec.

**Conclusion:** negative competitive result. No scientific breakthrough. Limitations: no external independent dataset, no AVIF/JXL, no CI reproduction, no isolated peak RSS, no lossy R-D curves. The local script and JSON must be persisted to GitHub in a follow-up execution; this report alone is not reproducibility evidence. Never promote demo/LinkedIn based on this result.
