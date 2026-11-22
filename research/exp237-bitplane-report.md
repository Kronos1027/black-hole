# Exp237 — bit-plane layout ablation (2026-11-22)

Prior art: Exp236 row/channel predictor; this run tests a different hypothesis: bit-plane transposition of horizontal/vertical modular residuals before entropy coding, alongside XOR horizontal and raw/standard residual controls.

Executed locally: `/mnt/data/exp237/bench.py`; full machine-readable per-image results: `/mnt/data/exp237/results.json` (NOT yet committed). Six deterministic 128x128 center crops from scikit-image: development astronaut, coffee, chelsea; held-out rocket, camera, colorwheel. 12 experimental methods (6 transforms x zlib-9/LZMA-6) plus PNG-9 and WebP lossless. 84 encode/decode cycles, all exact RGB. The experimental format is a real 14-byte serialized header plus compressed payload; decode parses it independently. All sizes are complete streams.

Method selected on development: horizontal modular residual + LZMA-6 (86,810 bytes dev). Held-out: 37,914 bytes vs WebP lossless 31,000 bytes and PNG 45,041 bytes; +22.303% over WebP (negative competitive result). Vertical LZMA was 35,146 bytes on held-out but was NOT selected by development, so this cannot be treated as generalization. Bit-plane horizontal LZMA was 48,770 bytes and bit-plane vertical LZMA 43,318 bytes on held-out: bit-plane layout did not help.

Environment: Python 3.13.5, Pillow 12.3.0, numpy 2.3.5, libwebp 1.6.0. Process max RSS 149,064 KiB (whole run, not isolated codec). Timings and SHA-256 stream hashes recorded in local JSON. First execution raised NumPy uint8 modulo OverflowError; corrected to use native modular uint8 subtraction and reran successfully.

Limitations: scripts/JSON not uploaded, no Actions run, no independent external corpus, no AVIF/JPEG XL or lossy rate-distortion curves. Do not claim a breakthrough, codec superiority or reproducible CI. No demo or LinkedIn publication.
