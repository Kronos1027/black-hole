# Exp214 — reversible byte-layout and bitplane holdout (2026-11-21)

Local prototype: six reversible transforms (raw, planar RGB, horizontal XOR, horizontal modular difference, vertical modular difference, bitplane transpose) crossed with zlib9, LZMA6 and bzip2-9.

Eight 128x128 scikit-image samples; four development (astronaut, coffee, chelsea, rocket) and four reserved test (camera, coins, moon, page), converted to RGB. Selection based only on development aggregate byte count: **subup/LZMA6**. All 144 encode/decode combinations passed pixel-exact roundtrip. The complete 10-byte BLKH header was counted.

Held-out aggregate: **39,992 B BLKH vs 33,784 B WebP lossless**, **18.38% larger**. Negative result; no codec superiority. WebP used Pillow lossless method=6, decoded pixels checked. Times recorded locally, but no isolated memory measurement. Python 3.13.5, numpy 2.3.5, Pillow 12.3.0.

**Important:** Executable Python and full JSON were generated locally but are not yet tracked by this commit. No GitHub Actions run. No independent external photos, AVIF/JXL, rate-distortion curves, or isolated RAM benchmarks. No public claims, demo, or LinkedIn warranted. Follow-up: commit reproducible script and dataset manifests, then independent held-out external benchmark and Actions.
