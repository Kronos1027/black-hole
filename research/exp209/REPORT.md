# Exp209 — Reversible spatial residuals with Brotli/LZMA/zlib (2026-11-21)

Exploratory local experiment, **negative**. Four modes (raw, left, up, green-channel decorrelation + up) crossed with three entropy backends (zlib9, LZMA6, Brotli9): 12 configurations on eight scikit-image samples (4 development, 4 held-out test), 96 exact RGB encode/decode round trips. Full container headers counted (15 bytes). Images resized to <=256 px and grayscale replicated into RGB; WebP lossless Pillow method=6, decoded to verify pixels.

Chosen on development only: green_up + LZMA6. Held-out test: 147,008 bytes versus WebP lossless 121,854 bytes (+20.64%, worse). Shared process peak RSS 183,556 KiB; not isolated. Python 3.13.5, Pillow 12.3.0.

**Caveats:** No external independent photo dataset, no JPEG XL or AVIF, no rate-distortion curves, no isolated memory benchmarks, no Actions execution. The executable script and full JSON results were generated locally but **are not committed here**. Do not claim reproducibility via Actions. The code's earlier Paeth mode was omitted from final sweep due to prohibitive Python runtime; this is not a Paeth benchmark.

**Decision:** no competitive discovery, no public demo, no LinkedIn. Follow-up: incorporate executable + JSON, use externally sourced holdout and conventional codec RD baselines, validate in Actions.
