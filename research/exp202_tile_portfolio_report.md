# Exp202 — Tile-wise PNG/WebP lossless portfolio (negative)

Local experiment 2026-11-20. Eight scikit-image images resized to 128x128 RGB, four development and four held-out test. 6 configurations: tile size 32,64,128 x WebP-only or choose smaller PNG/WebP per tile. Full container accounts for header, tile codec identifiers and lengths. 48 complete encode/decode round trips, exact RGB reconstruction and truncated payload rejection.

Winner selected ONLY on development: tile=128, WebP-only. Held-out test: 33,820 bytes including container versus 33,744 bytes WebP lossless; +0.2252% (worse). Selected encode total 1.159 s; decode 0.00257 s (single local run, not robust timing). Shared peak RSS 147,868 KiB, not codec-isolated. Python 3.13.5, Pillow 12.3.0, libwebp 1.6.0.

**No compression discovery.** The winner is WebP encapsulation, not a novel codec. Limitations: scikit-image bundled samples are not an independent external photographic dataset; no AVIF/JXL, no lossy RD curves, no GitHub Actions validation. No demo or LinkedIn.

Local executable script and raw JSON generated at /mnt/data/blkh_exp202/experiment.py and /mnt/data/blkh_exp202/results.json (not yet committed to repository). Follow-up: commit runnable script, add external data with hashes and reproducible CI, compare codecs under equivalent distortion constraints.
