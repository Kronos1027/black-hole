# Exp221 — Indexed-palette reversible encoding (2026-11-21)

New hypothesis: for images with at most 256 RGB colors, encode an exact palette and index map; compare packed indices versus horizontal modular index differences, each compressed with zlib9 or LZMA6. This is not a lossy experiment.

Executed locally with Python 3.13.5, Pillow 12.3.0, NumPy 2.3.5, deterministic seed 221. Eight independently generated 128x128 RGB images, four development and four reserved test; 32 complete custom encode/decode exact round trips and eight WebP lossless round trips. Selection used development only: horizontal index differences + zlib9. Dev: 1,312 complete bytes. Held-out: 27,588 complete bytes vs WebP lossless 23,652 bytes (+16.64%). Shared process peak RSS: 121,888 KiB, NOT isolated codec memory. Result NEGATIVE; no competitive discovery.

Limitations: procedural images only, no external photos, no AVIF/JXL, no R-D curves, no Actions reproduction. Exact decoder source and JSON results remain in local artifact /mnt/data/blkh_exp221; not committed here. Do not claim CI reproducibility or superiority. Next step: commit source and fixture hashes, independent image corpus, baseline codecs, CI verification. Do not publish demo/LinkedIn.
