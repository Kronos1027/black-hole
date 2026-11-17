# Exp117 — palette index predictor ablation (2026-11-17)

Continuation of Exp116, isolated research branch. Tested **raw, XOR-left, modular subtract-left, modular subtract-up and modular Paeth** on 128×128 RGB palette images, with **zlib-9 and LZMA-6** per mode (10 candidate combinations). Complete encoded files include a 12-byte header, 3 bytes per palette entry and compressed payload. Decoder reconstructs the exact RGB image. Five image families: random palette16, 8×8 blocks, vertical stripes, diagonal bands, repeated random runs. Dev seeds: 101–103; held-out test seeds: 9101–9103. Images are synthetic and not a photographic external corpus.

Local environment: Python 3, NumPy 2.3.5, Pillow 12.3.0. Local run: **300 exact BLKH roundtrips**, **30 independently decoded WebP lossless roundtrips**. Peak process RSS 118200 KiB, including Python, libraries, and data; this is NOT isolated codec RAM.

| Split | Family | WebP mean bytes | BLKH oracle mean bytes | BLKH wins |
|---|---|---:|---:|---:|
| dev | palette16 | 8338.7 | 8680.0 | 0/3 |
| dev | blocks | 372.7 | 378.7 | 1/3 |
| dev | stripes | 117.3 | 91.0 | 3/3 |
| dev | diagonal | 206.0 | 157.0 | 3/3 |
| dev | runs | 118.7 | 97.0 | 3/3 |
| test | palette16 | 8342.0 | 8680.0 | 0/3 |
| test | blocks | 386.0 | 378.0 | 3/3 |
| test | stripes | 118.7 | 91.7 | 3/3 |
| test | diagonal | 206.7 | 157.0 | 3/3 |
| test | runs | 116.0 | 93.3 | 3/3 |

**Crucial caveat:** BLKH scores are oracle minima across 10 configurations *selected after seeing each test image*, rather than a locked deployable selector. Thus these wins are exploratory, not evidence of competitive superiority. The repeated synthetic generator families also limit independence. WebP lossless used Pillow method=6, with RGB reconstruction verified. AVIF and JPEG XL were not tested; no lossy RD curve was measured.

**Reproducibility blocker:** The full Python script and detailed JSON were executed and saved in the local experiment environment but are not committed in this branch because the local runtime and GitHub connector do not share file bytes. No GitHub Actions validation is claimed. Next run should upload the script and JSON, add CI with exact dependency pins, then rerun a locked dev-selected configuration on genuinely external holdout images, measuring full-file bytes, time, and isolated memory. No demo or LinkedIn publication.
