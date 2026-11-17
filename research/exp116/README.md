# Exp116 — spatial XOR of palette indices (2026-11-17)

Follow-up to Exp115; **exploratory, not a competitive codec claim**. Tests three reversible palette-index representations (raw, XOR-left, XOR-up) with zlib-9 and LZMA-6, each with a complete 12-byte header, palette (3 bytes/color), and compressed payload. Deterministic RGB images are 128×128; development seeds 101–103 and held-out seeds 9101–9103. The decoder reconstructs exactly and WebP lossless (Pillow method=6) is independently decoded and checked.

Local execution (Python 3.13, NumPy 2.3.5, Pillow 12.3.0): 108 successful BLKH encode/decode roundtrips and 24 WebP reconstructions. Peak process RSS approximately 125,676 KiB, including imports/dataset. Reported BLKH values are **per-image oracle minima across six configurations**, not a deployable adaptive selector.

| Split | Family | Mean BLKH bytes | Mean WebP bytes | BLKH wins |
|---|---|---:|---:|---:|
| dev | blocks | 377 | 373 | 1/3 |
| test | blocks | 377 | 386 | 3/3 |
| dev | palette16 | 8680 | 8339 | 0/3 |
| test | palette16 | 8680 | 8342 | 0/3 |
| dev | stripes | 91 | 117 | 3/3 |
| test | stripes | 92 | 119 | 3/3 |
| dev | random RGB | not applicable | 49244 | 0/3 |
| test | random RGB | not applicable | 49241 | 0/3 |

The stripe advantage (~27 bytes) is a tiny synthetic-container effect; no evidence of generalizable advantage. Palette16 loses, and the palette-only codec cannot encode full RGB noise. Results from the local script are recorded here; the published script is an equivalent independently authored reproduction, and its exact output must be checked in GitHub Actions before treating CI reproducibility as established. The local run initially failed during summary formatting for non-palette noise; the summary handling was corrected and rerun successfully. No AVIF/JPEG XL binaries were available locally. No photographic external corpus, fair lossy rate-distortion comparison, or isolated codec RAM measurement yet.

Reproduce: `python -m pip install numpy==2.3.5 Pillow==12.3.0 && python research/exp116/experiment.py`. GitHub Actions workflow is included separately. Keep earlier branches unchanged. No demo or LinkedIn publication.
