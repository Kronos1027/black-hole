# BLKH Experiment 56 — real DEM **lossless** compression, with held-out regions

**Date:** 2026-10-27  
**Scientific status:** reproducible and useful *engineering result*, **not** a novel codec theorem or general state-of-the-art claim.  
**Public-demo decision:** **HOLD** until broader benchmark and integration review.

## Core finding

A reversible predictive byte codec for signed-int16 meter-scale elevation combines **established** techniques: one of four reversible spatial predictors, ZigZag/LEB128 variable-length residual symbols, and Brotli level 8. The encoder selects the smallest complete stream among the four predictors, records the chosen predictor ID and dimensions in the bitstream, and the decoder independently reconstructs every elevation value from those bytes.

- **Exactly lossless:** 16 of 16 test configurations decoded bit-perfectly, with 0 m maximum absolute error. Truncated streams are rejected.
- **Four geographic sources:** `N37W119`, `N37W122` from prior research, plus *preregistered, newly selected* `N40W106`, `N46W122`.
- **Reproduced in GitHub Actions:** https://github.com/Kronos1027/black-hole/actions/runs/40597617584 — full codec round-trip, mandatory Pillow/OpenJPEG lossless JPEG2000 baseline, PNG-optimized baseline, real-source checks, deterministic SHA-256 and timing measurements; **all passed**. Earlier complete run: https://github.com/Kronos1027/black-hole/actions/runs/40597240191.
- **Files:** `research/universe/experiment_56_reversible_dem.py` and `experiment_56b_competing_codecs.py`; workflow `.github/workflows/exp56.yml` uploads the saved `.bh56` streams and the two result JSON files.

## Representative real DEM lossless corpus measurements

For each source, `six_64x64` contains 6 fixed nonoverlapping 64×64 geographic tiles; `one_256x256` is one contiguous patch at 256². All methods store enough stream metadata to recover original signed-int16 elevation meters. PNG and J2K include origin-offset metadata and four bytes per image length.

| Source | Input | BH56 (B) | Optimized PNG16 (B) | JPEG2000 lossless (B) | BH56 smaller than J2K? |
|---|---|---:|---:|---:|---|
| N37W119 (core) | 6 × 64² | **9,304** | 13,159 | 9,811 | Yes |
| N37W122 (core) | 6 × 64² | **12,904** | 18,837 | 13,431 | Yes |
| N40W106 (held out) | 6 × 64² | **10,157** | 15,089 | 10,424 | Yes |
| N46W122 (held out) | 6 × 64² | **10,742** | 14,433 | 11,171 | Yes |
| N37W119 (core) | 256² | 22,810 | 30,030 | **19,675** | No |
| N37W122 (core) | 256² | 35,043 | 47,136 | **31,466** | No |
| N40W106 (held out) | 256² | 26,130 | 36,175 | **22,040** | No |
| N46W122 (held out) | 256² | 30,186 | 40,354 | **25,983** | No |

The other eight configurations are single 64² and 128² patches from the same source areas. Cases are **correlated**, not sixteen independent geographic regions.

**All 16 configurations:**
- BH56 beat Pillow's default PNG16 in **16/16** with 26.0–34.5% less serialized bytes.
- BH56 beat Pillow PNG16 with `optimize=True` in **16/16**; weighted corpus saving **27.75%** (correlated data; an aggregate of test cases, not a population estimate).
- BH56 beat lossless JPEG2000 in **9/16** configurations. **Important: this 9/16 figure uses separately encoded tiles for the six-tile cases; see the contiguous-mosaic correction below.** On the 4 × 6-tile bundles it was approximately **3.86% smaller** than JPEG2000 by weighted bytes; on 4 standalone 64×64 tiles approximately **6.51% smaller**. BH56 was **15.13% larger** than JPEG2000 in aggregate over the four 256² cases (equivalently, JPEG2000 used about 13.1% fewer bytes than BH56).
- Each entry has original-data hash, saved-stream hash, decoded equality check, and source compressed/uncompressed HGT SHA256.

## Critical fairness correction — contiguous mosaic (Exp 56C)

**Official reproduction:** https://github.com/Kronos1027/black-hole/actions/runs/40598130150. The scripts `experiment_56c_contiguous_mosaic.py` and `.github/workflows/exp56.yml` compare all methods on **exactly the same continuous 128×192 image**. This removes the six repeated per-image PNG/JPEG2000 headers and allows those image codecs to exploit spatial correlations across tile boundaries. All methods are lossless and decoded from actual stored bytes.

| HGT source | BH56 mosaic (B) | Optimized PNG mosaic (B) | **Lossless JPEG2000 mosaic (B)** |
|---|---:|---:|---:|
| N37W119 | 9,179 | 12,011 | **8,047** |
| N37W122 | 12,728 | 17,581 | **11,655** |
| N40W106 (held out) | 9,953 | 13,809 | **8,592** |
| N46W122 (held out) | 10,871 | 13,760 | **9,469** |

**Conclusion of this critical control:** BH56 remained below optimized PNG in 4/4 continuous mosaics, but **lost to JPEG2000 in all four**. Total BH56 bytes for mosaics = 42,731 B versus JPEG2000 = 37,763 B (**BH56 is 13.2% larger**). The 4/4 favorable BH56-versus-JPEG2000 result for *separately packed small tiles* does **not** demonstrate an intrinsic codec advantage once the rival is given the option of contiguous storage.

**Scope:** preserving independently retrievable tile boundaries may still matter for some applications. A fair random-access benchmark must explicitly count any region index/cropping/decode costs for both methods. Do not claim a general JPEG2000 win from the small-tiles-only subset.

## Encoding and decoding cost

GitHub runner Python 3.11, median of 3 in-process trials **per input**, followed by median across 4 geographic regions (not a formal benchmark on dedicated hardware):

| Input size | Median BH56 encoding | Median BH56 decoding |
|---|---:|---:|
| 6 × 64² | 0.079 s | 0.018 s |
| 1 × 64² | 0.013 s | 0.004 s |
| 1 × 128² | 0.054 s | 0.012 s |
| 1 × 256² | 0.212 s | 0.048 s |

No measurements of competing *encoding speed* are claimed. Optimize only after fair end-to-end comparisons, including implementation/environment versions.

## Independent two-region validation

The final algorithm and predictor selection set were defined using the older N37W119/N37W122 corpus. The two **additional** sources were named before the real HGT was downloaded or tested and were not used to alter the predictor modes. Their observed bit-perfect results and size improvements support transfer **within the SRTM-derived terrain data domain**, not universal generalization.

Reproduce:

```bash
pip install numpy Pillow brotli
python research/universe/experiment_56_reversible_dem.py --self-test
python research/universe/experiment_56_reversible_dem.py --out /tmp/exp56
python research/universe/experiment_56b_competing_codecs.py --out /tmp/exp56b
```

## Interpretation / next gating work

1. **This is not new neural compression** and must never be marketed as a breakthrough in SIREN or AI. The existing methods are widely known; the useful result is a clean, compact, decoder-verified implementation for elevation tiles.
2. **Strong specialized rivals are not fully covered:** lossless JPEG-LS, JPEG XL, FFV1, optimized TIFF with predictive filtering, zstd predictive encodings, or domain-specific DEM compressors may outperform this implementation.
3. The experiment treats CRS, pixel spacing, geospatial coordinates, vertical datum and source nodata policy as **shared external metadata** for all methods. A production standalone DEM file must serialize them and include them in byte accounting.
4. Add independently selected topographic environments and a much larger set before making general claims.
5. **Publish demo/LinkedIn only after** scientifically credible comparisons against high-quality *domain-specific* baselines, clean user-facing file format, release tests, practical encode/decode timing, and substantive and repeatable domain wins. A transparent engineering blog describing the current narrowly scoped experiment is more appropriate than a compression world-record claim.

**Conclusion:** BLKH achieved a **real, lossless, reproducible engineering improvement over standard PNG16** on terrain patches, including two new geographical sources; it does **not** yet establish a competitive general-purpose or state-of-the-art compression result against JPEG2000 or specialist formats.
