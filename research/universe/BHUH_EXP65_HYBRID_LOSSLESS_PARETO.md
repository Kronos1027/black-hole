# BLKH Experiment 65 — adaptive compact/fast lossless DEM blocks

**Date:** 2026-10-28 UTC. **Status:** Reproducible exploratory **engineering Pareto** study, not a new general-purpose compression discovery. Full algorithm and decoder are committed in `experiment_65_hybrid_pareto.py`. GitHub workflow `.github/workflows/exp65.yml` verifies the 24 serialised archives and exact signed-int16 reconstruction from the saved bytes.

**Original successful workflow:** https://github.com/Kronos1027/black-hole/actions/runs/40612871100

## Hypothesis

Exp63 showed that variable-length BH60/BH58 block decoding is slow on cache misses (~0.14ms on its CI run). Exp64 showed that fixed-width uint16 delta+Brotli can speed real cache misses, but expands files by approximately 17.6% on the same six areas. Instead of selecting one codec for the whole file, Exp65 selects a codec **per 32×32 geographic block** based on a fixed upper bound on extra compressed payload bytes.

- **Codec ID 2**: compact BH58 reversible ZigZag/LEB128 + Brotli.
- **Codec ID 3**: faster BH64 reversible fixed-width modular difference + Brotli.
- Rule: choose the faster form whenever `fast_payload_bytes <= compact_payload_bytes * (1 + tolerance)`; otherwise choose the smaller form.
- The four tolerance levels are **0%, 15%, 20%, 25%**. They are an exploratory sweep; **these six HGT regions were already seen in the preceding experiments**. Results cannot be called blind holdout confirmation.
- BH65 files include full counted georeferenced metadata, standalone index and per-block codec IDs, all complete file sizes and SHA256. 256×256 signed int16 elevations, 64 individually decodable 32×32 blocks; cycle all blocks 3 times with persistent file handles and **no decoded-block cache**. Verify all 64 blocks reconstruct exactly.

## Reproduced overall rate–decode-time frontier

Six independent source HGT grids, all six previously used: N37W119, N37W122, N40W106, N46W122, N39W105, S20W044. Each is represented by one 256×256 patch; thus **six images, not 24 independent regions**. Total latency below is the mean of six within-region mean times from the official hosted GitHub runner; repeated cycles are correlated observations.

| Extra bytes tolerated per block | Six complete archives (B) | Bytes vs compact 0% | Mean uncached ROI time (ms) | Time reduction vs compact |
|---|---:|---:|---:|---:|
| 0% | 227,899 | baseline | 0.07510 | baseline |
| 15% | 238,548 | +4.67% | 0.05057 | 32.7% faster |
| 20% | 249,827 | +9.62% | 0.03443 | 54.2% faster |
| 25% | 258,249 | +13.32% | **0.02568** | **65.8% faster** |

**Exact reconstruction: zero meters elevation error** in all 24 files, verified from decoded bytes and serialised metadata.

## Per-region case examples

| Region | Compact 0% bytes / mean ms | Hybrid 15% bytes / mean ms | Hybrid 20% bytes / mean ms | Hybrid 25% bytes / mean ms |
|---|---|---|---|---|
| N37W119 | 32,645 / 0.0741 | 33,951 / 0.0554 | 36,367 / 0.0320 | 38,019 / 0.0194 |
| N37W122 | 46,173 / 0.0698 | 46,173 / 0.0692 | 47,239 / 0.0632 | 49,804 / 0.0525 |
| N40W106 | 39,235 / 0.0751 | 42,525 / 0.0351 | 44,057 / 0.0228 | 44,570 / 0.0183 |
| N46W122 | 40,475 / 0.0763 | 41,487 / 0.0635 | 44,139 / 0.0441 | 46,875 / 0.0270 |
| N39W105 | 28,291 / 0.0724 | 30,333 / 0.0374 | 32,024 / 0.0178 | 32,113 / 0.0160 |
| S20W044 | 41,080 / 0.0830 | 44,079 / 0.0428 | 46,001 / 0.0267 | 46,868 / 0.0208 |

The 15% threshold chooses fast decoding for 0–46 of the 64 blocks depending on source. There is no single threshold that guarantees a particular file size or latency, and a lower median latency should never hide true miss cost.

## Verification and applicability

- All bitstreams were decoded independently from their saved `.bh65` files; complete input hashes were checked, corruption rejection self-test passed, and all 192 ROI reads per variant compared with source signed-int16 elevations.
- All codec sizes include metadata, per-block index and headers. No pre-decoding PSNR or bogus use of PNG lossless-vs-lossy comparisons.
- The benchmark uses a **shared hosted runner**; the absolute runtimes vary between runs. Exp65 compact baseline ran at ~0.075ms versus Exp64's ~0.157ms BH60 miss path, even though the underlying compact block codec is the same. This illustrates why **speedup comparisons must stay within the same run**. Do not combine timing values from Exp64 and Exp65 as if host loads were controlled.
- The study **does not include matched GeoTIFF, JPEG2000 or JPEG-LS baselines on the same Exp65 runner**, so it cannot claim that any hybrid threshold outperforms all alternatives.
- No newly held-out geographical regions, reproducible dedicated CPU benchmark, large-file memory stress, full vertical datum, standalone GIS interoperability, or production API security review yet.

## Decision

**Promising ability to configure a size/latency trade-off for exact geospatial data.** No novel neural or state-of-the-art theory is established. The owner-approved release/demonstration remains pending until stronger competitors and genuinely unseen data are tested at fixed settings. Keep research on the draft PR branch, not `main`.
