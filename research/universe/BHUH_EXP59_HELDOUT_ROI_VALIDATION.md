# BLKH Experiment 59 — preregistered held-out 32×32 ROI confirmation

**Date:** 2026-10-27 local Brazil, 2026-10-28 UTC.  
**Status:** ✅ Reproduced on GitHub Actions, **restricted-domain engineering validation**, not universal codec SOTA.  
**Scientific outcome:** the Exp58 hypothesis that vectorized lossless 32×32 elevation tiles can be simultaneously smaller and faster to read than identically indexed reversible JPEG2000 **generalized to two previously unexamined geographic HGT tiles**.

## Independent geographic test

The experiment's source names and fixed 32×32 block geometry were committed **before downloading or inspecting results**:

- `N39W105` — Colorado, United States
- `S20W044` — Minas Gerais, Brazil

Both sources use public Mapzen/AWS Skadi, SRTM-derived HGT signed-int16 integer-meter rasters. The source URL, complete compressed/uncompressed source SHA-256, exact cropped data SHA-256, saved archive SHA-256 and serialized size are retained in `results.json`. The experiment uses two *correlated input arrangements per geographic source* (128×192 mosaic and contiguous 256×256), **not four independent regions**. The layout/codec selected in Exp58 was not tuned on either new region.

**Reproducible run:** https://github.com/Kronos1027/black-hole/actions/runs/40606269204 — successful; `research/universe/experiment_59_independent_roi_regions.py`, workflow `.github/workflows/exp59.yml`.

## Byte-accounted and post-decode results

All methods preserve signed int16 elevations **exactly** with zero meter error. The JPEG2000 and BH58 variants have the same header/index-entry bytes and the same 32×32 independently decodable block layout. A real random-access method opens the saved archive, reads only its full header+index, seeks to the requested compressed block and decodes it. No other block payload is read.

| Source / coverage | BH58 indexed | JPEG2000 indexed | Saving over J2K | Median BH58 ROI | Median J2K ROI | p95 BH58 ROI | p95 J2K ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| N39W105 · 128×192 | **10,278 B** | 14,174 B | 27.5% | **0.198 ms** | 0.233 ms | **0.242 ms** | 0.277 ms |
| N39W105 · 256×256 | **27,985 B** | 38,314 B | 27.0% | **0.213 ms** | 0.249 ms | **0.242 ms** | 0.273 ms |
| S20W044 · 128×192 | **14,235 B** | 16,521 B | 13.8% | **0.203 ms** | 0.256 ms | **0.247 ms** | 0.290 ms |
| S20W044 · 256×256 | **40,774 B** | 46,310 B | 12.0% | **0.215 ms** | 0.272 ms | **0.245 ms** | 0.304 ms |

The original **BH56** codec remained smaller than BH58 but far slower to seek/decode a single tile (~1.13–1.16 ms median at 32² versus ~0.20–0.21 ms for BH58).

**Read sample design:** 24 fixed pseudorandom distinct ROI block positions per source arrangement, five repeats per position, codec execution order rotated by repeat number; **120 ROI observations per method per arrangement**, 480 per method across four arrangements. The table's median and p95 are measured directly on the shared GitHub Actions runner, not inferred or guaranteed production latencies. Both new regions passed full-image reconstruction, each ROI decoded byte-for-byte, metadata checksum, source provenance, and verification that physical bytes read are fewer than archive bytes.

## What the evidence establishes

1. **Narrow positive:** For independently indexed, reversible 32×32 DEM blocks, the vectorized BH58 backend beat a same-topology JPEG2000 implementation on *both archived size and measured ROI read/decode latency* in all four held-out region-arrangement tests, following all eight earlier Exp58 cases. In total this is **12 correlated image-arrangement cases across 6 geographic regions**, with real held-out regional confirmation.
2. **Not a world record:** These results do not cover every geospatial or image codec, nor tuned JPEG2000 region-of-interest APIs that may decode parts of a non-indexed codestream. A whole-map unindexed JPEG2000 file can be smaller than either fully indexed archive. Comparisons with COG/GeoTIFF, JPEG-LS, JPEG XL, zstd predictive tiling, and established optimized native implementations are still needed.
3. **Statistical boundary:** Two held-out regions, four related arrangements, and timings on a single shared hosted runner cannot support generalization to all terrains or guaranteed latency SLOs.
4. **Engineering boundary:** Encoding throughput, CPU utilization, memory footprint, decode scaling with larger maps, CRS/vertical datum/NODATA handling and true production GIS random-access workloads are not sufficiently measured. This prototype is not integrated with the original public demo.
5. **Novelty boundary:** Vectorized modular differencing, ZigZag/LEB128 and Brotli are known techniques. A successful combination for a narrow domain is engineering progress, not proof of a new compression principle.

## Decision gate for release and professional announcement

**Do not change the public demo or make a sweeping LinkedIn breakthrough claim yet.** The present data are adequate for an accurately scoped *research progress update*, explicitly describing the limited benchmark and stating it is not world-best.

Before a release:
- Multiple independent held-out regions across different topographies beyond the two above; fixed predeclared protocol;
- Matched random-access benchmarks against tiled GeoTIFF/COG, JPEG-LS and other relevant implementations, including source and output metadata;
- Robust repeated latency trials on a dedicated CPU, cold/warm cache testing, encoding cost and output-format safety;
- Verified reproducible decoder package, standalone geospatial metadata and clear format-version backward compatibility;
- Merge and end-to-end demo verification only with explicit owner approval.

**Current plan:** keep all experiment files in the GitHub draft PR, separate from `main`, and retain raw benchmark logs and artifacts for independent inspection.
