# BLKH Experiment 64 — fixed-width reversible DEM, faster true cache-miss decoding

**Date:** 2026-10-28 UTC. **Result:** positive but domain-limited engineering trade-off, **not a world record or a novel neural codec**.

**Official GitHub Actions reproduction:** https://github.com/Kronos1027/black-hole/actions/runs/40612481338  
**Reproducible source:** `research/universe/experiment_64_fixed_delta_roi.py`; workflow `.github/workflows/exp64.yml`.  
**Artifact:** `exp64-fixed16-dem-roi` contains complete independent `.bh64`, `.bh60`, GeoTIFF files and `results.json`.

## Motivation and design

Experiment 63 proved BH60's traditional ZigZag/LEB128+Brotli path can have a ~0.14–0.16 ms block decode on an LRU cache miss; GeoTIFF can be faster in miss-heavy workloads. Experiment 64 tests **existing standard engineering techniques**, not a new mathematical breakthrough:

- Lossless unsigned 16-bit **modular delta** for each 32×32 elevation block; choose horizontal or vertical prediction based on actual compressed bytes.
- Encode all 1,024 residuals as **little-endian fixed-width uint16**, then compress each block with Brotli q8. Unlike prior work, no ZigZag/variable-length unpacking is needed.
- Decoding: stream-decompress exactly 2,048 bytes, `numpy.frombuffer`, cumulative sum and cast back to signed int16. Values reconstruct **exactly**, including negative altitudes and wraparound extremes.
- A complete standalone `BHG64` archive stores explicit geospatial JSON metadata (EPSG:4326, affine transform, region ID, signed-int16 meter elevations, nodata sentinel) plus the same fixed per-block index layout as BH60. **All metadata and indexes count**.
- No decoded-block caching in the BH64 reader, even for repeated requests. BH60 comparison explicitly uses LRU capacity 32, forced to miss on **all 192 cyclic reads**; tested also against persistently opened 32px-tiled GeoTIFF ZSTD and LZW.
- Six real independent geographic source HGT tiles, one 256×256 image each: N37W119, N37W122, N40W106, N46W122, N39W105, S20W044. These are **already used in previous experiments**; do not mislabel them as newly held-out.
- Three full file encodings and 192 ROI read+decode timings per format/source. Every source and 64/64 indexed blocks were verified independently with exact pixel equality, SHA256 and serialized size in CI.

## Measured outcomes (GitHub hosted CPU)

| HGT source | BH60 (B) | BH64 (B) | TIFF ZSTD (B) | TIFF LZW (B) | BH60 forced-miss mean (ms) | BH64 uncached mean (ms) | TIFF ZSTD mean (ms) | TIFF LZW mean (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N37W119 | 32,645 | 38,149 | 41,941 | 44,339 | 0.1555 | **0.0379** | 0.0610 | 0.0542 |
| N37W122 | 46,173 | 57,993 | 61,787 | 65,845 | 0.1628 | **0.0442** | 0.0683 | 0.0561 |
| N40W106 | 39,235 | 44,570 | 47,905 | 50,685 | 0.1533 | **0.0399** | 0.0616 | 0.0537 |
| N46W122 | 40,475 | 48,162 | 51,045 | 53,919 | 0.1598 | **0.0407** | 0.0620 | 0.0555 |
| N39W105 | 28,291 | 32,113 | 36,849 | 39,435 | 0.1526 | **0.0359** | 0.0611 | 0.0525 |
| S20W044 | 41,080 | 47,095 | 54,987 | 58,071 | 0.1582 | **0.0396** | 0.0623 | 0.0545 |

**Aggregate over six cases:**

- BH64 uncached ROI **mean of region means = 0.03971 ms**, versus BH60 forced misses = **0.15701 ms**. This is an observed **74.7% lower mean latency** for the test.
- BH64 was also faster in measured ROI mean than both ZSTD and LZW GeoTIFF in **6/6 source images**, in this experimental setup.
- BH64 files were **8.97% fewer bytes than ZSTD** across the six files (total BH64 = 268,082 bytes vs total TIFF ZSTD = 294,514 bytes), and likewise fewer than LZW. However BH64 files were **17.63% larger than BH60**, reflecting a clear rate–latency trade-off.
- Repeated full encode+save median averaged over six sources: BH60 **0.0200 s**, BH64 **0.0149 s**, TIFF ZSTD **0.0050 s**, TIFF LZW **0.00245 s**. Thus BH64 writes faster than BH60 but remains slower than native GeoTIFF encoders under these settings.

## Scientific limitations and FAIR interpretation

1. **No cache match guarantees:** BH64 never caches decoded blocks; the GDAL baselines were kept open, so they may cache decompressed blocks. This makes a favorable BH64 result interesting, but not a hardware-neutral or library-neutral decoder benchmark.
2. **Codec scope:** TIFF DEFLATE, JPEG2000, JPEG-LS, optimized zstd predictor variants, COG and tiled native codecs were **not** compared in this experiment. Prior Exp60 TIFF DEFLATE had smaller files than BH64 in some images, so **do not claim dominance over all GeoTIFF formats**. JPEG2000 can also compress full contiguous regions very efficiently, particularly when random access is not required.
3. **Sample scope:** Six previously studied source HGT regions, small 256×256 patches. More independent terrain/geographic sources, realistically large datasets, and dedicated machine testing are needed.
4. **Vertical datum/NoData:** Horizontal CRS, pixel spacing, source region and nodata sentinel are serialized, but a full vertical datum description and source-level consistency checks remain outside this prototype. EPSG:4326 is only horizontal.
5. **No novelty claim:** Fixed-width delta encoding and Brotli are known algorithms. The contribution is a concrete, reproducible trade-off and georeferenced experimental format, not new neural compression theory.

## Decision and next falsifiable experiment

**Hold the main branch, public demo, and sweeping LinkedIn breakthrough claim.** BH64 is sufficiently reproducible to justify developing a **restricted real-terrain viewer/storage prototype**, but not to advertise universal compression leadership.

A fair next experiment should compare BH64, BH60 and optimized DEFLATE/JPEG2000 at identical random-access block sizes, and test **blockwise hybrid BH60/BH64 selection** to find a size/latency Pareto frontier. Tune on existing regions, then preregister and verify genuinely unseen regions before making a wider claim.
