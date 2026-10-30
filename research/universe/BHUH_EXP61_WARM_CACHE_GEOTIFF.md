# BLKH Experiment 61 — corrected warm-cache ROI comparison

**Date:** 2026-10-27 in Brazil / 2026-10-28 UTC.  
**Status:** GitHub Actions completed successfully. **Reverses Exp60's apparent read-latency advantage.** This is essential negative evidence, not a failure of the codec.

**Source and workflow:** `research/universe/experiment_61_warm_geotiff.py`, `.github/workflows/exp61.yml`, successful workflow https://github.com/Kronos1027/black-hole/actions/runs/40609193889 .

## Why this was necessary

Experiment 60 measured **open → read 32×32 window → decode → close** for each request. It showed BH60 faster than GeoTIFF when both were reopened every time. However, real GIS systems generally **keep GeoTIFF datasets open and cache decoded blocks**. Therefore, the earlier result could be dominated by library initialization and metadata parsing.

Exp61 revisited the same six real HGT regions, two correlated image arrangements per region, and four exactly lossless codecs (BH60 with counted EPSG:4326 affine, nodata, meter elevation metadata; 32px tiled GeoTIFF using DEFLATE, ZSTD and LZW). It:

1. opens BH60 and all three GeoTIFF datasets **once**;
2. reads each of 24 fixed tile positions on all codec backends **before starting the clock**;
3. performs five repetitions per ROI in rotated codec order, keeping handles open;
4. verifies exact int16 elevation output on every request, unchanged georeferenced metadata and complete on-disk SHA-256.

This specifically evaluates **warm-cache, persistent-file** ROI latency. It is **not** a cold-storage benchmark or a guarantee of production speed. The BLKH prototype does **not** cache decoded blocks internally; GDAL does, which reflects an important capability gap.

## Official reproduced median warm ROI latency (milliseconds)

| Region | Shape | BH60 | GeoTIFF DEFLATE | GeoTIFF ZSTD | GeoTIFF LZW |
|---|---|---:|---:|---:|---:|
| N37W119 | 128×192 | 0.099 | 0.027 | 0.023 | **0.021** |
| N37W119 | 256×256 | 0.100 | 0.028 | 0.023 | **0.021** |
| N37W122 | 128×192 | 0.100 | 0.028 | 0.022 | **0.021** |
| N37W122 | 256×256 | 0.098 | 0.028 | 0.022 | **0.021** |
| N40W106 | 128×192 | 0.099 | 0.027 | 0.022 | **0.020** |
| N40W106 | 256×256 | 0.101 | 0.027 | 0.022 | **0.020** |
| N46W122 | 128×192 | 0.098 | 0.027 | 0.022 | **0.021** |
| N46W122 | 256×256 | 0.098 | 0.027 | 0.022 | **0.021** |
| N39W105 | 128×192 | 0.097 | 0.027 | 0.022 | **0.020** |
| N39W105 | 256×256 | 0.097 | 0.027 | 0.022 | **0.020** |
| S20W044 | 128×192 | 0.100 | 0.027 | 0.022 | **0.021** |
| S20W044 | 256×256 | 0.103 | 0.028 | 0.022 | **0.020** |

**Actual finding:** On all twelve tested geospatial instances, warm persistent GeoTIFF handles provided far lower median ROI latency than the non-caching BH60 reader. That is approximately 3.5–3.8× faster for DEFLATE and 4–5× for ZSTD/LZW, on the shared hosted runner. Do **not** cite Exp60's open+read figures as proof of an intrinsically faster BLKH decoder.

## Storage result remains valid

Exp61 did **not** invalidate Exp60's real byte-accounted advantage: at fixed 32px indexing and complete horizontal geospatial metadata, BH60 yielded smaller lossless files than 32px tiled DEFLATE/ZSTD/LZW GeoTIFF in 12/12 cases. The storage and latency axes must be presented separately.

## Required next steps before production / demo

- Add a bounded decoded-block cache to the BH60 library (for example, LRU with explicit memory limit), then retest under the same warmed access pattern with both systems caching.
- Measure cold and warm accesses on a dedicated machine, using repeated mixed reads that exceed cache size, and compare realistic user workloads.
- Test encode time, memory, sidecar index costs and robust corruption handling.
- Check vertical datum and coordinate conventions before declaring the format geospatially interoperable. EPSG:4326 alone does not define the source vertical datum.
- Compare against tuned domain-specific GeoTIFF/COG, JPEG-LS and JPEG2000 implementations.

**Decision:** retain research in draft PR. The current evidence establishes a narrow **lossless storage benefit**, not universally faster or novel compression, and is not sufficient for a public benchmark victory claim.
