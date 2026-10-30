# BLKH Experiment 62 — bounded decoded-block caching for georeferenced BH60

**Status:** reproduced in GitHub Actions on 2026-10-28 UTC. This is a **system optimization**, NOT a new compression result.

**Successful workflow:** https://github.com/Kronos1027/black-hole/actions/runs/40609585115  
**Sources:** `research/universe/experiment_62_bounded_cache.py`, `.github/workflows/exp62.yml`.

## Motivation

Exp60 reported faster open/read/close BH60 queries than GeoTIFF, but Exp61 exposed an important library initialization/cache bias. Under persistent file handles and GDAL internal block cache, **GeoTIFF was faster than uncached BH60** (roughly 0.020–0.028 ms vs BH60 0.097–0.103 ms). This experiment added a simple `OrderedDict` LRU that stores already reconstructed 32×32 signed-int16 tiles under a strict budget and returns copies (to protect cache contents from caller mutations).

This is a known computing technique. A cache hit **does not run the decompressor**. It must never be described as faster neural or novel lossless decompression.

## Protocol and successful validation

- Six public geographic Skadi HGT tiles already used in Exp60, two correlated image arrangements each = 12 cases.
- Actual complete georeferenced BH60 files still smaller than the fixed-block GeoTIFF variants in 12/12 cases (storage result unchanged).
- BH60 and TIFF datasets remain open. Query workload = **24 distinct tiles**, all warmed before timing, with **5 repeated passes** in rotating codec order.
- BH60 LRU holds up to **32 decoded 32×32 int16 tiles**, at most **65,536 bytes (64 KiB)** of decoded pixel storage. The 24 chosen tiles **all fit in cache**; all 120 timed BH60 reads per case are cache hits.
- Self-test covers exact losslessness, eviction when capacity is smaller, copy-on-return (external mutation cannot corrupt cache), and memory bound.
- GitHub tests verified all twelve georeferenced file sizes, SHA-256, integer source equality and all warm-cache hit/miss counts. An initial CI run failed due to a duplicate output folder, then was fixed and passed. The failure remains recorded in Actions history.

## Measured warm-return latency — not a decompression benchmark

Medians from the successful hosted runner (all measured after prewarming; input returns are exactly equal to original, zero elevation error):

| Method | Representative per-ROI median |
|---|---:|
| BH60 warm LRU hit | **0.0006–0.0007 ms** (0.6–0.7 microseconds) |
| GeoTIFF DEFLATE warm GDAL read | 0.0101–0.0109 ms |
| GeoTIFF ZSTD warm GDAL read | 0.0100–0.0108 ms |
| GeoTIFF LZW warm GDAL read | 0.0100–0.0108 ms |

These tiny microbenchmark values depend on implementation overhead and the shared CI machine. The BH60 path returns a copy of a cached NumPy array; the rasterio path returns a GDAL-extracted NumPy array. The two APIs have different overhead and the GH runner is not a dedicated test fixture. **This does not prove the BH60 decompressor is faster**. The previous Exp61 negative result remains the valid reference for **cache-miss/persistent-open** performance.

## Limits and decision

The test is designed to make every timed BH60 query a cache hit; it is appropriate for **highly repetitive in-process map views**, but not a realistic bound for large random-area reads. A subsequent experiment MUST test query working sets exceeding cache capacity, mixed locality, cold opens, cache invalidation and threading/concurrency; report memory, hit rates and throughput along with latencies.

The file format and geospatial metadata still need end-to-end production readiness (vertical datum, independent metadata checks, NoData interpretation, GIS interoperability), and stronger specialized baselines.

**Conclusion:** BH60 now has a functional, bounded decoded-block cache that can make repeated tile reads efficient. This is a concrete, reproducible improvement in user-facing behavior, but **not justification for a worldwide compression breakthrough or an automatic public demo update**.
