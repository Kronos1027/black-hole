# BLKH Experiment 63 — cache misses, realistic ROI locality and encode cost

**Status:** reproduced by GitHub Actions 2026-10-28 UTC. **Negative result for cache-thrashing reads; positive only with reuse of cached blocks.** This is a cache/system benchmark, not a new compression algorithm.

**Official workflow:** https://github.com/Kronos1027/black-hole/actions/runs/40611996896  
**Implementation:** `research/universe/experiment_63_cache_miss_roi.py`, `.github/workflows/exp63.yml`.  
**Artifact:** `exp63-real-dem-cache-miss-artifacts` with real .bh60 and 32px-tiled GeoTIFF files, hashes, all per-case JSON metadata.

## Methodology

Six independent real geographic source HGT files, one 256×256 elevation-meter crop each (64 individually decodable 32×32 tiles); `N37W119`, `N37W122`, `N40W106`, `N46W122`, `N39W105`, `S20W044`. Lossless int16 reconstruction of all source pixels. Georeferenced EPSG:4326 and nodata metadata included in all stored file sizes, with provenance hashes.

BH60 used separate LRU readers with capacities 8, 32 and 64 decoded blocks (16, 64 and 128 KiB of decoded pixel arrays max). GeoTIFF used persistent GDAL dataset handles and its normal internal/OS cache, **not an equivalently constrained decoded-block LRU**; the file-cache semantics are not perfectly matched. Each format saw the same deterministic sequence, with codec invocation order rotated. The three sequences:

1. **cyclic64:** 64 distinct blocks read three times in order = 192 requests. 8/32-block BH60 LRU necessarily misses on EVERY request; 64-block LRU misses on first 64 and hits on 128.
2. **uniform256:** 256 independent random block indices among 64 positions. 32-block LRU realized 135 hits and 121 misses (52.7% hit rate).
3. **hotspot256:** 256 accesses, roughly 85% of requests targeted eight hot blocks. 32-block LRU realized 220 hits and 36 misses (85.9% hit rate).

All source bytes, decoded results, archive SHA-256, decoded pixel cache bounds, source metadata, and sample counts were independently verified in CI.

## Actual six-region aggregated observations

Below is the **mean of six per-region measured means**, not a pooled sample percentile. These are shared GitHub Actions runner timings, **milliseconds per ROI read**. Report mean rather than median alone: a roughly 50% cache hit rate can make the median look excellent while the average remains poor.

| Workload | BH60 32-block cache hit rate | BH60 mean (ms) | TIFF ZSTD mean (ms) | TIFF LZW mean (ms) | Conclusion |
|---|---:|---:|---:|---:|---|
| cyclic64 | 0% | **0.1441** | **0.0547** | **0.0508** | GeoTIFF 2.6–2.8× faster |
| uniform256 | 52.7% | **0.0730** | **0.0526** | **0.0494** | GeoTIFF faster on average |
| hotspot256 | 85.9% | **0.0231** | 0.0480 | 0.0454 | BH60 faster on average |

At exactly 32 cache blocks, a true cache miss took approximately **0.14–0.15 ms** and a cache hit approximately 0.002 ms in this test. The `uniform256` scenario had a BH60 median around 0.0031 ms, but **mean ~0.0730 ms**. This difference is why median-only conclusions are misleading. 64-block capacity improves performance for cyclic64 after initial misses; this costs 128 KiB of decoded pixel storage, double the 32-block budget.

**Estimated transition:** with hit time ~0.002ms and miss time ~0.15ms, versus GeoTIFF LZW mean ~0.045–0.050ms, BH60 would need roughly >68–72% cached hits to beat it on *mean* latency. This is a simple heuristic under these configurations, **not** a validated universal threshold. Verify with mixed workloads and variance on dedicated hardware.

## Encoding speed including file write and close

Each source image was encoded and written three times for each codec. Mean of six region-specific **median** end-to-end encoding times:

| Method | Median encoding seconds, averaged across regions | Total bytes for six 256² georeferenced files |
|---|---:|---:|
| BH60 | **0.02005 s** | **227,899** |
| Tiled TIFF DEFLATE | 0.01397 s | 271,842 |
| Tiled TIFF ZSTD | 0.00487 s | 294,514 |
| Tiled TIFF LZW | 0.00249 s | 312,294 |

BH60 saved 16.2% bytes relative to this DEFLATE collection while taking ~1.4× as long to write, and was ~4.1× slower to write than ZSTD under these settings. **Do not describe it as universally faster at compression.** Geospatial storage savings depend on the chosen 32-pixel tiles, predictor and compression options; stronger codecs and block configurations remain to be tested.

## Current research decision

The confirmed performance profile is now clearer:

- BH60 has a demonstrable *space* benefit against the tested 32px tiled GeoTIFF configurations, with zero elevation loss and georeferencing. This is useful for storage-limited and locality-heavy ROI applications.
- BH60's current miss path is slower than native GDAL TIFF decoding. Its warm-cache hits can be fast, but do not measure decompressor performance.
- Normal real-world workloads mix hit rates and require realistic time/space/cold/warm evaluation. It would be incorrect to recommend a public demo advertising across-the-board latency superiority.

**Next falsifiable avenue:** eliminate Python-level decoding overhead by testing a direct fixed-width modular delta codec (Brotli over native 16-bit residuals) against the variable-length residual backend **on the same images**, then compare both bytes and true cache-miss decoding time. This is ordinary codec engineering, not new compression theory. Protect the public demo/`main` until the next evaluation.
