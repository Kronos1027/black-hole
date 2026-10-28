# BLKH Experiment 68 — contiguous stripes do NOT rescue compression/ROI trade-off

**Date:** 2026-10-28 UTC. **Status:** successful, fully reproduced GitHub Actions run, scientifically **negative** outcome for the tested stripe architecture. Existing algorithms only; no new compression theory.

**Official successful workflow:** https://github.com/Kronos1027/black-hole/actions/runs/40618164991  
**Source / tests:** `research/universe/experiment_68_contiguous_stripes.py`, `.github/workflows/exp68.yml`.  
**Reproducibility:** 2 artificial input self-tests; all stored real HGT files verified independently from decoded bytes with SHA-256; real `results.json` and bitstreams stored as workflow artifact.

## Hypothesis and methodology

The baseline Exp67 found a critical limitation: **one whole-image JPEG2000 file can serve 32×32 GDAL windows without individually compressing 64 tiles**. Exp68 explores a family of intermediate spatial granularities: independent horizontal stripes of heights 32, 64 and 128, each spanning the full width (256 pixels), rather than 64 independent blocks.

Three lossless coding modes for every stripe height: (1) horizontal/vertical reversible modular int16 spatial deltas + Brotli; (2) reversible JPEG2000 per stripe, with counted signed-elevation origin; (3) per-stripe choose the smaller byte stream. All complete output files count serialized geospatial metadata, stripe indices, block headers and bitstreams.

**Four source grids:** N39W107, N46E011, N35E138, S22W044, each one 256² int16 meter-elevation raster. These four regions were **already inspected in Exp66/67**, so this is exploratory, **not** independent confirmation. Also compare original BH65 indexed codec, full continuous JP2 (including counted external metadata), and 32px-tiled GeoTIFF ZSTD. GDAL/JP2 and TIFF readers have possible internal caching; BH68/BH65 directly decode their compressed payloads on every request.

**Quality:** all 36 stripe-mode georeferenced bitstreams reconstructed exact input elevation meters, zero loss. Same 128 spatial window read samples per method per region, identical ROI order and persistent file handles. Unreported larger maps or cold I/O are not implied.

## Selected official measurements

All sizes are whole files, including geodata; mean millisecond latency includes a persistent-handle 32×32 ROI read and required decode.

| Region | BH65: bytes / ms | 32-high JP2 stripes: bytes / ms | 128-high JP2 stripes: bytes / ms | **Whole-JP2: bytes / ms** |
|---|---|---|---|---|
| N39W107 | 45,484 / 0.091 | 28,964 / 0.786 | 26,394 / 2.471 | **25,982 / 0.081** |
| N46E011 | 50,716 / 0.114 | 33,570 / 0.888 | 30,888 / 2.829 | **30,484 / 0.082** |
| N35E138 | 49,070 / 0.167 | 35,572 / 0.934 | 32,877 / 3.004 | **32,436 / 0.084** |
| S22W044 | 51,666 / 0.083 | 30,740 / 0.815 | 28,000 / 2.555 | **27,590 / 0.081** |

**All 36 stripe configurations were dominated on both complete stored bytes and measured mean ROI latency by the continuous whole-JP2 file in this single hosted runner.** The adaptive per-stripe mode selected JPEG2000 over Brotli for all stripes in all four observed regions; it adds no compression advantage for this corpus.

A larger independent stripe's JP2 container captures global spatial regularities and reduces overhead; however, decoding an entire full-width stripe to recover 32×32 ROI incurs high cost. For example N39W107 stripe128 JP2 is only 412 bytes larger than whole-JP2, but a 32px ROI needs ~2.47 ms vs ~0.081 ms in the same experiment. A narrow 32-high stripe costs ~11.5% extra bytes and ~9.7× measured time versus whole-JP2.

## What this does and does NOT prove

- It **rejects this particular row-stripe design as a competitive improvement** under the tested 256² terrain and GDAL ROI workload; it does NOT prove all striping, multi-resolution or neural schemes are inferior.
- Internal GDAL OpenJPEG tile/cache policies were not constrained to the custom codecs' memory and could influence timing. Whole-JP2 `rasterio.Window` calls were truly executed, but physical bytes transferred by GDAL were not measured.
- Four source geographies, each sampled once, are too few for a general population claim. These are small SRTM-derived rasters and use known codecs, not validation of 3DEP/vertical datum accuracy.
- Exp68's hybrid selection criterion was **storage-only**; JPEG2000 winning all stripe-size comparisons should not be interpreted as a globally optimal size/latency objective.
- No superiority announcement, product demo update, or main-branch merge is warranted.

## Next scientific question

Exp68 and Exp67 show whole-JP2 can look superior on **small 256² arrays**. A potentially falsifiable issue is **scaling to larger continuous rasters**, in which the cost and cache behavior of extracting a tiny 32² window from a single JP2 codestream may differ. The correct next experiment should compare truly 512² and 1024² elevation rasters (same source grid, metadata counted) against a generalized indexed BLKH codec, with repeated *cold/warm and low/high locality* workloads and memory measurements. Do not guess a winner in advance.

**Decision:** preserve Exp68's negative result in research branch, public main/demo unchanged.
