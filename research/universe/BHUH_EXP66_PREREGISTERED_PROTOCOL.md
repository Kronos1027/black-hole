# BLKH Experiment 66 — frozen external-region test protocol

**Pre-registration, written before fetching or inspecting new HGT source data.**

This is an external validation of the **Exp65** 32×32 lossless hybrid codec. The encoder, thresholds and input sampling footprint are **frozen**. No parameters may be selected or altered based on Exp66 terrain values. An unavailable source or a crop containing a `-32768` missing-elevation sentinel must be reported as an exclusion/failure, not silently replaced by a favorable region.

## Previously unseen Skadi HGT source grid IDs

1. `N39W107` — western Colorado, USA
2. `N46E011` — Alpine Italy, Europe
3. `N35E138` — central Honshu, Japan
4. `S22W044` — southeastern Brazil

All originate from `https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat}/{tile}.hgt.gz`, SRTM-derived 3601×3601 int16 meter raster. **Fixed crop:** source rows `1200:1456` and columns `1300:1556`; one 256×256 input per region, exactly 64 tiles of 32×32. Record compressed and raw source SHA256; confirm exact reconstruction after decoding *saved files*.

## Freeze settings, comparisons and acceptance criteria

- **BH65 fixed hybrid selector:** `fast_payload_bytes <= compact_payload_bytes * 1.20` (the previously explored 20% per-block overhead tolerance). Do not choose a new threshold from Exp66 results.
- BH65 all-compact 0% reference and BH64 all-fixed reference (these are ablations, not additional independent geographic tests).
- **Fair competitor:** independently indexed, *lossless* JPEG2000 with the **same block size**, same per-block index entry length and counted JSON geospatial metadata. It must support seeking to one tile without reading the other tile payloads.
- 32px tiled GDAL/Rasterio GeoTIFF with ZSTD predictor=2 and DEFLATE predictor=2 at fixed settings already used in Exp60. Include all TIFF metadata. GeoTIFF's GDAL block cache is permitted but acknowledged, so its latency is not strictly comparable to the uncached archive readers.
- **Optional whole-image JPEG2000 baseline** with the same geospatial metadata: not randomly seekable under the independent block-access protocol, but useful to expose the potential full-file size penalty of indexing.
- Read access: one persistent handle per file, same `[0..63] * 3` sequence, all reconstructed ROI pixels verified. Strictly verify archive header/index, byte sizes, SHA256 and all 64 independent block decodes from stored bytes.
- Report **file bytes, zero meter error, encode-to-file time** (three trials, including candidate generation and actual file write/close), full read mean/p95/median time, and source SHA256. No confidence claim beyond these sources and single CI runner.
- **Success condition for limited use-case claim:** at least three of four new sources must show fixed-hybrid fewer bytes than identically indexed JPEG2000 **without** worse mean uncached ROI read time in the same run. This is a predeclared heuristic threshold, *not* proof of statistical significance. If any step fails, do not claim a win or update the demo.

## Non-claims and constraints

Existing modular differences, Brotli, ZigZag, LEB128 and per-block selection are established methods. This cannot by itself prove a new information-theoretic compression breakthrough or superior neural compression. Geolocation uses EPSG:4326 horizontal coordinates, but vertical datum and complete nodata handling need further engineering. No production-merge or public-demo update until independent source and dataset stress tests are examined.
