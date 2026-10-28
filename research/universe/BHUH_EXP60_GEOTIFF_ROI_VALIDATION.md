# BLKH Experiment 60 — georeferenced lossless DEM versus tiled GeoTIFF

**Status:** Reproduced in GitHub Actions; promising specialized **engineering** finding. **Not an established universal codec advantage** or proof of a new compression theory. Do not merge into `main` or update public demo solely from this result.

**Successful reproducible run:** https://github.com/Kronos1027/black-hole/actions/runs/40608998608

**Code:** `research/universe/experiment_60_geotiff_roi.py` and `.github/workflows/exp60.yml`.

## What changed relative to Exp58/59

Exp58/59 established a vectorized reversible codec and compared against identically indexed JPEG2000. Exp60 added a container `BH60G` with **counted geospatial metadata** (EPSG:4326 coordinate reference, sample-centre derived affine geotransform, signed int16 elevations, meter units, source tile ID, `-32768` NoData sentinel and image geometry). It was compared with real GDAL/Rasterio tiled GeoTIFF, also storing CRS, affine, NoData, source ID and vertical units. The same original signed meter grids and fixed 32×32 independent tile geometry were used in both arms.

- Four previously examined public Skadi HGT sources: `N37W119`, `N37W122`, `N40W106`, `N46W122`.
- Two additional sources, also used as held-out in Exp59: `N39W105` (Colorado) and `S20W044` (Minas Gerais).
- Two correlated geometries per HGT source: 128×192 mosaic of six 64² tiles and one 256×256 contiguous region. There are **six independent geographic source tiles, not twelve independent regions**.
- Four competing versions: BH60 with vectorized Brotli backend; TIFF with DEFLATE predictor 2; TIFF with ZSTD predictor 2; TIFF with LZW predictor 2. All GeoTIFF files were created with 32×32 physical blocks. Saved full file size and SHA256, full roundtrip, ROI equality, CRS/transform/unit metadata were checked.

## Measured storage: all formats **strictly lossless**

All sizes refer to the complete source-aligned geospatial file, **including all archive indexes and metadata**, not the compressed block bytes alone.

| HGT source | Shape | BH60 | TIFF DEFLATE | TIFF ZSTD | TIFF LZW |
|---|---|---:|---:|---:|---:|
| N37W119 | 128×192 | **13,282** | 15,499 | 17,039 | 17,863 |
| N37W119 | 256×256 | **32,645** | 37,783 | 41,941 | 44,339 |
| N37W122 | 128×192 | **17,519** | 22,017 | 23,305 | 24,791 |
| N37W122 | 256×256 | **46,173** | 58,751 | 61,787 | 65,845 |
| N40W106 | 128×192 | **15,196** | 17,077 | 18,691 | 19,787 |
| N40W106 | 256×256 | **39,235** | 43,817 | 47,905 | 50,685 |
| N46W122 | 128×192 | **13,986** | 16,845 | 18,293 | 19,045 |
| N46W122 | 256×256 | **40,475** | 47,311 | 51,045 | 53,919 |
| N39W105 | 128×192 | **10,584** | 12,667 | 14,187 | 15,225 |
| N39W105 | 256×256 | **28,291** | 32,695 | 36,849 | 39,435 |
| S20W044 | 128×192 | **14,541** | 18,353 | 19,725 | 20,735 |
| S20W044 | 256×256 | **41,080** | 51,485 | 54,987 | 58,071 |

**Result:** BH60 achieved **12/12 smaller complete files** than the three tested, 32×32 tiled GeoTIFF compression configurations. All height values were recovered exactly, with **zero meters of reconstruction error**. This is a real and reproducible specialized engineering win; it is not a novelty or all-codecs leadership claim.

## Window-read benchmark — important limitations

24 predetermined different block positions per test case × 3 repetitions, in rotating codec order. Measured wall-clock includes opening the file, fetching a 32×32 window, decoding it and closing the file.

- BH60 median ranged approximately **0.230–0.249 ms**.
- GeoTIFF DEFLATE medians ranged approximately **0.515–0.530 ms**, ZSTD **0.515–0.531 ms**, LZW **0.500–0.515 ms**.

**Do not claim the underlying compression/decompression algorithm is inherently twice as fast.** The GeoTIFF side uses GDAL's full dataset-open and raster-window machinery, with different library and metadata parsing overheads. It also can use internal GDAL caching; no physical on-disk bytes read for TIFF were measured. The BH60 side physically reads only header+index and the needed compressed tile. The difference may be substantially reduced or reversed with persistent open handles, GDAL block cache, specialized COG/GeoTIFF APIs or native optimized codecs.

## Required further validation before the demo or LinkedIn feature announcement

1. Compare **persistently open GeoTIFF dataset handles** against a persistently open BH60 file and report equal warm-cache/cold-cache conditions.
2. Include **COG/GeoTIFF with appropriate block sizes** (16/32/64/128) and tuned ZSTD/DEFLATE predictors, plus J2K/JPEG-LS/JPEG XL, using the same georeferenced metadata and ROI semantics. Current positive result applies to the fixed 32px tile baseline only.
3. Run on a dedicated CPU under repeatable conditions with variance and encode-time comparisons.
4. Add **vertical datum**, complete NoData policy, independent checksum tests, input validation hardening, large-raster stress tests and fuzzing, and document any incompatible geospatial assumptions. EPSG:4326 does not specify the elevation vertical datum.
5. Reproduce in a GIS-facing command-line demo and verify end-to-end interoperability with other tools. No external tool is currently assumed able to ingest BH60 natively.

**Decision:** continue the research on a protected branch and leave the public demo untouched. An accurately qualified research update is possible; stronger launch/marketing claims are not yet justified.
