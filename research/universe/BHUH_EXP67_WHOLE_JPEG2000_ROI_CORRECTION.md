# BLKH Experiment 67 — whole-image JPEG2000 **does** permit GDAL ROI window reads

**Date:** 2026-10-28 UTC. **Status:** ✅ Reproducible, full workflow successful, **important negative benchmark correction**. Do not market BH65 as superior over a well-used whole-image JPEG2000 file just because separately indexed J2K blocks were less competitive.

**Official run:** https://github.com/Kronos1027/black-hole/actions/runs/40614674008  
**Implementation:** `research/universe/experiment_67_full_j2k_roi.py` and `.github/workflows/exp67.yml`.  
**Evidence artifact:** `exp67-full-jpeg2000-real-roi`, with signed-elevation geospatial wrappers, extracted raw JP2 images, complete original input SHA256 and full ROI quality/timing JSON.

## Why the test matters

Exp66 correctly preregistered new sources and reproduced a fixed BH65 hybrid win against **64 independently indexed reversible JPEG2000 blocks** on 3 of 4 HGT sources. But JPEG2000 can also encode **one full 256×256 continuous image** and expose efficient spatial window access through the **GDAL JP2OpenJPEG driver**, without creating 64 separate JP2 files and their repeated codec headers. Therefore, comparing only against independently compressed J2K blocks can dramatically overstate BH65's general competitive advantage.

The full-JP2 output of Exp66 was used **unchanged**. It is contained in a custom counted wrapper with EPSG:4326 horizontal reference, affine, source tile ID, height units, a signed 16-bit offset and all necessary height metadata; the actual `.jp2` bytes were extracted purely for GDAL's `rasterio.open` window read. **All size comparisons count the complete original wrapper**, not only raw JP2. Note: the extracted raw JPEG2000 file is **not independently georeferenced**, so the current implementation is not yet a standard standalone geospatial JPEG2000 interchange format.

Four real new HGT sources from Exp66: `N39W107`, `N46E011`, `N35E138`, `S22W044`. All have 256×256 signed 16-bit elevation meter pixels; original source and extracted JPEG2000 decoded exactly, with zero meters of reconstruction error. 192 requests per codec, fixed `[0..63] * 3` 32×32 pixel window sequence; persistent file handles; codec mode order rotated; GDAL's internal cache allowed for JP2/TIFF. **No new geographic holdout:** these are the same four sources of Exp66, examined again.

## Official measurements: all geospatial metadata included in complete sizes

| Source | BH65 hybrid indexed (B) | J2K whole-image (B) | Mean BH65 ROI (ms) | Mean GDAL whole-J2K ROI (ms) | Mean GeoTIFF ZSTD ROI (ms) |
|---|---:|---:|---:|---:|---:|
| N39W107 (Colorado) | 45,484 | **25,982** | 0.0822 | **0.0813** | 0.0566 |
| N46E011 (Italy Alps) | 50,716 | **30,484** | 0.1023 | **0.0830** | 0.0567 |
| N35E138 (Japan) | 49,070 | **32,436** | 0.1485 | **0.0829** | 0.0565 |
| S22W044 (Brazil) | 51,666 | **27,590** | **0.0723** | 0.0810 | 0.0563 |

**Key conclusion:** Whole-image JPEG2000 was **strictly smaller in all four** sources, and faster on mean ROI read+decode in **three of four** cases in the *same Exp67 run*. Total bytes: BH65 = **196,936 B**, whole-JP2+geodata = **116,492 B** — whole-JP2 requires approximately **40.9% fewer bytes than BH65** on this corpus. For the Brazilian region BH65's read was slightly faster in this run, but J2K still stored far fewer bytes. GeoTIFF ZSTD was faster on the measured read path than both BH65 and whole-JP2 in all four sources, but stored more bytes than whole-JP2.

The equally indexed JPEG2000 reference was still much larger (204,069 total bytes) and substantially slower than BH65 in Exp67, which confirms that the indexing/compression strategy **changes** the answer. A codec comparison must match the *application's* access requirements without assuming the competitor has to store each window in a separate file.

## Credibility / interpretation

- This result does **not** prove the mathematically optimal or hardware-independent J2K read path: GDAL/OpenJPEG may cache source data; BH65 opens persistent file handles but **does not cache decoded tiles**. There is no equivalent per-codec memory cap. The shared GitHub Actions CPU shows between-run timing variation.
- GDAL's `JP2OpenJPEG` driver successfully served real spatial `Window(32,32)` requests from the **same full codestream**; this functionality has now been independently reproduced across all four sources.
- The geographic source is a small SRTM-derived HGT crop, not a continental-scale survey dataset or a proven production GIS benchmark.
- All schemes tested are **exactly lossless**; earlier SIREN 98× lossy-vs-lossless PNG examples are not part of this claim.
- The current BH65 uses known modular differencing, Brotli, ZigZag/LEB128 and block selection, not a novel neural compressor.

## Corrected decision on demo / LinkedIn

**Do not claim that BLKH has outperformed JPEG2000 or achieved best-in-class random-access DEM compression.** The Exp66 frozen 3/4 finding remains accurate for an explicitly **equally indexed 32px-J2K** comparator, but this is insufficient for a broader superiority announcement. GeoTIFF ZSTD also dominates ROI latency under the current limited benchmark.

**Further test opportunities:** truly independent, more varied production-scale GIS tiles; continuous or wavelet-aware coding; equal memory/caching constraints; byte-accounted geospatial standards; and application-level workloads where truly independent tile fetches and low-size headers have demonstrable value. No main-branch or public demo changes are recommended.
