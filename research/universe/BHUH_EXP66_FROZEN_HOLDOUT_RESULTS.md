# BLKH Experiment 66 — PRE-REGISTERED external regional confirmation

**Status:** ✅ GitHub Actions completed successfully, all source HGT input bytes and encoded files verified. This is a **narrow regional validation of an engineering trade-off**, **not** a novel compression theorem or proof of across-codec superiority.

**Pre-registration committed before data access:** [BHUH_EXP66_PREREGISTERED_PROTOCOL.md](BHUH_EXP66_PREREGISTERED_PROTOCOL.md), commit `f4cf3c1f226e40b6b90fe502b32dc31142896234`.  
**Actual complete successful workflow:** https://github.com/Kronos1027/black-hole/actions/runs/40614221031  
**Code:** `research/universe/experiment_66_preregistered_holdout.py` and `.github/workflows/exp66.yml`.  
**Evidence:** `exp66-frozen-independent-regions` zipped workflow artifact: `results.json`, 4 original source SHA-256 hashes, encoded file hashes, files for six codecs per source plus full-image J2K reference and complete benchmark measurements.

## Frozen hypothesis and source provenance

The Exp65 hybrid decision was fixed at **20% extra payload bytes per block** versus compact BH58, choosing known fixed-width BH64 when allowed. Four *previously uninspected* public Mapzen/Skadi SRTM-derived HGT regions were predeclared: `N39W107` (Colorado), `N46E011` (Alpine region, Italy), `N35E138` (Japan) and `S22W044` (southeastern Brazil). One crop per source, integer meter elevations, 256×256 pixels, fixed original HGT row/column 1200:1456, 1300:1556; 64 independently decodable 32×32 blocks, zero decoded meter error.

**Predeclared success rule:** at least **3 of 4 new sources** must produce smaller *full indexed archive bytes* and lower or equal *mean read+decode* than **identically indexed reversible JPEG2000**, at the same 32×32 independent block geometry, complete geospatial JSON and header/index overhead accounted for. There were no source exclusions.

## Official actual results

| New HGT region | BH65 20% (B) | J2K individually indexed (B) | BH65 storage gain | Mean BH65 ROI (ms) | Mean indexed-J2K ROI (ms) | Pass both? |
|---|---:|---:|---:|---:|---:|---|
| N39W107 | **45,484** | 47,553 | **4.35%** | **0.0416** | 0.1531 | Yes |
| N46E011 | **50,716** | 52,388 | **3.19%** | **0.0507** | 0.1635 | Yes |
| N35E138 | **49,070** | 54,513 | **9.98%** | **0.0725** | 0.1669 | Yes |
| S22W044 | 51,666 | **49,615** | **-4.13%** | **0.0347** | 0.1507 | **No** |

**Result:** 3/4 new geographic inputs pass the frozen rule, including the unfavorable Brazilian case without concealing it. Total across all four: **196,936 B** BH65 vs **204,069 B** individually indexed J2K, a **3.50% total storage reduction**. The latency advantage reflects a very specific implementation and benchmark, not optimized JPEG2000 ROI performance.

## Critical fairness baseline: whole-image J2K

Storing one continuous full-image JPEG2000 image WITHOUT dividing it into 64 separately JPEG2000-encoded tiles can be much more efficient. Exp66 also saved this codec, wrapping it in geospatial JSON, signed-elevation origin offsets and counted file headers. It was decoded losslessly.

| Region | BH65 32px indexed (B) | Complete JPEG2000, *not individually indexed* (B) |
|---|---:|---:|
| N39W107 | 45,484 | **25,982** |
| N46E011 | 50,716 | **30,484** |
| N35E138 | 49,070 | **32,436** |
| S22W044 | 51,666 | **27,590** |

**BH65 does NOT outperform whole-image J2K on storage; the difference is very large.** This is a caveat, not a footnote. JPEG2000 codecs can often decode regions from a full codestream, depending on driver parameters, tiling, precincts, cache and container. We must test **GDAL/JPEG2000 ROI reading on the full codestream** before claiming indexed-BH65 has a unique random-access benefit. The codec's own block metadata and JPEG2000 raw decode behavior must be measured, not assumed.

## Other relevant limitations

- BH65 was smaller than fixed 32px GeoTIFF ZSTD and DEFLATE in all four tested inputs, but its ROI **mean was slower** than these GDAL-backed TIFF implementations in all four. No all-around dominance.
- Three encode-and-save timing trials show BH65 20% around **0.0256–0.0276 seconds**, versus J2K independently indexed ~0.0173–0.0258 seconds on this runner. Both codec paths include internal code verification to different degrees; this is not a standardized encoder throughput study.
- Tests use exactly one small 256² crop per geographic source, same hosted CPU and 192 predetermined repeated ROIs, not a production representative query distribution or cold-disk test.
- All four selected sources were fixed before acquisition, but only four truly independent geographic grids; the 192 repeated reads are NOT independent geography samples.
- Horizontal EPSG:4326, source tile identifier, affine, height unit and a -32768 sentinel are stored; a complete vertical datum definition, NoData robust handling, security/fuzz testing and third-party GIS integration remain pending.
- Fixed-width deltas, Brotli, ZigZag/LEB128, spatial indexing and per-block codec selection are known algorithms; no novel neural or information-theoretic breakthrough is established.

## Release gate

**Do not upgrade the public demo with claims of superior general DEM compression, and do not post sweeping LinkedIn breakthrough claims.** The frozen holdout test established the predeclared 3/4 restricted-domain rule, but we need matched whole-JPEG2000 ROI evaluation and larger independent datasets to know whether this yields an actual application-level advantage.

Next test (Exp67): persisted full-image JPEG2000 driver reading actual 32×32 windows with GDAL, compare end-to-end ROI and bytes against BH65. The JPEG2000 codec's ability to directly read a spatial window MUST be taken seriously.
