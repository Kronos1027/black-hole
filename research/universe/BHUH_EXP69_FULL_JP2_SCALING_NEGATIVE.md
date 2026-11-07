# BLKH Experiment 69 — 256² to 1024²: full JPEG2000 continues to dominate indexed lossless BLKH

**Date:** 2026-10-28 UTC. **Status:** CI reproduced; strong **negative result** for a simple wider-map size/ROI scaling hypothesis, on two previously studied geographic sources. NOT conclusive for all maps, codecs or access patterns.

**Official workflow:** https://github.com/Kronos1027/black-hole/actions/runs/40618566375  
**Implementation:** `research/universe/experiment_69_geoscale_full_j2k.py`; `.github/workflows/exp69.yml`.  
**Reproduction:** all six saved source-size conditions passed: random signed int16 self-test, complete per-block source pixel equality, full-JP2 reconstruction, 64 exact unique ROI queries and 192 repeated ROI queries per method. Evidence artifacts include complete `.bh69`, counted geospatial JP2 wrappers, actual raw `.jp2` for GDAL only, and SHA256 hashes.

## Falsifiable idea and controls

Exp68's independent horizontal stripes were much slower to decode than the GDAL whole-JP2 windows while remaining larger. One potential limitation of a **whole-JP2** design could have been that decoding a tiny ROI gets more costly as the source map grows. Exp69 tested **fixed 32×32 independently decodable** BH69 block archives with a frozen Exp65 20% fast-delta selection rule against one **continuous whole-image reversible JPEG2000 file**, including the same geospatial JSON and original signed-height offset in complete file size. Input domain: SRTM-derived Skadi elevation in signed int16 meters.

Controlled same HGT source origin (row=1200, col=1300) and crops 256², 512², 1024²; two known regions from Exp66, `N39W107` and `S22W044`, not new independent heldouts. No source was excluded by the void/NODATA policy.

64 deterministic randomly selected distinct ROI windows per source size followed by three repeated passes over the same locations, two persistent open handles with real GDAL window decode for JP2, physical seek/decode one 32² BH69 block per read. GDAL cache and I/O semantics remain different from Python BH69 and were not equalized. Sampling windows and sources are correlated; this is not a large population estimate.

## Full-file byte sizes, lossless, with georeferencing included

| Source | Pixels | BH69 complete bytes | JP2 complete bytes | Whole JP2 saving relative to BH69 |
|---|---:|---:|---:|---:|
| N39W107 Colorado | 256² | 45,508 | **26,020** | 42.82% |
| N39W107 Colorado | 512² | 186,937 | **102,682** | 45.07% |
| N39W107 Colorado | 1024² | 754,939 | **416,107** | 44.88% |
| S22W044 Brazil | 256² | 51,690 | **27,628** | 46.55% |
| S22W044 Brazil | 512² | 203,517 | **108,214** | 46.83% |
| S22W044 Brazil | 1024² | 812,936 | **446,626** | 45.06% |

**Result:** **6/6** full-JP2 complete files smaller. At 1024² source pixels, JPEG2000 saved ~45% bytes. Zero elevation error for both formats.

## Real ROI read timing, same CI run

Average milliseconds per **32×32 window**. These reflect library/caching effects and are **not** a bare algorithm decoder benchmark.

| Region | Pixels | 64 distinct ROIs: BH69 / whole JP2 (ms) | 192 repeated ROIs: BH69 / whole JP2 (ms) |
|---|---:|---|---|
| Colorado | 256² | 0.0778 / **0.0575** | 0.0772 / **0.0546** |
| Colorado | 512² | 0.0743 / **0.0552** | 0.0691 / **0.0529** |
| Colorado | 1024² | 0.0714 / **0.0578** | 0.0690 / **0.0539** |
| Brazil | 256² | 0.0688 / **0.0556** | 0.0676 / **0.0533** |
| Brazil | 512² | 0.0755 / **0.0567** | 0.0739 / **0.0534** |
| Brazil | 1024² | 0.0898 / **0.0561** | 0.0851 / **0.0532** |

**Result:** **6/6** full-JP2 window mean latencies smaller for distinct and repeated ROI reads, despite smaller complete compressed files. No increase in JPEG2000 ROI latency with this raster size was demonstrated for this workload on this host.

**Limitations:** two correlated real-world sources, a single hosted runner, no cold uncached disk measurements or equalized GDAL cache, only source sizes up to 1024², only signed int16 HGT DEM and 32px ROI granularity. The output raw `.jp2` used by GDAL has its geodata as counted sidecar/wrapper rather than native JP2 GIS tags. Wider map systems may have different behavior at file scale or ROI locality, so this is a negative result for the tested setting, not a theorem.

## Research decision / pivot

The accumulated Exp67–69 results show that simply merging more small independent blocks/stripes **does not create a competitive advantage** over an optimized and spatially aware whole-JPEG2000 codec that already supports windows. Continuing to adjust only stripe width or index sizes on the same terrain dataset is unlikely to be productive.

**Suggested next work, to measure rather than presume:** explore non-GIS application domains where a parameterized procedural representation has its own useful capabilities (editable continuous-resolution rendering, vector-like parametric sprites, zooming, rendering at arbitrary coordinates), and independently benchmark rate–distortion/compute against AVIF/JPEG XL/WebP at matched quality. Or explore large multicamera datasets with genuinely shared scene representation and counted per-image conditioning. Neither direction has a verified breakthrough yet.

**Public action:** keep demo, `main`, marketing and LinkedIn breakthrough announcements on HOLD; preserve all positive and negative evidence in the draft research PR.
