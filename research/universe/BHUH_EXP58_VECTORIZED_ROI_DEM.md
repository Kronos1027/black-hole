# BLKH Experiment 58 — vectorized lossless ROI decoder, rate/latency trade-off

**Status:** reproduced by GitHub Actions; promising **narrow indexed-tile engineering result**, not general codec state-of-the-art. The decoder was accelerated by using established NumPy vectorization for row/column modular uint16 differences, ZigZag/LEB128 and Brotli. Elevation is restored **exactly** as signed int16 meters, even across wraparound extremes.

**Reproduction:** https://github.com/Kronos1027/black-hole/actions/runs/40606022773 . Code: `experiment_58_vectorized_decoder.py`, automatic tests `.github/workflows/exp58.yml`. 20 terrain/geometry settings across 4 Skadi SRTM-derived HGT regions, each compared with three physically seekable, equally indexed archive variants (existing BH56, reversible JPEG2000, new BH58). All 60 comparative archives were confirmed lossless.

## Observed behavior at 32×32 independently decodable blocks

| Source / input area | BH58 bytes | Equal-index JPEG2000 bytes | Saving vs J2K | BH58 ROI time | J2K ROI time |
|---|---:|---:|---:|---:|---:|
| N37W119 mosaic 128×192 | 12,976 | 15,845 | 18.10% | 0.177 ms | 0.225 ms |
| N37W119 single 256×256 | 32,339 | 40,925 | 20.98% | 0.196 ms | 0.240 ms |
| N37W122 mosaic 128×192 | 17,213 | 19,587 | 12.12% | 0.177 ms | 0.262 ms |
| N37W122 single 256×256 | 45,867 | 53,269 | 13.90% | 0.187 ms | 0.267 ms |
| N40W106 mosaic 128×192 | 14,890 | 16,624 | 10.43% | 0.178 ms | 0.223 ms |
| N40W106 single 256×256 | 38,929 | 44,046 | 11.62% | 0.186 ms | 0.260 ms |
| N46W122 mosaic 128×192 | 13,680 | 17,212 | 20.52% | 0.175 ms | 0.236 ms |
| N46W122 single 256×256 | 40,169 | 47,652 | 15.70% | 0.190 ms | 0.272 ms |

**8/8 smaller AND faster vs equally indexed J2K for block size 32** in the measured short-run benchmark on the same hosted runner. This is a domain- and granularity-specific rate/latency finding; it does **not** establish superiority over an unindexed JPEG2000 mosaic or optimized JPEG2000 region-decode API. The test used only two selected corner ROIs repeated three times per file. Latencies can fluctuate significantly on shared CI hardware.

For block side 64, BH58 was generally **larger** than J2K but decoded ROIs much faster; at block side 128, it was larger but still substantially faster. The previous BH56 with 4 predictors generally compressed 32×32 and 64×64 more tightly than BH58 but had a much slower Python pixel-by-pixel ROI decoder.

## Important controls

- Both BH58 and J2K archives use the same fixed header and a 9-byte per-block index entry. Complete archived sizes include that index and per-block metadata.
- `decode_roi_from_file` physically seeks to one selected block after loading only the header and index. There is **no hidden full-file read**.
- 4 source regions, 2 correlated image arrangements each; these are **not eight independent region datasets**.
- Encoding speed was **not** fairly compared in this experiment; do not claim end-to-end speed advantage.
- Geographic/CRS metadata is externally defined for all codecs; a real product would need to count and preserve it.
- Reversible J2K can support ROI processing through different libraries, and other codecs such as JPEG-LS/JPEG XL/COG have not been rigorously compared in this exact workload.

## Decision

**Promising niche:** independently decodable **32×32** elevation tiles with low-latency read access.

**Not ready for demo/LinkedIn breakthrough claims.** First preregister geographically independent regions, test dozens of ROI locations and a stable repeated-time distribution, benchmark encoding time, include other ROI-capable codecs, and ensure a standalone geospatial file with CRS and nodata policies.
