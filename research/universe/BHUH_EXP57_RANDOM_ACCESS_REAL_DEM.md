# BLKH Experiment 57 — indexed lossless DEM with TRUE physical byte-seek ROI

**Status:** GitHub Actions reproduced the 60-archive experiment; a domain-specific storage trade-off, not a new general-purpose compression theorem or breakthrough.

**Run:** https://github.com/Kronos1027/black-hole/actions/runs/40605672478

## What changed from Experiment 56

Exp 56 compressed the full tile corpus as one Brotli stream, so it **did not support random access to independently decodable individual tiles**. Exp 57's `BH57I` archive has a counted fixed header and an explicit per-block byte offset/length/codec index. `decode_roi_from_file` uses Python file seek and reads only **the complete index and one compressed block**. It never reads or decodes the rest of the archive. This was verified by re-reading the actual `.bh57` files.

The two separately indexed pure codecs use the **same** archive/index overhead and block sizes: BH56 reversible predictive/Brotli vs reversible JPEG2000. A hybrid codec chooses the smaller per-block representation and stores a per-block codec ID; it has the same per-block index overhead. All format sizes include compressed headers, indexing and each block's signed-elevation metadata.

Four existing real SRTM-derived Skadi HGT regions were used, N37W119, N37W122, N40W106 and N46W122, with two image arrangements (128×192 mosaic built from six adjacent 64×64 tiles and one contiguous 256×256 area); 32, 64, and 128-pixel block sides, where geometrically valid. These are **correlated variants of four sources**, not twenty independent regions. Total: 4 regions × 5 layouts × 3 modes = **60 exactly lossless archive tests**.

## Reproduced size examples

| Source and geometry | Block | BH56 indexed | J2K indexed | Hybrid indexed | Observation |
|---|---:|---:|---:|---:|---|
| N37W119 128×192 | 32 | **11,001** | 15,845 | 11,001 | BH56 smaller |
| N37W119 128×192 | 64 | 9,807 | 9,874 | **9,760** | adaptive small gain |
| N37W119 256×256 | 32 | **27,363** | 40,925 | 27,363 | BH56 smaller |
| N37W119 256×256 | 128 | 23,117 | **20,747** | 20,747 | J2K smaller |
| N40W106 128×192 | 64 | 10,626 | 10,493 | **10,487** | adaptive tiny gain |
| N40W106 256×256 | 128 | 26,621 | **23,215** | 23,215 | J2K smaller |
| N46W122 256×256 | 64 | 31,046 | 31,543 | **30,989** | adaptive small gain |

All values are **complete indexed archives in bytes**, not just individual block payloads.

Across the four geographic sources:
- **32-pixel blocks:** BH56 was smaller than an identically indexed reversible J2K archive in **8/8 cases**.
- **64-pixel blocks:** BH56 was smaller in **6/8 cases**.
- **128-pixel blocks:** J2K was smaller in **4/4 cases**.
- Hybrid never exceeded the best pure archive at fixed block size, by construction of the per-block selector. Its *extra* gain beyond the best pure archive was generally small.
- The **full-mosaic JPEG2000 archive with no per-block index** is usually smaller than an indexed archive; therefore, smaller indexed BH56 than indexed J2K does **not** mean it is smaller than optimally packed, non-random-access J2K.

## Honest read-time trade-off

Sample official runner measurements when reading header+index+one physically stored block:

| Source and geometry | Block | BH56 median read+decode | J2K median read+decode |
|---|---:|---:|---:|
| N37W119 mosaic | 32 | 1.17 ms | **0.23 ms** |
| N37W119 mosaic | 64 | 4.50 ms | **0.41 ms** |
| N37W119 256×256 | 128 | 17.61 ms | **0.99 ms** |

Benchmarks use a shared GitHub Actions runner, three chosen ROI coordinates and three repetitions, not a stable production latency or randomized ROI benchmark. BH56's pure-Python prediction/varint decoding is slower than native JPEG2000. A small on-disk advantage **cannot justify claims of overall performance leadership** without measured query workload.

## Verdict

**Technically significant implementation milestone:** real byte-seek, independently decodable lossless random-access DEM tiles, with an auditable format and sensible fairness constraints.

**Not yet grounds for a demo claiming superior compression:** indexed J2K often decodes faster by an order of magnitude, best block size depends on data, and winning 32×32 storage comparisons does not establish general superiority. A future useful design should optimize the CPU decoder and evaluate true GIS-style ROI workloads, decode cache behavior, geospatial metadata, and spatially indexed JPEG2000/COG/JPEG-XL competitors.

The PR remains a draft; original BLKH `main` and public demo remain unchanged.
