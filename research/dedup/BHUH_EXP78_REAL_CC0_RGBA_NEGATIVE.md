# BLKH Experiment 78 — REAL Kenney CC0 RGBA UI sprites: 0/3 indexed dedup wins

**Status:** ✅ successful GitHub Actions verification, all source-art source PNG files and **21 actual complete encoded archives** reconstructed exactly to every selected RGBA pixel, **including hidden RGB under zero alpha**. **Negative compression result** for this single historical Kenney UI pack; there is no evidence that indexed 16px exact-block sharing wins over a well-built lossless atlas of these independent sprites.

**Prerecorded protocol before source data inspection:** [BHUH_EXP78_REAL_CC0_RGBA_PROTOCOL.md](BHUH_EXP78_REAL_CC0_RGBA_PROTOCOL.md).  
**Official reproducible GitHub Actions run:** https://github.com/Kronos1027/black-hole/actions/runs/40707019708  
**Implementation/CI:** `research/dedup/experiment_78_real_cc0_rgba_sprites.py`, `.github/workflows/exp78.yml`.  
**Official evidence artifact:** `exp78-real-kenney-ui-rgba-lossless`, containing `results.json`, `source_inventory.json`, three reconstructed-original atlases, and every complete compressed `.bl78` file for five conventional and two custom arms. Independently decoded and SHA256-verified in CI.

## Real independent artistic source provenance

- Original creator: Kenney; OpenGameArt mirrored [Kenney's UI Pack](https://opengameart.org/content/ui-pack), marked Creative Commons Zero / CC0 (see also [Kenney official UI pack](https://kenney.nl/assets/ui-pack)).
- Exactly the preregistered archive `https://opengameart.org/sites/default/files/kenney_ui-pack.zip`; **ZIP SHA256 `a8a14a234911eb648c062622915c93e79e94e97cb7f9f375a70f6617f1174318`**, exactly **1,229,750 bytes**.
- The source archive contained 870 PNGs; **864** satisfied the fixed alpha/dimension/sprite path rules. Use **the lexicographically first 96**, then nested sets of 32, 64, 96. Selected sprites begin `PNG/Blue/Default/arrow_basic_e.png`, `arrow_basic_e_small.png`, etc. This is a **nonrandom subset of ONE art/UI pack and mostly a single Blue variant**, NOT independent art families, real file-edit revisions, or real time-series data. Do not generalize to all Kenney packs or sprites.
- All images converted to RGBA byte arrays. Source image dimensions, source PNG file SHA, decoded RGBA byte SHA and the number of transparent and partially alpha pixels are retained in a shared full-file manifest, along with a lossless ordered-shelf packing layout. **Pixel-color values under alpha=0 were preserved** with `WebP lossless exact=True`. The target is exact decoded RGBA sprite values, not identical original source PNG metadata, ICC profiles or original PNG byte arrangement. A separate baseline preserves original individual PNG bytes.

## Locked seven compared methods

1. `tile16_zstd`: exact 16×16 RGBA tile SHA256 dictionary, all 16-bit IDs, unique tile pixels compressed by zstandard 9.
2. `tile16_webp`: same dictionary/ID table, unique RGBA tiles packed into an exact-lossless WebP dictionary atlas.
3. `atlas_png`: original unmodified sprites placed in one transparent-alpha PNG atlas.
4. `atlas_webp`: same atlas, one WebP lossless `exact=True` file.
5. `atlas_raw_zstd`: complete exact atlas RGBA bytes + zstd9.
6. `original_png_zstd`: ALL original individual source PNG files, length-prefixed and jointly zstd9-compressed (preserves original PNG bitstream bytes).
7. `optimized_png_zstd`: independent optimized PNGs length-prefixed and jointly zstd9-compressed.

**Each file separately includes the same source origin, sprite dims, RGBA hashes, full atlas layout and source data hashes.** All seven were decoded exactly for each N=32/64/96; metadata-size accounting is equalized by shared manifest. No lossy-vs-lossless comparisons.

## Official complete archive byte sizes

| Format (complete bytes) | 32 sprites | 64 sprites | 96 sprites |
|---|---:|---:|---:|
| tile16_zstd | 27,874 | 50,049 | 77,962 |
| **tile16_webp** (best BLKH indexed) | **23,197** | **41,816** | **64,073** |
| atlas_png | 37,686 | 69,429 | 106,450 |
| **atlas_webp (best standard)** | **21,678** | **40,153** | **61,146** |
| atlas_raw_zstd | 27,210 | 49,899 | 77,138 |
| original_png_zstd | 28,449 | 51,743 | 76,783 |
| optimized_png_zstd | 46,267 | 85,538 | 125,353 |

**Winner: conventional full WebP atlas in all 3 nested subsets.** The BLKH indexed dictionary is larger by **+7.01%, +4.14%, +4.79%** respectively. The prespecified all-three win rule fails **0/3**.

### Deduplication *did* happen; it still wasn't efficient enough

| Subset | Stored 16×16 positions | Unique tiles | Duplicate references | Counted index bytes |
|---|---:|---:|---:|---:|
| 32 | 720 | 204 | 516 | 1,440 |
| 64 | 1,152 | 325 | 827 | 2,304 |
| 96 | 1,664 | 543 | 1,121 | 3,328 |

Even with extensive **exact repeated transparent-background tiles**, WebP's spatial/entropy mechanisms compressed the same atlas more efficiently without an explicit tile ID table and dictionary relocation. Duplicate-reference *count* alone is thus insufficient for a meaningful codec win.

## Limits and research decision

- The selected 32/64/96 are **nested**, correlated subsets from **one** Kenney UI pack, sorted lexicographically; this is **not** three statistically independent source groups. Results do not establish a global sprite-asset conclusion.
- The source archive is a **single creator's static UI collection**, not an actual version-control history of edited PNG sprites. No artificial edits were introduced, but neither were real frame revisions measured. Actual edited asset histories remain untested.
- Existing generic content-addressed tile dictionary + WebP combinations are not a new theoretical compressor.
- No throughput-optimized native decoder comparison was performed. Encode/decode times are present in the results, from a single shared GitHub Actions CPU.
- Source ZIP hash, SHA256 of each original PNG and each exact RGBA decoded image, atlas file hashes and all 21 full archive hashes are present, making this negative result inspectable.

**Release decision:** do not alter public demo, `main` or announce broad BLKH compression superiority on LinkedIn. Preserve source-negative evidence.

**Next falsifiable test:** The same source pack has authored **real four-color UI variants** (Blue, Green, Red, Yellow) of the same shape filenames. Pre-register deterministic *shape-matched* variant ordering, and compare the existing **unchanged** Exp78 codecs on **32, 64, 96** images (8/16/24 shapes ×4 colors), to isolate whether actual artist-authored alternate color versions create greater shared structure. Again: these are real color variants, NOT chronological revisions or independent artists.
