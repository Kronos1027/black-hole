# BLKH Experiment 84 — real Pixelorama sprite-history sparse RGBA patches (NEGATIVE)

**Status:** independent GitHub Actions audit succeeded; ALL **56 complete stored codec archives** independently reconstructed every historical old/new RGBA sprite, including hidden RGB values under alpha=0. The additional `bsdiff4` comparator also independently reconstructed the **original PNG byte streams**, not just their decoded pixels. The preregistered win condition **failed (0/4 groups)**.

- [Preregistered protocol](BHUH_EXP84_SPARSE_PIXEL_PROTOCOL.md), committed before measuring sparse code.
- [Source/decoder implementation](experiment_84_sparse_pixel_patches.py).
- [Successful official Actions workflow and audit](https://github.com/Kronos1027/black-hole/actions/runs/40829855325).
- Reproducible GitHub Actions artifact: `exp84-pixelorama-real-sparse-patches-vs-bsdiff`, containing actual `results.json`, all **56 complete archived bitstreams**, 38 actual old/new source PNGs and full original SHA256/provenance records.
- Scope: **same 19 Pixelorama actual modified-PNG pairs from three real Git commits** as Exp83P, not an unseen dataset and not 19 independent creative teams. Pixelorama project repository is under MIT; historical file source and commit SHA pointers are recorded.

## Hypothesis and fully reversible archive

For every actual old/new same-path sprite pair, map decoded old RGBA pixels into a zero-filled new-size canvas (clipping if dimensions decreased). Produce a row-major **bit-packed 1-bit-per-pixel changed mask**; for every changed pixel save all **4 exact replacement RGBA bytes**. Every byte of the original old sprite baseline, changed mask, new RGBA values, reference dimensions, source/commit/file checksums and file headers is included in the complete archive, not supplied to the decoder from an external cache.

Two fixed representations:
- `sparse_webpold_zstd`: original old RGBA sprite atlas as WebP lossless `exact=True`, plus all bit masks and replacement RGBA bytes compressed using Zstandard level 9;
- `sparse_pngold_zstd`: original historical old PNG source file bytes compressed together using Zstandard 9, plus the same patch stream.

Strong comparison: **all 11 fully decoded legacy Exp83P exact-RGBA formats** (XOR/mod difference with WebP/Zstd, original file bytes Zstd, optimized source PNGs Zstd, whole RGBA atlases, exact 16px block dictionaries) **plus independently decoded bsdiff4 binary patches** to recover the original old and new PNG byte blobs and then the same exact RGBA source pixels. This totals **14** methods per group and **56** actual saved files.

## Complete serialized byte results (official audit)

| Real Git source group | Changed RGBA pixels / new pixels | Best BLKH sparse mask bytes | Best of 12 references bytes | Best reference | Relative size overhead |
|---|---:|---:|---:|---|---:|
| All 19 real edited PNG pairs | 5,141 / 15,676 = **32.795%** | 38,320 | **38,276** | tile16_webp | **+0.115%** |
| Pan/cursor source commit (13 pairs) | 4,477 / 12,772 = **35.053%** | 26,455 | **26,278** | tile16_webp | **+0.674%** |
| Curve/3D tools commit (4 pairs) | 250 / 1,936 = **12.913%** | 9,028 | **8,904** | atlas_webp | **+1.393%** |
| Isometric icons commit (2 pairs) | 414 / 968 = **42.769%** | 5,562 | **5,377** | atlas_webp | **+3.440%** |

**0/4** groups strictly beat their strongest reference. Success gate required aggregate win **AND** wins in at least two of three commits; **FAILED**. The overall miss is only 44 complete file bytes (0.115%) but still a loss; do **not** round or phrase this as a tie or win. The real changed-pixel fraction can be low (13% in the curve group) without making a naive mask codec smaller than well-packed whole PNG/WebP images.

## Comparability caveat

The Exp84 complete archive includes additional self-contained `old_atlas_positions`, dimensions and hashes in **every arm's** counted metadata, for both custom codecs and all references. This differs from Exp83P's header. Therefore, do **NOT** compare Exp84 file sizes directly to Exp83P absolute bytes as if only the compression algorithm changed; compare the within-Exp84 methods on the same original source group and matched metadata.

The source family is one editor project's limited historical icon selection. Both exact pixel mask encoding and binary deltas are established compression strategies. This result is neither neural compression nor a novel theoretical result. The binary-delta strong reference is a real `bsdiff4` implementation, but not a native Git packfile comparison; decoder/library executable footprints were excluded on all methods, and CPU times were not performance-controlled.

## Science decision

The experiment demonstrates that **simply transmitting modified-pixel coordinates and full RGBA values is not sufficient to beat native lossless WebP+atlas/dictionary encodings** on these 19 correlated actual edited icon pairs. Preserve both this negative finding and Exp74's bounded positive result for artificially reused tile pools; the latter does not generalize to true artwork histories tested here.

**Next step:** preselect a **different independent open-licensed game/image-asset repository** with actual same-path Git PNG modifications, inspect only Git path/status (not compression outcomes), freeze source SHAs and eligibility, and perform a fresh source-level compression test with unchanged codecs and strong original-file/binary-delta references. If no eligible source exists, publish that dataset failure rather than substituting artificial edits.

**Release gate HOLD:** main and demo remain untouched; research PR #1 stays draft; no global image-compression breakthrough announcement.
