# BLKH Experiment 84 — exact sparse RGB-alpha change masks on true Git PNG history: NEGATIVE

**Status:** ✅ official [GitHub Actions PASSED](https://github.com/Kronos1027/black-hole/actions/runs/40829855325), including second independent decoder/source-file audit of **56 actually saved compressed archives**; all true original 19 old/new Git PNG pairs reconstruct byte-exact decoded RGBA, and the BSDiff4 baseline additionally reconstructs **original old and new PNG file bytes byte-for-byte**. **0/4** commit/aggregate groups beat the best valid reference.

- **Pre-registered** protocol (before new patch sizes): [BHUH_EXP84_SPARSE_PIXEL_PROTOCOL.md](BHUH_EXP84_SPARSE_PIXEL_PROTOCOL.md).
- Source code: `research/dedup/experiment_84_sparse_pixel_patches.py`, CI: `.github/workflows/exp84.yml`.
- Artifact `exp84-pixelorama-real-sparse-patches-vs-bsdiff` contains source PNGs, four complete group results, each of 14 actual compressed files, full counted provenance and per-method hashes.
- Source: exact 19 genuine Pixelorama `Orama-Interactive/Pixelorama` MIT repo historical Git PNG modifications from Exp83P, 3 related commit groups, with all Git first-parent and old/new image SHA256 verified. **Same already studied 19 pairs, NOT independent holdout.** Original source pixels, alpha and transparent RGB under alpha=0 preserved exactly.

## Frozen sparse representation, all bytes counted

For each old/new pair, store the **unchanged old RGBA image** in a single WebP lossless atlas (`exact=True`) or as original source old PNGs length-prefixed and ZSTD9-compressed. New image: align old image top-left, pad with transparent RGBA zeros if new dimensions grow, emit one **bit per new pixel** indicating if *any* of 4 RGBA channels changed, then the **four original RGBA bytes only for each changed pixel** in chronological row-major order. Concatenate all masks/color bytes and ZSTD9-compress the patch stream. All patch lengths, original old bytes, source dimensions, checksum metadata and the full index or atlas are counted. True decoded images are checked *after saving*.

References: the **same eleven full-file codecs from Exp83P** (XOR/modulo WebP+ZSTD, whole PNG/WebP lossless atlases, raw atlas+ZSTD, historical original PNG bytes+ZSTD, individually optimized PNG+ZSTD, exact tile16 WebP/ZSTD dictionaries) plus an independently implemented **bsdiff4** reference encoding *both original PNG source byte streams* with binary source patches + ZSTD9. All 14 approaches receive the same complete source history metadata; all 56 saved streams were independently reloaded and decoded.

## Official within-run results (bytes include complete source-history manifest)

| Actual historical Git image group | Target pixels changed | Best sparse patch bytes | Best of 12 exact lossless reference bytes | Best reference | Relative sparse overhead |
|---|---:|---:|---:|---|---:|
| All 19 file revision pairs | **5,141 / 15,676 (32.80%)** | **38,320** | **38,276** | tile16_webp | +44 B (+0.115%) |
| Pan/cursor 13 pairs | 4,477 / 12,772 (35.05%) | 26,455 | **26,278** | tile16_webp | +177 B (+0.674%) |
| Curve/3D 4 pairs | 250 / 1,936 (12.91%) | 9,028 | **8,904** | atlas_webp | +124 B (+1.393%) |
| Isometric 2 pairs | 414 / 968 (42.77%) | 5,562 | **5,377** | atlas_webp | +185 B (+3.441%) |

Best sparse method on all four groups: **`sparse_webpold_zstd`**. The source PNG binary-delta reference `bsdiff_png_zstd` took respectively **78,278 / 64,761 / 10,206 / 6,147 B**, larger than the corresponding best source-RGBA atlas/tile codec for this task. BSDiff4 restores **original PNG bytes**, a stronger target than decoded pixels, and its poorer size here is not a universal criticism of all binary deltas or Git packfile compression.

**Official success criterion:** aggregate strict win plus at least two of three per-commit wins. **Observed: zero aggregate wins and 0/3 commit wins; criterion FAILS.** Within these samples, even localized genuine changes did not produce a stored-patch win over an optimized conventional atlas or compact unique 16px tile dictionary.

## Significant methodological caveat

**Do NOT compare the absolute Exp84 archive sizes with Exp83P's earlier sizes.** Exp84 serializes an extra full `old_atlas_positions` list and old atlas SHA/shape fields in the **shared complete manifest of ALL 14 arms**, even the reference codecs that do not need them. This duplicates some source information already in the regular manifest, increasing all tested complete archives by multiple KB relative to the earlier Exp83P run. Its actual **within-Exp84 comparison** uses the same manifest across candidate and reference files and is valid under that explicit common-metadata convention; however, it is not optimal packaging and is not directly comparable to Exp83P's reported 27,215/26,492 B values. A future complete-file benchmark should use minimally sufficient *per-codec* metadata with all common required source provenance accounted for, explicitly compare against tight Git packfile/binary-delta references and avoid favoring a format via unnecessary header duplication.

Other limitations: only 19 real changed pairs across 3 commits from **one editor's asset family**; target is decoded RGBA rather than metadata/PNG file identity (except BSDiff source baseline); no pixel editing fidelity outside original 38 sprite versions, no codec decoder software footprint/interactive latency, no independent third-party project replication.

## Decision

**Reject on these sources:** all-pixel changed-bit-mask + full changed RGBA values plus a source frame do not beat the best complete file-size method even when only ~13% of pixels change in the curve/3D group. It may be useful for *selective editing or random access*, but those benefits were NOT measured, so do not claim them.

**Research next step:** stop optimizing the same 19 known image pairs post hoc. Investigate real **independently authored** changes in a second project, try a practically relevant asset-update workload with documented scene/UI revision times, and include proper binary/git packfiles/avif-ls or deduplicated archives. Keep (\text{main}), public demo and broad scientific publicity on HOLD; PR #1 remains draft.
