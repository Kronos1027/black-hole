# BLKH Exp85 — independent Mindustry Git icon-revision transfer (PREREGISTERED)

**This protocol is frozen before downloading, opening, or measuring any historical PNG pixel data.** Source selection used Git diff **status=modified** and paths only, not pixel content or measured compression results.

## Independent original source

New repository not used in Exp81–84: `Anuken/Mindustry` at https://github.com/Anuken/Mindustry (root GPL-3.0; record source license and creator). Three nonmerging historical source commits, frozen with their **actual single first parent** resolved through GitHub's commit API; select **the first eight lexicographically sorted, same-path, Git status=modified original PNG files under `core/assets-raw/sprites/`** at each commit. Refuse unpinned paths, missing files and any change in original Git status. The exact pinned commits:

1. `b9ccc528e933e705d7f737c781e57176aead5cc3` — 8 unit sprite PNG edits (all eight modified sprite paths).
2. `b2a4cb42cf9c3a435ec5915e5b8a722c9e0b36f4` — first eight sorted modified `core/assets-raw/sprites/` PNG files (bridge conveyors, conduits etc).
3. `f0f7b5aaf7edce01c023f408d0e48a2c8f552e0e` — first eight sorted modified `core/assets-raw/sprites/` PNG files (overflow gate, drills, power etc).

Exactly **24 actual same-path historical old/new PNG pairs**, 48 PNG source files from ONE independently sourced creator/repo. The 3 commits are correlated and NOT 3 independent source repositories. The source family is external to the Pixelorama data used to devise Exp84. Some files may be purely PNG-container rewrites with identical RGBA; report this openly, count them in the frozen dataset, and do not silently substitute them or remove favorable/unfavorable data. Confirm *exact first-parent Git modified status* for all 24 paths using actual GitHub compare API. Record parent/commit SHA and every original source PNG SHA256, decoded RGBA SHA256, dimensions, path and changed-pixel fraction.

## Frozen codec and integrity test

Reuse byte-for-byte unchanged **two sparse original-old+patch bit-mask RGBA codecs** from Exp84:
- `sparse_webpold_zstd`, 16/any-bit source atlas of old RGBA pixels encoded as WebP lossless `exact=True`, bit-packed 1-bit changed mask, 4 bytes for every changed RGBA pixel, Zstd9;
- `sparse_pngold_zstd`, original old PNG bytes+Zstd9 and the identical sparse patch stream.

Compare with all **12** fixed previous strong lossless reference arms, including:
- 4 reversible XOR/modular true RGBA atlas differences compressed as WebP/Zstd;
- lossless complete RGBA atlas PNG/WebP, raw RGBA atlas +Zstd;
- exact original historical PNG bytes+Zstd and optimized individual PNG files+Zstd;
- 16px exact-tile SHA256 dictionary + WebP/Zstd;
- actual binary diff of *original old/new PNG blobs* using bsdiff4, compressed jointly with Zstd9.

Every method must serialize complete matching source manifest + old atlas geometry, exact original per-file checksums, source identifiers, pair ordering and all dictionary/index/binary-patch bytes. **No changed predictor, mask, block size, optimizer, quality knob, atlas order or source filtering relative to Exp84.**

Save **four separately encoded bundles** (all 24 pairs plus 8 pairs in each individual commit) × **14 methods** = **56 actual complete archive files**. Independently reopen and reconstruct all 48 source-version RGBA arrays *exactly, including hidden RGB when alpha=0*, and every original old/new PNG byte for bsdiff. Re-download all source bytes and verify the original Git changed status in a separate CI step. Preserve the original member-by-member bytes in the artifact for audit.

## Predeclared decision rule

Narrow external-dataset victory: best of two sparse candidates strictly smaller **in complete counted file bytes** than min of all 12 strong controls **both for aggregate 24-pair file and at least 2/3 commit groups**. A mixed result is a failure of this gate, even if a small sample wins. This only tests ONE second repository (Mindustry), not the population of all version histories. Pixelorama Exp84 is a separate already-known negative and must be included in final interpretation. Neither success nor failure can claim a universal invention, novel neural compressor or production throughput.

**No public demo/main change or LinkedIn breakthrough:** only protected draft PR; original PNG copyright/license attribution retained.
