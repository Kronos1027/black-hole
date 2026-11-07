# BLKH Experiment 80 — lossless within-shape colorway delta on real CC0 sprites

**Frozen before any Experiment 80 compression measurement.** Follow-up to Exp79's 0/3 size results; dataset previously inspected and correlated, **not** an independent holdout or a new theory.

## Data and representation

Exactly the fixed Kenney UI Pack ZIP SHA256 `a8a14a234911eb648c062622915c93e79e94e97cb7f9f375a70f6617f1174318`, with the **same 24 first eligible real shape suffixes** and four original artist-produced color variants in frozen order **Blue, Green, Red, Yellow**. Nested 8/16/24 shapes = 32/64/96 PNG RGBA assets. Source rights Kenney CC0 via https://opengameart.org/content/ui-pack. No generated edits, color palettes substituted, rescaling or source member selection after looking at compression.

Each shape must have identical dimensions across its four artist-authored colorways; if this fails, **abort without replacing the shape**. The target fidelity is **exact 8-bit RGBA for all four channels, including hidden RGB when alpha=0**; original compressed PNG byte equivalence is a separate reference.

## Four fixed source-aware transforms

For each shape's original Blue RGBA bytes, store its original pixels unchanged as the baseline anchor. For the three other same-shape colorways compute a **bytewise and reversible per-pixel residual relative to the corresponding Blue RGBA pixel at the same coordinates**. Both alpha and RGB residuals must be included:

1. `xor_webp`: transformed Blue plus the three per-channel XOR residual sprites, packed into the **same exact ordered RGBA shelf atlas** as Exp79, compressed as one WebP lossless `exact=True` file.
2. `xor_zstd`: same transformed atlas, raw RGBA bytes compressed Zstandard level 9.
3. `mod_webp`: per-channel unsigned modular difference (current−Blue) modulo 256, same atlas and WebP lossless `exact=True`.
4. `mod_zstd`: same modular difference and Zstandard level 9.

Transparent black padding and original placement are unchanged. The **same complete source manifest**, original file SHA256, original decoded RGBA SHA256, positions, atlas dimensions and shape/color labels are stored in each custom and reference archive. The decoder has **no access to the original ZIP**: read the saved residual atlas, recover each Blue sprite, invert subsequent residuals, and verify original full atlas and every original sprite's RGBA SHA256. Never zero hidden RGB or normalize alpha.

## Strong unchanged references

Exactly the seven previously audited Exp79 formats, run on the **same original 32/64/96 colorway images and Exp79 packing**: tile16 zstd/WebP dictionaries, full optimized PNG and exact RGBA WebP atlas, raw full atlas zstd, joint zstd over original source PNG file bytes, joint zstd over optimized PNG member bytes. These include full metadata, atlas coordinates, and duplicate references. The **best standard benchmark** is the minimum complete file bytes across **all seven** existing formats (including any previous BLKH dictionary), not just PNG or a single naive compressor. We are measuring a *new differential representation's* incremental benefit relative to the best previously available option.

For each N, 11 complete actual archives (4 new + 7 established) must be independently decoded, SHA-verified and counted. The protocol success gate is a strictly smaller **complete transformed archive** than the **best of seven references in all 3 nested sizes**, and explicit disclosure that these are 3 correlated subsets from 1 previously seen CC0 art pack, not independent source families or real time-history revisions. If mixed/negative, publish exactly. No source-dependent transformation switching or quality sacrifice.

**Public release/demo/main/LinkedIn claims: HOLD.** This experiment does not prove a general image codec, new neural compression, or novel theoretical method.
