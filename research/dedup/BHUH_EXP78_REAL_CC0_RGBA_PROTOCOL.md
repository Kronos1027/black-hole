# BLKH Experiment 78 — original CC0 UI sprites, true RGBA preservation (PREREGISTRATION)

**Fixed and committed before examining source pixels or compression results.**

## Data, creator, license and independence

Acquire exactly `https://opengameart.org/sites/default/files/kenney_ui-pack.zip`, a Kenney UI Pack mirror published on OpenGameArt by Kenney and marked **Creative Commons CC0**; source: https://opengameart.org/content/ui-pack and creator: https://kenney.nl/assets/ui-pack. Verify the ZIP size <20 MB, log its SHA256, require a nonempty PNG set and never replace URL or artwork to improve outcomes.

Source selection is deterministic **before inspecting outcomes**: lexicographically sort ZIP member paths ending in `.png`, exclude filenames/parents containing `spritesheet`, `preview`, `sample`, `sheet`, `tilemap`; require each decoded image's width and height between **16 and 512** and **at least one pixel with alpha <255**; take the **first 96 valid images**. If fewer than 96 qualify, the preregistered run fails, with a full source inventory saved; **do not substitute**. (A filtered collection must not be called a random representative sample of the pack.) Record ZIP hash and original member file hash, raw pixel SHA256, dimensions and alpha counts for each sprite.

Three nested, **correlated subsets of 32, 64 and 96 real unique source sprite PNGs** from the same one Kenney UI pack; they are *not* independent artist families or natural revisions. **Do not claim** evidence on authored sprite change histories, videos, or global image compression. Original source PNG encoded bytes are available to a valid conventional archive comparator.

All sprite content is decoded into exact straight RGBA uint8. The **actual RGBA values including RGB underneath fully transparent pixels are required to survive exactly**; source PNG metadata/ICC are not part of the target representation, and raw source PNG-byte preservation is a separate archive baseline. Never silently replace transparent RGB pixels.

## Frozen formats and deterministic atlas

Deterministically construct one spatial atlas per subset with **no rescaling, pixel changes or overlaps**, using first-fit ordered-shelf packing with maximum row width **1024 pixels**, sprite positions/shape written in a shared JSON manifest. Padded atlas pixels are transparent black (0,0,0,0), source sprite pixels exactly retained. Manifest records sorted original filenames, exact RGBA SHA256, file SHA256, bounds, alpha statistics, source ZIP SHA256, title/license, atlas geometry. Save all actual counted bytes.

Predeclared custom candidate methods:
1. `tile16_zstd`: 16x16 atlas tile, exact four-channel SHA256 deduplication in first-occurrence order, 16-bit tile IDs for **all** atlas cells, unique RGBA tile bytes compressed with Zstandard level 9.
2. `tile16_webp`: same exact indexed 16x16 RGBA dictionary, unique RGBA tiles packed in 16-column WebP lossless atlas (with full transparent-RGB preservation requested); if WebP modifies any hidden-RGB or alpha, **fail the custom variant visibly**, do not call it lossless.

Strong **real lossless** comparators, with **the same metadata and ordered set of original assets**:
- one optimized PNG RGBA atlas;
- one WebP-lossless RGBA atlas (exact transparent RGB preservation required);
- one atlas raw RGBA buffer compressed by Zstandard level 9;
- individual original PNG file bytes concatenated with a fixed length prefix, globally compressed by Zstandard level 9 (this can often beat a naive dictionary);
- individually optimized lossless PNGs globally compressed by Zstandard level 9, checked by decoding each image.

For all seven arm types, decoding saved bytes must recover each of the N original sprite RGBA pixel arrays with exact dimensions, ordering, alpha and raw RGBA SHA256 (NOT merely the padded atlas). No estimated bitstream sizes or uncounted dictionary. Report actual complete byte sizes, encode/decode wall-clock times on same runner (not calibrated performance), unique tile count, index bytes, and exact hashed output per arm.

## Predeclared claim rules

**Win at a given N** means the smaller of two exact-dictionary codecs has strictly fewer **whole-file bytes including manifest and all tile indexes** than the smallest *valid* conventional reference, including raw original PNG files packed by zstd. For a narrow verified engineering benefit, demand wins in **all three nested subsets (32,64,96)**, while explicitly stating they represent only **one artist pack**, not three independent datasets. If any baseline is invalid/inexact, disclose it and do not represent it as a lossless codec. If one required baseline codec cannot preserve exact RGBA here, the workflow must report a **protocol limitation/failure** rather than switch to a lossy comparison.

A finding on the real source pack does not imply the technique wins on actual authored revisions or original source PNG bytes. No `main` merge, no demo update and no LinkedIn breakthrough post from this pilot alone.
