# BLKH Exp100 — frozen alternative residual representation on real PNG Git edits

**Protocol frozen before running the new representation benchmark.**
**Status:** exploratory/retrospective, not a blinded independent holdout; public release **HOLD**.

## Hypothesis

In exact historical PNG sprite edits, changed RGBA values can be costly to encode even when a reference image exists. Encode the same changed pixels as either **absolute RGBA bytes**, **bitwise XOR with predicted old RGBA**, or **modulo-256 channel subtraction**. A transformation can expose low-entropy differences to the same Zstandard backend. Avoid cherry-picking: run the full frozen 2×3 factorial grid.

## Frozen sources, methods and comparisons

Use original upstream Git parent/child PNG edit pairs and metadata fixed in Exp87 for the Wesnoth source: all 21 pairs, and its three overlapping groups. Reuse exactly the Exp93 pixel-mismatch-minimizing shift selector and Exp95 per-pair Zstandard patch-minimizing selector, ±8 pixels. For each selector test all three residual modes raw, xor, sub with identical predicted old pixels, masks, Zstandard level and self-contained WebP-lossless old atlas, plus complete source provenance metadata. No tuning on results. Full 6 candidate archives **and all 14 Exp84 exact RGBA reference codecs** saved per group (80 archives total); measure whole archive including metadata, not patch payload only.

### Binary format

New magic `BL100`, explicit version byte 1 (raw) / 2 (xor) / 3 (modulo-256 subtract). Identical fixed <int8 dx, int8 dy, uint32 changed count> record followed by little-bit-order bitmap and four residual bytes per changed pixel; Zstandard compresses the concatenated stream. Decode with the original old atlas, reverse the transformation and verify exact hidden RGB below transparent alpha. Source PNG **file bytes** are not claimed to be preserved; target is exact decoded RGBA.

### Fair comparator and release gate

Use the minimum measured *complete* reference archive (all 14 existing codecs) in **each** group. Also report old motion codecs for context. Independently fetch upstream source originals a second time, check their provenance and every persisted archive byte length, SHA-256, full metadata and exact RGBA. All source groups must be reported, including losses. Aggregate/subgroups overlap and cannot be counted as 4 independent replications. The test is exploratory on previously seen historical PNGs; wins here do not establish novelty/generalization.

No claim of compression breakthrough, demo or professional superiority before new preregistered independent source families, best-codec fairness, full computation/memory costs and rate-distortion at matched error for lossy codecs. Report all negatives.

## Future transfer without algorithm changes

If core implementation is sound, reuse the same frozen six methods without changing them on the previously studied Godot/SuperTux history (transfer stress tests only), and later on new independently preregistered original source revisions. Never relabel previously seen data as unseen.
