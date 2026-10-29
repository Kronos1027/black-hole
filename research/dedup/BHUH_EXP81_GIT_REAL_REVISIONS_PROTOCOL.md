# BLKH Experiment 81 — real historical PNG source-file revisions (PRE-REGISTERED)

**Scientific protocol frozen before downloading or decoding any of these historical PNG pairs in Experiment 81.** This is a post-Exp80 pivot away from repeated Kenney pack parameter tuning. The revisions were identified by GitHub's per-file **status=modified** (not added or renamed) comparison, but pixel differences have not yet been examined. All source status/metadata and possible failures must be disclosed; do **not** select files based on resulting compressed sizes.

## True provenance and licensing

Public original Pixelorama application repo, `Orama-Interactive/Pixelorama`, with repository root **MIT License** (copyright 2019-present Orama Interactive/contributors). Files chosen only under `assets/graphics`, excluding external addon and authored splash illustration directories. Actual Git commits are independently verifiable at `https://github.com/Orama-Interactive/Pixelorama/commit/<sha>`. For each commit, resolve its **first actual parent** from GitHub's commit API, then download each exact same historical PNG path from **both commits** at fixed SHAs. Refuse unexpected parent ambiguity/HTTP failures and record original PNG file bytes' SHA256 and original decoded RGBA SHA256, dimensions, alpha counts. Never use synthetic replacements. Original source PNG metadata or ICC profiles are not the decoded RGBA compression target, but raw PNG bytes are retained and compared as a strong baseline.

Three fixed commits and exactly 19 original modified PNG paths:
- **Pan/cursor update** `9f8c9807e789bf476f95ac7045409e943b5fc328`: `assets/graphics/{blue_themes,dark_themes,light_themes}/tools/{pan.png,pan_l.png,pan_l_r.png,pan_r.png}` (12), plus `assets/graphics/cursor_icons/pan_cursor.png` (1).
- **Curve and 3D tool icon changes** `9611bf72631c3eaed8e4f0281a6628e069ecaa63`: `assets/graphics/tools/3dshapeedit.png`, `assets/graphics/tools/cursors/3dshapeedit.png`, `assets/graphics/tools/cursors/curvetool.png`, `assets/graphics/tools/curvetool.png` (4).
- **Isometric box icon update** `13c2af3a2a53bfa1439d7a6ca0e3ddbd136a91b5`: `assets/graphics/tools/cursors/isometricboxtool.png`, `assets/graphics/tools/isometricboxtool.png` (2).

Ordering: fixed group order above; within each group lexicographically sort full paths; for each filename store **old from parent then new from commit**. Total 19 historical pairs = 38 source PNG RGBA images. They are related GUI sprite sets across three actual commits **in ONE project** — NOT independent creators or a representative random sample of history. If any source pair's decoded RGBA pixels are identical (PNG-only re-encoding), record and report the count; do not pretend it is a true pixel revision. The predeclared full group must still be included rather than cherry-picked. If downloads fail, fail the run visibly.

## Fixed exact RGBA file representation and test codecs

Pack all 38 images in one **ordered shelf atlas** of max width 1024 px, padded to 16px tile alignment, without modifying source RGBA pixels (including RGB beneath fully transparent pixels). The same full manifest of commit SHA/parent SHA, original Git path, source PNG byte hashes, pixel hashes, shapes, source original license, original old/new revision order, atlas positions and sizes goes into every codec archive. Per-pair old/new dimensions may differ: the new sprite's residual references original old RGBA pixels at shared (x,y), and uses transparent black reference beyond old bounds; that exact rule is frozen and reversible.

Four new reversible delta candidates: **XOR(old,new) and unsigned modulo-256 new-minus-old**, encoded as a single exact RGBA WebP lossless atlas (`exact=True`) or ZSTD level 9 over the same atlas bytes. Unchanged "old" sprites are stored as originals and new slots carry the per-pixel residual. Full new/old original RGBA must be independently decoded and SHA256-verified from actual saved bytes.

Strong fair prior references: one whole lossless RGBA WebP atlas (`exact=True`), optimized PNG atlas, full padded RGBA atlas+ZSTD9, original **historical PNG files (both old and new) joined with explicit lengths+ZSTD9**, and independently optimized PNG member bytes joined+ZSTD9. Also include one 16px indexed exact SHA256 block dictionary with unique 16px RGBA tiles compressed by lossless WebP and full **uint16 tile reference table**, plus same with ZSTD9. A method wins only if the *complete* file is strictly smaller than the **best of all seven** unchanged conventional/dictionary references, including source PNG bytes, and each original RGBA image is losslessly recovered.

## Claim gate

The 19 pairs are correlated, derived from only **three true historical commits in one MIT-licensed app**. Besides aggregate results, report **individual per-commit results with their own counted full archive** (13/4/2 pairs), otherwise a single large revision may mask two negatives. A modest engineering success requires **aggregate full archive win and wins in at least two of three commit groups**, with all 19 file pairs reconstructed exactly (whether or not each actually changed visually). No claims about world-leading compression, all sprite histories or neural learning. No post-hoc changed-file filtering or redesign after seeing outcomes.

**Release:** draft research PR only; `main`, public demo and breakthrough announcement remain on HOLD.
