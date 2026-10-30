# BLKH Experiment 79 — matched authored colorway variants of one CC0 UI pack (preregistered before Exp79 compression)

**IMPORTANT:** This is a follow-up hypothesis generated after Exp78 results and its full source directory inventory were inspected. It is **exploratory on the SAME ONE Kenney artist pack**, not an independent holdout. The selection and parameters below are now frozen before evaluating variant pixel similarities and compression outcomes.

## Source and reproducible selection

Exactly the same SHA256-validated Kenney UI Pack historical ZIP as Exp78, from `https://opengameart.org/sites/default/files/kenney_ui-pack.zip`, expected complete source ZIP SHA256 `a8a14a234911eb648c062622915c93e79e94e97cb7f9f375a70f6617f1174318`. Author Kenney; license CC0; provenance https://opengameart.org/content/ui-pack. **Fail** on source hash mismatch or unavailable exact source member; do not silently replace.

Real creator-authored color variations are stored at paths `PNG/{color}/Default/{shape_file}.png` where `color` is the **frozen order** `Blue, Green, Red, Yellow`. Determine the exact intersection of PNG suffixes across all four colors. Lexicographically sort the intersection; select the **first 24** suffixes with all color PNGs having width/height in [16,512], nonopaque alpha, 8-bit straight RGBA pixel decoding. In source order **shape-major then fixed color order**, build three nested datasets of **8 shapes × 4 colorways =32 assets, 16×4=64 assets, 24×4=96 assets**. These are actual authored related UI sprites, not generated recolors and not historical commits/revisions. Record all source member hashes and each exact RGBA SHA256. Do not reorder or retrospectively choose apparently compressible shapes.

## Frozen fair coding and success criterion

Reuse the **exact same Exp78** deterministic 1024-pixel-wide shelf packing, tile size 16, SHA256 exact RGBA block dictionary compressed using WebP lossless with `exact=True` or Zstandard9, and all five equally metadated strong baselines (WebP lossless whole atlas, optimized PNG atlas, raw atlas+zstd, concatenated original PNG files+zstd, individually optimized PNGs+zstd). Use imported unchanged `encode_case` and `decode_case` from Exp78. **All seven actual serialized archives must reconstruct every original RGBA byte including hidden colors**, with source images and JPEG/WebP lossless decoding independently checked in CI; no missing alpha, image dimensions or source PNG metadata equivalence claims.

Record per-variant source SHA256, unique 16×16 tiles and all index bytes. Winner at N if best complete custom indexed dictionary (incl. metadata/index) is strictly smaller than smallest valid comparable conventional reference. Limited exploratory success rule: win in all three nested sets. Never call these independent n, a held-out family, or a general theory. If result negative, explicitly publish.

**No merge to main or changes to demo/LinkedIn announcement** based only on this result. Actual version-control histories and multi-artist independent samples remain separate future research.
