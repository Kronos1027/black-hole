# BLKH Experiment 79 — creator-authored CC0 colorway variants, exact RGBA: 0/3 wins

**Status:** ✅ official GitHub Actions PASSED, all 21 complete source-backed archives independently decoded and checked pixel-for-pixel, including hidden RGB of zero-alpha pixels; **negative** storage result. This is a test of **real artist-created colorway variants of the same UI elements**, but **not an artist's chronological version-control history** and not three independent datasets.

**Precommitted protocol:** [BHUH_EXP79_REAL_COLORWAYS_PROTOCOL.md](BHUH_EXP79_REAL_COLORWAYS_PROTOCOL.md), committed before Exp79 compression. Exp79 was designed **after seeing** the Exp78 source-pack inventory and its failures, so is an **exploratory follow-up on known source data**, not an independent held-out test.  
**Successful official GitHub Actions:** https://github.com/Kronos1027/black-hole/actions/runs/40707860237  
**Code:** `research/dedup/experiment_79_authored_colorways.py`, using the **unchanged** seven Exp78 encoders and decoders in `experiment_78_real_cc0_rgba_sprites.py`.  
**Artifact:** `exp79-real-kenney-authored-color-variants`; actual 21 encoded complete `.bl78` files, original source RGBA atlases, source inventory and result JSON with all SHA256 hashes, reconstructed pixels and byte counts, independently audited in CI.

## Provenance, selection, independence

- Source: actual Kenney creator-produced CC0 [UI Pack](https://kenney.nl/assets/ui-pack) mirrored on [OpenGameArt](https://opengameart.org/content/ui-pack), exact archive `https://opengameart.org/sites/default/files/kenney_ui-pack.zip`.
- **Fixed exact ZIP SHA256** `a8a14a234911eb648c062622915c93e79e94e97cb7f9f375a70f6617f1174318`, also the Exp78 source archive. No substitute images.
- From `PNG/{color}/Default/{filename}`, use true real colorways in **frozen order** `Blue, Green, Red, Yellow`; take the first lexicographically sorted **8 / 16 / 24 shapes** with all four author-provided PNG variants and valid transparency and pixel geometry.
- 8×4=32, 16×4=64, 24×4=96 RGBA sprites. Ordering is **shape-major then color**, and the same spatial shelf atlas is used by ALL methods. These subsets are **nested and correlated**; only ONE creator/pack and not historical revisions or independent art families.
- The targets are exact straight RGBA pixel arrays decoded from original PNG files. Each original source PNG byte hash and RGB-alpha pixel hash and geometry are recorded; all seven variants must reconstruct these exact pixels, including RGB channels under alpha=0. Original PNG bitstream bytes are also retained as a separate standard archive control.

## Complete byte-counted comparison

| Lossless method | 8 shapes / 32 RGBA sprites | 16 shapes / 64 RGBA sprites | 24 shapes / 96 RGBA sprites |
|---|---:|---:|---:|
| BLKH tile16 + Zstd9 | 23,894 | 48,681 | 73,408 |
| **BLKH tile16 + WebP lossless dictionary** | **19,960** | **41,031** | **63,512** |
| Atlas PNG optimized | 29,582 | 66,151 | 94,114 |
| Atlas WebP lossless exact | 19,559 | 40,216 | **58,839** |
| Atlas raw RGBA + Zstd9 | 22,724 | 46,649 | 67,446 |
| Original PNG source files bundled + Zstd9 | **19,346** | **39,322** | 58,971 |
| Optimized separate PNGs bundled + Zstd9 | 30,613 | 70,698 | 100,380 |

**Results:** BLKH best custom indexed archive vs *best of all five* actual conventional schemes:

| Shapes | Exact RGBA sprites | Best BLKH indexed bytes | Strongest conventional complete bytes | BLKH extra bytes | Status |
|---|---:|---:|---:|---:|---|
| 8 | 32 | 19,960 | **19,346**, source PNGs+Zstd | **+3.17%** | fail |
| 16 | 64 | 41,031 | **39,322**, source PNGs+Zstd | **+4.35%** | fail |
| 24 | 96 | 63,512 | **58,839**, full WebP atlas | **+7.94%** | fail |

**0/3 predeclared wins.** All seven complete archive files for each source subset were decoded exactly. The output remains a real *pixel-lossless* image-asset compression evaluation, not original-PNG metadata/ICC equivalence.

## Why matching variants did not create the hoped-for reuse

| Colorway set | Atlas tile positions | Unique 16×16 RGBA tiles | Repeated tile references | Counted uint16 index bytes |
|---|---:|---:|---:|---:|
| 32 sprites | 128 | **128** | **0** | 256 |
| 64 sprites | 256 | **256** | **0** | 512 |
| 96 sprites | 2,048 | 474 | 1,574 | 4,096 |

In the first two shape-major sets there was **not a single exactly duplicated 16×16 RGBA tile** across the true creator-authored colorways. Color variations usually change RGB values, even where the UI element's shape/alpha is similar. At N=96, extensive exact reuse occurs, but padded transparent atlas space also causes many trivial identical tiles. Although indexed dictionary+WebP is compact, direct whole WebP uses the same repeated transparent image structures without extra 4,096 bytes of tile references and dictionary spatial rearrangement. Neither condition establishes a new compression advantage.

A good follow-up hypothesis is a *reversible alpha/color separation* or *predictive colorway delta* **on the same explicitly matched shape families**. These methods could exploit shared geometry despite different RGB values. They are established representation techniques, must be fully byte-counted and benchmarked against source-PNG+zstd and full lossless WebP; the source family is already explored, so any such positive outcome would need independent external artist packs for confirmation. Do not mask weak results by treating different colorways as exact pixel duplicates.

## Limitations and decision

- One CC0 pack (same as Exp78), no independently held-out artists and no actual authored chronological image revision history.
- Three nested subset sizes are correlated, **not three independent experiments**.
- No testing of alternative tile sizes or threshold tuning after viewing results. The *same seven* existing codecs were used unchanged.
- Source ZIP original and 21 compressed saved-file hashes/reconstructed RGBA per-sprite pixel hashes stored in official evidence artifact.
- No speed-optimized native application/runtime evaluation; Encode/decode time samples are available but are one runner only.
- This rejects a narrow exact-RGBA tile-dedup win for these true colorway variants, not all sprite asset storage domains.

**Release decision:** main branch, public demo and LinkedIn breakthrough publication stay **on HOLD**. Preserve negative evidence alongside Exp74's artificial-exact-copy win; a real positive requires an application-level advantage beyond ordinary WebP/PNG/Zstd.
