# BLKH Exp80 — same-shape authored colorway deltas (MIXED, 1/3)

**Status: successful GitHub Actions CI and independent per-file exact RGBA decoding.** One narrowly positive complete-byte saving of **132 bytes (0.224%)** at 96 related sprites, but **2/3 nested subsets failed** to beat the best existing complete-file baseline; therefore the preregistered 3/3 success criterion was NOT met. This is a **post-Exp79 follow-up on the SAME ONE Kenney UI Pack**, not an independent held-out dataset or evidence of new compression theory.

**Precommitted protocol:** [BHUH_EXP80_RGBA_COLORWAY_DELTA_PROTOCOL.md](BHUH_EXP80_RGBA_COLORWAY_DELTA_PROTOCOL.md).  
**Successful official run:** https://github.com/Kronos1027/black-hole/actions/runs/40713759209  
**Code:** `research/dedup/experiment_80_rgba_colorway_deltas.py` and `.github/workflows/exp80.yml`.  
**Evidence:** GitHub Actions `exp80-authored-rgba-differential-vs-all-references` contains 33 actual persisted archives (11 codec variants across each of 32/64/96 images), source atlas PNG files, per-file SHA256, exact source metadata and machine-readable results.json.

## True source and scope

Same unmodified Kenney CC0 `https://opengameart.org/sites/default/files/kenney_ui-pack.zip`, source SHA256 `a8a14a234911eb648c062622915c93e79e94e97cb7f9f375a70f6617f1174318`, used in Exp78–79. Preselected 8, 16 or 24 lexicographically first eligible original UI shape names, each with **four distinct artist-authored colorway PNGs: Blue, Green, Red, Yellow**, total 32,64,96. Within each shape the original 4 color variants have identical dimensions. The groupings are nested and strongly correlated; no authored chronology, historical revisions or independent new artist/source pack.

**All source PNG pixels interpreted as exact straight 8-bit RGBA, including RGB values under fully transparent pixels, and separately source PNG hashes.** Same ordered-shelf packing with fixed 1024 width maximum, same source manifest and total metadata as all previous Exp79 comparisons. No artificial recoloring, source shifting or alpha normalization.

## Four fixed new fully reversible methods

For each shape, the Blue RGBA sprite is the unmodified per-group anchor. Three other original creator colorway RGBA arrays are transformed pixel by pixel relative to Blue in all four channels using **unsigned XOR** or **modulo-256 subtraction**, then placed at exactly the same coordinates in the original atlas geometry. The residual atlas is compressed as either exact RGBA **WebP lossless** (`exact=True`) or **ZSTD level 9**. All original source PNG RGBA file hashes, shape positions/dimensions, colorway order, and headers are accounted for. The standalone NumPy decoder reads only saved bytes, recovers Blue and all other colors, and verifies each original sprite RGBA SHA256 and full reconstructed atlas hash.

There were 4 new transform+entropy combinations and **all 7 unchanged Exp79 reference formats**: indexed 16px WebP/ZSTD tile dictionaries, whole RGBA PNG/WebP atlases, whole raw RGBA+ZSTD atlas, length-prefixed original source PNG files+ZSTD, and optimized per-sprite PNG files+ZSTD. The best reference is the actual minimum complete file across **all seven**, not just a favorable weak PNG baseline.

## Official complete-file, metadata-counted result

| Original Kenney sprite count | Best reversible colorway residual | Encoded candidate bytes | Best of 7 original references | Reference bytes | Delta vs reference |
|---|---|---:|---|---:|---:|
| 32 (8 shapes ×4) | mod_webp | 19,415 | **original_png_zstd** | **19,346** | **+69 bytes** / +0.357% |
| 64 (16 shapes ×4) | mod_webp | 39,952 | **original_png_zstd** | **39,322** | **+630 bytes** / +1.602% |
| 96 (24 shapes ×4) | **xor_webp** | **58,707** | atlas_webp | 58,839 | **−132 bytes** / −0.224% |

Every one of **33 complete encoded files** independently decoded exactly to each original artist PNG's RGBA byte values, including alpha and hidden transparent RGB; original source files were also independently SHA256-verified. No accuracy or quality trade-off.

**Success rule:** 3/3 nested subsets strictly smaller than best unchanged reference. **Actual: 1/3**, **criterion FAILED**. The 132-byte 96-sprite improvement is legitimate in this measured context, but is a negligible effect that may change with small metadata/packing changes or different unrelated samples. It should not be described as a general neural/shared compression breakthrough or as independently replicated validation.

## Next decision

This follow-up changes the data representation in an intuitive and reversible way. Its limited 1/3 result does not justify repeated parameter sweeps on the same known 24 shapes, additional favorable source ordering, or post hoc thresholds. Instead, future evaluation should test source-controlled **actual version histories of independently created sprite assets** and strong alternative version-control/dedup archives, with all artifacts and decoded pixels preserved. This one pack has already been reused across Exp78–80 and is unsuitable as an unbiased held-out sample.

**Release:** keep main, demo and public LinkedIn claims unchanged, research PR #1 draft. Do not merge or announce breakthrough based on a 132-byte within-corpus result.
