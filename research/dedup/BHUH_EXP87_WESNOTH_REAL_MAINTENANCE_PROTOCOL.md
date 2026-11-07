# BLKH Experiment 87 — PRE-REGISTERED Wesnoth real artist sprite maintenance, independent project holdout

**Frozen before accessing any selected PNG bytes, decoded RGBA pixel change ratios or compressed archive sizes.** An unrelated upstream repository (new relative to Pixelorama, Mindustry and Godot), and exactly three manually selected Git commits from Git **message and filename metadata only**. Do not alter, trim, or substitute source paths after source reading. This is a narrow prospective test of authentic maintenance edits, not a population-level artist experiment.

## Upstream and exact immutable source paths

- Git project: `wesnoth/wesnoth`, https://github.com/wesnoth/wesnoth
- Repo-root `COPYING` carries GPLv2 license text, but **specific art can have distinct contributor rights**. Record project-root license, `copyrights.csv`, source-path history, author Git provenance, raw PNG SHA256 and decoded pixel SHA256. Do not publish original artwork into BLKH repository; preserve it with attribution only in time-limited CI benchmark artifacts.
- Each selected commit has exactly one Git parent, checked again in the source audit.
- **Source A — `thea_shadow_fix` (14 original status=modified PNG paths):** commit `ee27cba746684bd72fb0499226db381bf85db5bd`. Message: “PaP: fix shadow on Thea's L2 sprites”. Base directory `data/campaigns/Of_Pearls_and_Pirates/images/units/thea2/`. Exactly: `longbowman-bow-attack-1.png`, `longbowman-bow-attack-2.png`, `longbowman-bow-attack-3.png`, `longbowman-bow-attack-4.png`, `longbowman-bow-defend.png`, `longbowman-bow.png`, `longbowman-melee-attack-1.png`, `longbowman-melee-attack-2.png`, `longbowman-melee-attack-3.png`, `longbowman-melee-attack-4.png`, `longbowman-melee-defend-1.png`, `longbowman-melee-defend-2.png`, `longbowman.png`. Also `data/campaigns/Of_Pearls_and_Pirates/images/units/thea3/masterbowman.png`.
- **Source B — `dune_firetrooper_cleanup` (4 original status=modified PNG paths):** commit `5d0c0ec39384c364d716fa5039d53913da4b1e20`. Message: “1.19 Dune firetrooper sprites — cleanup”. Base directory `data/core/images/units/dunefolk/burner/`. Exactly: `firetrooper+female.png`, `firetrooper.png`, `scorcher+female.png`, `scorcher.png`.
- **Source C — `ant_sprite_distinction` (3 original status=modified PNG paths):** commit `811a2c5b05df2d7e48c57f49c4f76c1c3f3923fc`. Message: “make soldier ant sprite a little more distinct”. Base directory `data/core/images/units/monsters/ant/`. Exactly: `soldier-attack.png`, `soldier-defend.png`, `soldier.png`.

Total frozen: **21 original author-maintained PNG pairs (42 historical PNG files), 3 commit groups, all-original bundle plus three fixed groups**. Each group is related artist work in the **same project**, not an independent organization. No based-on-pixel-similarity filtering or cherry-picking.

## Unchanged fixed archive evaluation

Reuse **exactly** Exp84–86's two 1-bit sparse RGBA mask codecs and all **twelve existing full lossless references**: WebP/PNG/Zstandard atlas, old original PNG, per-file optimized PNG, XOR/modulo RGBA deltas, 16px tile dictionaries, real PNG-original bsdiff4 binary deltas. Preserve their existing parameters (Zstandard level 9, WebP lossless, fixed tile size, global atlas layout), complete-source metadata under the same shared manifest convention, and encode all original old images into each actual archive (no free decoder side information). Require real source `status=modified` against the frozen Git parent in the independent audit; require original PNG bytes and independently decoded exact RGBA including invisible RGB. Every archive must round-trip exactly and its SHA256, full file bytes, codec name, encoding and decoding times must be counted.

Evaluate **56 saved compressed files (4 cases × 14 codecs)** with a second independent read and byte-for-byte verification and source original PNG file hash verification; the bsdiff arm additionally verifies exact source PNG files. All original sources and reference outputs in temporary GitHub Actions artifact with `COPYING` and attribution; not in BLKH Git tree. Strictly fail source qualification if any fixed file is missing/oversized/non-PNG or 14-codec source memory bounds fail.

## Frozen scientific decision

For a prospective **whole-corpus transfer success** require **all**:
1. Sparse beats the smallest of **all twelve reference archives on the exact complete 21-pair archive**;
2. Sparse beats the smallest reference archive in **at least 2 of 3 original commit groups**;
3. All 21 source pairs were Git-`modified` and at least **two** show actual decoded RGBA changes;
4. 56 archives pass the persisted-byte and original-content audit.

A source-size or decoding-limit failure means **source-unqualified/inconclusive**, never a win. If the aggregate loses, immediately report a negative outcome even if one subgroup wins. Small-change fractions measured *after* source selection and shared metadata redundancy are limitations. No retuning or new source cherry-picking after viewing results. One new upstream repository is not enough to establish universal state-of-the-art; keep `main`, public demo, marketing unchanged, and draft PR unmerged.
