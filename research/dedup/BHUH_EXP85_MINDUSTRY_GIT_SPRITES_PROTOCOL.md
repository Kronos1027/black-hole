# BLKH Experiment 85 — independent-project real author-edited sprite PNG Git history

**PRE-REGISTRATION.** This protocol and the exact source paths below are recorded **before any Exp85 original sprite pixels, file sizes or codec outcomes are inspected**. Files were chosen solely from fixed Git change metadata (original PNG path status=modified in a real parent→child commit), not from compression scores. No changes or replacement paths are allowed after source inspection.

## New source repository and licensing

**Anuken/Mindustry**: https://github.com/Anuken/Mindustry, independent of Pixelorama Exp83P–84. Root license is GNU GPLv3 (https://github.com/Anuken/Mindustry/blob/master/LICENSE); **individual artwork provenance or exceptions must not be assumed**. Retain links, original Git commit and parent SHAs, repo-level LICENSE text, and source PNG SHA256 for every original asset. Distribute source PNGs *only* in a retained short-lived GitHub CI result artifact alongside a copy of the GPLv3 license and attribution, not copied into our BLKH source repository. Verify all 30 selected paths still have Git status `modified` in their pinned commit diff and that both originals can be opened as PNG. If any expected path is unavailable, oversized, or not eligible, the experiment must fail visibly; no substitute paths.

### EXACT pinned source selection — 30 same-path historical PNG pairs

- **defense** bae93299c7721a35431daa33e3af8b1f6c596c8d: 2 original Git-status=modified PNG files:
  - `core/assets-raw/sprites/blocks/defense/large-shield-projector-team.png`
  - `core/assets-raw/sprites/blocks/defense/large-shield-projector.png`

- **texture_cleanup** 916a00b6ff8f1bc16427ef430df9223b3855d908: 15 original Git-status=modified PNG files:
  - `core/assets-raw/sprites/blocks/campaign/launchpod.png`
  - `core/assets-raw/sprites/blocks/distribution/phase-conveyor.png`
  - `core/assets-raw/sprites/blocks/drills/blast-drill-rotator.png`
  - `core/assets-raw/sprites/blocks/drills/oil-extractor-rotator.png`
  - `core/assets-raw/sprites/blocks/drills/water-extractor-rotator.png`
  - `core/assets-raw/sprites/blocks/environment/pine.png`
  - `core/assets-raw/sprites/blocks/environment/snow-pine.png`
  - `core/assets-raw/sprites/blocks/environment/spore-pine.png`
  - `core/assets-raw/sprites/blocks/liquid/phase-conduit.png`
  - `core/assets-raw/sprites/blocks/power/power-source.png`
  - `core/assets-raw/sprites/blocks/power/power-void.png`
  - `core/assets-raw/sprites/blocks/production/pulverizer.png`
  - `core/assets-raw/sprites/blocks/storage/vault.png`
  - `core/assets-raw/sprites/blocks/turrets/swarmer.png`
  - `core/assets-raw/sprites/units/eclipse.png`

- **walls** 0aaa77a5eae7c3e3474bc869d6b622253f729575: 13 original Git-status=modified PNG files:
  - `core/assets-raw/sprites/blocks/walls/scrap-wall-gigantic.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall-huge1.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall-huge2.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall-huge3.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall-large1.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall-large2.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall-large3.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall-large4.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall1.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall2.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall3.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall4.png`
  - `core/assets-raw/sprites/blocks/walls/scrap-wall5.png`

**Natural grouping:** Three commits from one independent project = **30 paired historical assets, 60 source PNG versions**. These are correlated source files/commits, not 30 independent repositories. Each old PNG is from the exact first parent commit; each new PNG from the selected child SHA. `status=modified` does not necessarily imply changed decoded RGBA; record separately the count of truly changed original RGBA source pixels in all three groups, and include *all selected files* even if the original PNG encoding alone changed. If any asset exceeds the unchanged Exp83 decoder limits, mark source failure instead of opportunistically selecting smaller assets.

## Frozen codec methodology

**Do not change any of the 14 encoding algorithms or parameter settings from Exp84:** two sparse 1-bit mask / full changed RGBA pixel methods (old WebP `exact=True` atlas or old exact source PNG bytes, plus Zstd9 mask/pixel stream), eleven original Exp83P fixed codecs (four XOR/modulo WebP/Zstd atlas differentials; five PNG/WebP/Zstd whole-atlas or original-PNG baselines; two exact 16px SHA256 indexed block dictionaries), and **bsdiff4** over original old/new PNG file bytes + Zstd9.

Use their existing unmodified `experiment_83_pixelorama_git_history.py` and `experiment_84_sparse_pixel_patches.py` codec functions. Only the **new data loader** and provenance, fixed path list, group labels and rights-attribution differ. Preserve 8-bit RGBA **including hidden RGB under zero alpha**; if source is paletted, decode to exact RGBA and record source PNG bytes. Same shelf width 1024, tile padding 16, Zstd level 9, and identical full manifest across all codecs. Original old and new dimensions may differ. Every saved file must independently recover both old/new source pixels exactly; **bsdiff4** must recover the *original PNG file bytes* as well.

Measure full archive bytes with complete source metadata, source hash, old atlas layouts, and all dictionary/index bytes. Compare the minimum of the two fixed sparse candidates against the minimum of **twelve references** for the **full set** and each of three natural commit groups — **56 saved archives**, 14 formats × 4 groups. Record per-case changed-pixel fraction and actual RGBA-changed-pair count. Note comparative differences with Exp84 may reflect *different source data and metadata*; only within-run comparisons support wins.

## Frozen decision rule

The **independent-repository transfer criterion** is **a full 30-pair sparse-patch bundle win AND at least 2/3 per-commit sparse wins** against all 12 references. If the new source has too few genuinely changed RGBA files or some source is invalid, report clearly and do not claim a full 30-pair win. The source has strong sprite repetition but only one new independent repository; do NOT claim a universal image compressor even if some results are positive.

No source-dependent changes of mode, thresholds, filtering or tile sizes after inspecting compression output. No production/demo/`main` change or broad LinkedIn breakthrough claim from this pilot alone.