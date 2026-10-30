# BLKH Experiment 81 — exact authored **historical PNG revisions** in Pixel-Boy NinjaAdventure

**Preregistration:** fixed before reading Git pixel contents or performing Exp81 compression. This test was proposed after Exp78–80 failures/mixed results on a single Kenney UI pack. This is a new creator/project source, but **ONE repository, not independently sampled artworks**.

## Source license and immutable Git inputs

Source game: `pixel-boy/NinjaAdventure`, https://github.com/pixel-boy/NinjaAdventure. Public Ninja Adventure art by Pixel-Boy and AAA is documented as CC0 at https://tmhsdigital.github.io/Free-Game-Dev-Assets/entry/ninja-adventure/ and original project licensing. **No assumptions that every unrelated repository file is CC0**: sample only PNG sprite-like assets in `content/character/`, `content/ui/`, `content/destroyable/`, `content/particle/`, `content/map/`; include source attribution and SHA256 of every PNG. Do not redistribute bulk source files in the BLKH Git repository. The CI clones the publicly available repo for verification only.

Frozen source `main` HEAD commit SHA **`6ac78232d5aedcc85ce5f27d060ea92366f7c24a`**. Traverse **exactly 11 consecutive parent→child transitions of its first 12 commits**, never later commits or a moving branch. Reproduce immutable Git object SHA by `git rev-parse HEAD` after checkout and each commit SHA of its first-parent path. There are no user-provided authored revisions here. On Git errors, **fail**, do not switch repositories.

## Deterministic genuine-change selection BEFORE compression

For each of the first 11 consecutive transitions newest→oldest, use `git diff --name-only --diff-filter=M <parent> <child> -- '*.png'` and restrict to the five approved `content/` sprite directories, explicitly excluding `.import`, atlas previews and source file additions/removals/renames. Read the *same path* in both Git commits using `git show sha:path`. Require PNG magic bytes, both images decode with Pillow, and both are true original PNG files; require both have **exactly the same width and height**, between 16 and 512, RGBA uint8 conversion, and **pixel RGBA bytes actually differ**. Record source path, parent/child commit SHA, original PNG SHA256 and original decoded RGBA SHA256 for both, source dimensions, and the number of changed channel values. No code-generated/simulated edits.

Sort candidates by **commit transition index (newest first)** then by original exact path lexicographically, preserving multiple genuinely distinct revisions to the same file if present. Select the **first 24 qualified transitions** (if fewer than 24, take all available and explicitly report the source-size limitation). No post-hoc cherry-picking. If fewer than 1 valid actual edited PNG pair exists, record inventory and abort, without substituting source or generating fake edits. The `source_inventory.json` must include every considered changed PNG and rejection reason. If enough candidates, preregistered nested samples are **4/8/16/24 consecutive pairs**, otherwise report all supported prefixes with no artificial duplicate pairs; pairs remain highly correlated within ONE project.

## Fixed exact RGBA compression experiment, paired revisions

Each pair supplies two actual historically distinct RGBA images, base and edited. For each nested prefix N of valid pairs, construct two deterministic RGBA arrays with identical pair geometry: original `before` and `after`. Use a shared source manifest and 16-pixel-aligned ordered shelf packing with max row width 1024; NO scaling, premultiplication or alpha replacement; all transparent hidden RGB values included.

Four frozen differential methods (one shared archive for all 2N versions):
1. `xor_webp`: unchanged source `before` image pixels plus bytewise XOR delta for `after` against `before`, stored in exactly the same ordered 2N-sprite atlas geometry, encoded with WebP lossless `exact=True`.
2. `mod_webp`: same with per-channel modulo-256 `after-before`, WebP lossless `exact=True`.
3. `xor_zstd`: same XOR transformed atlas as an RGBA byte stream, Zstandard level 9.
4. `mod_zstd`: same modular transformed atlas, Zstandard level 9.

**Mandatory strong original-asset references**: (i) exact original before+after PNG files packed as length-prefixed byte records and jointly Zstandard9 compressed, (ii) identical original 2N-source ordered-pixel WebP `exact=True` lossless atlas, (iii) optimized RGBA PNG atlas, (iv) original atlas RGBA byte buffer Zstandard9. All variants must have the **same full metadata manifest**, source SHA256, Git SHA and path/pair mapping and actually decode to exactly **all 2N original RGBA arrays**, including pixel channels under alpha zero. Preserve correct same-path source version pairing; ignore network/Git checkout bytes in *every* codec file equally but count all per-asset context in every archive. No decoder access to source Git.

For all formats, count complete serialized file size (headers+metadata+payload), verify saved SHA256, original and decoded RGBA hashes and individual source metadata. Evaluate per pair counts and aggregate byte overhead. Predeclared **exploratory success** requires the best of four differentials smaller than the strongest of four references on at least **three of four** nested prefixes **IF all four exist**; otherwise explicitly report insufficient samples and exploratory only. This is still one project with correlated revisions.

### Nonclaims

Sprite revisions are genuinely historical source edits, not synthesized. Results do **not** imply that exact content-addressing, conditional differencing or zstd/WebP are new methods; cannot extrapolate to all art families, videos, or GUI version-control stores. No main/demo merge and no LinkedIn breakthrough claims solely on Exp81.
