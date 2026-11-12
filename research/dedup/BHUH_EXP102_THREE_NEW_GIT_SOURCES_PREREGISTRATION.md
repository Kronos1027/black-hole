# BLKH Exp102 — pre-registered THREE NEW original Git PNG source repositories

**Source selection frozen BEFORE reading any source image pixels or running Exp102.**
**Status at freeze:** data not downloaded; no results. New independent source repositories distinct from the BLKH Pixelorama, Mindustry, Godot, Wesnoth, SuperTux, OpenGFX2, Endless Sky cases.

## Non-pixel source discovery and frozen manifests

All paths below are original tracked PNG files, status **modified** according to GitHub commit comparison `child^...child`. Use the single first parent of each child commit as the old revision. Do not substitute failed/missing/oversized images or change the source list after seeing their pixel values.

### Family A: Luanti / Minetest Game original texture edits

Repo: `luanti-org/minetest_game`.
Child commit: `960aff2f6366c2de864bdd9d4fd4f5eafa956bf3` ("Improve opaque leaves textures").
Selected: **all ten** PNG files modified by this commit, lexicographic Git path order:
```
mods/default/textures/default_acacia_leaves.png
mods/default/textures/default_acacia_leaves_simple.png
mods/default/textures/default_aspen_leaves.png
mods/default/textures/default_blueberry_bush_leaves.png
mods/default/textures/default_blueberry_overlay.png
mods/default/textures/default_jungleleaves.png
mods/default/textures/default_jungleleaves_simple.png
mods/default/textures/default_leaves.png
mods/default/textures/default_leaves_simple.png
mods/default/textures/default_pine_needles.png
```
Root `LICENSE.txt` includes media CC BY-SA 3.0 information and points to mod-specific author attribution; retain it and independently confirm asset rights before redistributing any imagery.

### Family B: Shattered Pixel Dungeon original sprite edits

Repo: `00-Evan/shattered-pixel-dungeon`.
Child commit: `14b1a7f2b7b656ffab922c349bc42aee39c04cb2` ("v4.0.0: various item sprite improvements!").
Selected: **both** modified PNG files:
```
core/src/main/assets/interfaces/icons.png
core/src/main/assets/sprites/items.png
```
Root `LICENSE.txt` contains GPLv3. Specific art authorship and terms still need confirmation before public distribution.

### Family C: Widelands original sprite sheet maintenance

Repo: `widelands/widelands`.
Child commit: `30d09f1ba2510478f73de7126d851ceb84f79260` ("Mipmap Spritesheets for all frisian units").
Selected: **first four lexicographically by path among the sixteen PNG files marked modified**, capped in advance to constrain atlas size and test runtime:
```
data/tribes/buildings/militarysites/frisians/outpost/idle_1.png
data/tribes/buildings/militarysites/frisians/outpost/idle_1_pc.png
data/tribes/buildings/militarysites/frisians/outpost/idle_2.png
data/tribes/buildings/militarysites/frisians/outpost/idle_2_pc.png
```
Root `COPYING` contains GPLv2; check asset-specific artist attribution before distribution.

## Frozen evaluation

Three genuinely new upstream repositories, **16 Git old/new image pairs**, not synthetic or cherry-picked by compression success. Run the six unchanged Exp100 lossless residual variants (two ±8 motion selectors × absolute/XOR/modulo-sub changed RGBA), plus **all fourteen Exp84 exact RGBA reference codecs** for each family: **60 fully persisted complete archives**. Same source manifest, full header and decoder for every complete archive. Include old-source WebP lossless bytes, metadata, patch bytes and codec headers in actual counts. Do not compare merely with PNG source bytes or a weak reference.

Verify independently with fresh upstream fetches, exact SHA-256 of original PNG file bytes, PNG original Git modification status, exact decoded RGBA including hidden RGB under alpha-zero, complete archive hashes, and successful decoder for all reference methods. Record time and peak RSS where available, group-level negative and positive results, and if geometry/size limits fail, mark **inconclusive** rather than replacing sources.

## Release decision frozen

Three **source repositories** are independent, but only one commit per source and 2 pairs in the Shattered family; do not equate this with statistical proof of general superiority. A promising result must beat the strongest same-fidelity reference by >=10% in each of three independent source families, decoding exact and with acceptable measured CPU/RSS; then perform larger holdouts and fair lossy rate-distortion versus JPEG/WebP/AVIF/JPEG2000 at matched error before claiming broad advantage. All predeclared failures and limitations must be published. Green CI alone is NOT a win. **Demo / LinkedIn HOLD** until audited cross-family results.
