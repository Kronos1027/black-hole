# BLKH Experiment 86 — preregistered independent Godot-authored PNG revision transfer

**Frozen before accessing any of the selected original PNG bytes, decoded RGBA values or compression outputs.** This is a third, unrelated project after Pixelorama and Mindustry. The exact Git commits and paths below were selected from GitHub's commit **metadata only**, with no source-image inspection. Do not substitute paths, adjust the encoders or tune results after inspection.

## Source and rights
- Repository: `godotengine/godot-demo-projects`, https://github.com/godotengine/godot-demo-projects
- Source license: project-root `LICENSE.md` (MIT, verified as text at the art-revision commit). Source-specific ownership/exceptions must not be assumed; retain repo, path, parent, commit, source-file SHA256, license and original contributor attribution in the artifact.
- Group A: `platformer_art_redraw`, Git commit `545d63cd2dbdb87222bfde5371c947483df811e7`, first parent from Git API; **13** `status=modified` PNGs:
  - `2d/platformer/assets/art/background/cloud_1.png`
  - `2d/platformer/assets/art/background/cloud_2.png`
  - `2d/platformer/assets/art/background/cloud_3.png`
  - `2d/platformer/assets/art/background/distant_platforms_1.png`
  - `2d/platformer/assets/art/background/distant_platforms_2.png`
  - `2d/platformer/assets/art/background/sky.png`
  - `2d/platformer/assets/art/coin/coin.png`
  - `2d/platformer/assets/art/enemy/enemy.png`
  - `2d/platformer/assets/art/platforms/moving_platform.png`
  - `2d/platformer/assets/art/platforms/one_way_platform.png`
  - `2d/platformer/assets/art/player/bullet/bullet.png`
  - `2d/platformer/assets/art/player/robot_demo.png`
  - `2d/platformer/assets/art/tileset/tiles_demo.png`
- Group B: `godotea_sprite_revision`, Git commit `765ca64680da6a251241cc66bb1c0ea5d7225a40`, first parent from Git API; **1** `status=modified` PNG:
  - `2d/sprite_shaders/godotea.png`

These are two commits **in one upstream project**, NOT two independent authoring repositories. Select only the 13 platformer paths from one demo subtree to avoid counting synchronized duplicated assets in the parallel physics-platformer tree. One single-image group is *not* a statistically reliable independent replicate. Both commits were found from messages describing actual art changes, not from measured compression performance.

## Fixed evaluation design
Reuse the same **14 complete, lossless, byte-counted formats** from Exp84/85 **without changing their code, method settings, or reference set**: 2 original sparse-mask formats vs the best of the same 12 WebP/PNG/Zstandard/XOR/modulo/tile16/bsdiff4 reference methods. Use existing `experiment_83_pixelorama_git_history.py` and `experiment_84_sparse_pixel_patches.py` functions. Count exact serialized archive bytes, including all required original source metadata and old images, at all times.

Evaluate (1) all 14 actual historical file pairs together, (2) platformer art 13 pairs and (3) the singleton sprite, for **42 persisted archives**. Verify all files independently through the actual saved bytes, exact original decoded RGBA including hidden RGB at alpha=0, hash checks and full PNG byte hashes for the bsdiff4 method. The source revision must show original `status=modified`, with both old/new PNGs available at fixed hashes; source failures are explicit failures, never silently dropped. Record total percent of changed pixels as an explanatory variable, never to select favorable examples.

## Preregistered decision gate
Mark transfer as favorable **only if** the aggregate 14-pair sparse archive is strictly smaller than the best of all 12 references **and both fixed groups win**. Also separately require at least 2 genuinely RGBA-changed source pairs in the complete dataset; if not, call the test *inconclusive due to source qualification*, not a successful transfer. Report all negative results, including difference in bytes, no retuning. A win in one group alone does not establish generalization. Because these art redraws are potentially different from the <1% pixel edits of Mindustry's texture_cleanup subset, this is a **domain-shift stress test**, not a preselected microedit success case.

The official CI result artifact must include original source PNGs, original root license, provenance, all 42 archives, JSON manifest and independent post-decode audit. Do not merge research PR #1, modify `main`, demo, or publicly market a breakthrough based on this experiment.
