# BLKH Exp93 post-audit decision and Exp94 preregistration

Status: **retrospective negative**, not a breakthrough; release/demo gate **HOLD**. Prepared after the completed independent source/archive audit in [Exp93 CI 41328753810](https://github.com/Kronos1027/black-hole/actions/runs/41328753810). Do not describe this as an independent *dataset* holdout: Wesnoth revisions were previously studied.

## Exp93 complete archive findings

All 60 persisted archives (4 groups x 15 methods) were decoded and source-checked in CI; the new motion codec reconstructs exact RGBA. The full archive byte counts, with all existing metadata and source data, were:

| Group | Motion codec B | Best prior sparse B | Best of 12 fixed references B | Motion penalty versus best of all prior B |
|---|---:|---:|---:|---:|
| all 21 pairs | 68,051 | 64,802 | 65,259 | +3,249 (+5.01%) |
| thea_shadow_fix | 47,012 | 43,783 | 43,964 | +3,229 (+7.37%) |
| dune_firetrooper_cleanup | 13,305 | 13,292 | 13,119 | +186 (+1.42%) |
| ant_sprite_distinction | 9,949 | 9,939 | 9,916 | +33 (+0.33%) |

The all-pairs row overlaps the three constituent groups and **must not** be counted as a fourth independent sample. The codec selected nonzero shifts in 7/21 pairs overall, all from the Thea group; the other two groups had zero shifted pairs. No group beat its strongest prior comparator. A winning shift-count or reduced changed-pixel fraction alone does not establish smaller compressed archive size: motion records, changed-mask entropy, residual values, old-source encoding and full headers matter.

## Decision: do not tune the Wesnoth data to rescue Exp93

A per-group codec selector could trivially choose the minimum already-known method and never exceed the best baseline *for each group*, but that is an oracle trained on evaluation groups. It adds a method ID/header and cannot substantiate a new compressor without a frozen rule and independent source families. Do not present oracle selection as an improvement.

## Exp94 preregistered, conditional on independently qualified data

**Hypothesis:** a fixed, causal motion/patch choice based only on a prior image and a new image, scored by *actual compressed candidate stream bytes* rather than changed-pixel count, can reduce complete archive size for genuinely translated authored sprite edits without sacrificing exact decoding. A result is only meaningful if it beats an equally optimized conventional complete archive, not just Exp93.

1. Freeze **before** loading new test images: source repository, immutable parent/commit pairs, modified PNG paths, upstream licenses, selection rule, exact group boundaries, and all codec parameters in a committed manifest. Require >=3 unrelated source repositories/artist families, no Wesnoth/Pixelorama/Mindustry/Godot/SuperTux sources used in earlier BLKH experiments. No generated, rotated or synthetically shifted images count as independent examples.
2. Training/calibration must be separate from test sources. Freeze integer translation window [-8,8], tie order, and deterministic candidate selection. A two-arm patch encoder may choose unshifted or shifted candidate **only by exact serialized compressed bytes**, with explicit per-pair method flag, source image, manifest, checksums and header counted. Compare to no-motion patch with identical container and backend, and to best fair conventional complete lossless codecs (including atlas, Zstandard, PNG source-byte mode where matching fidelity). Ensure identical original RGBA fidelity and provenance requirements across arms.
3. Decode each persisted file in a fresh process; verify every exact RGBA SHA256, complete source hashes and metadata. Measure file bytes, encoding/decoding wall time, peak RSS, source copyright policy and GitHub Actions reproducibility.
4. Predeclared success: >=10% complete-file byte savings against the **strongest same-fidelity baseline** in >=3 independent source families, without >2x median decode latency or >2x peak RSS; all data and scripts reproducible in CI. If any source cannot be fetched or licensing is ambiguous, label inconclusive; if valid but larger, label negative. Report all cases, including failures.
5. Keep lossy rate-distortion claims **out of scope** for this exact-RGBA study. For any future lossy claim, independently benchmark actual serialized post-decode PSNR/SSIM and size against AVIF/JPEG XL/WebP at matched quality; PNG lossless alone is not a fair reference.

**Current state:** Exp94 is a frozen proposal only. No independent source manifest, implementation, CI result or measured Exp94 advantage exists yet. The branch and PR remain research-only. No demo/LinkedIn breakthrough announcement.
