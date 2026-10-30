# BLKH Exp88 — independent SuperTux original PNG Git revisions: NEGATIVE, source and all 56 archives verified

**Official scientific outcome: FAILURE of the preregistered transfer criterion.** Successful GitHub Actions run [40935835289](https://github.com/Kronos1027/black-hole/actions/runs/40935835289), with independent source Git status and file-content audit and independent full persisted-archive decode audit. All 56 actual byte-counted archives reproduce the original 8-bit RGBA source pixels *exactly*, including RGB stored under fully transparent pixels; the `bsdiff4` reference also reconstructs the original **PNG file bytes**. The run saved an 18 MB source-data/archive/results artifact named `exp88-supertux-real-git-png-history-56` (artifact ID `12924911444`); no source artwork is added to the BLKH Git repository.

## Frozen selection and provenance

The protocol [BHUH_EXP88_SUPERTUX_PREREGISTERED_PROTOCOL.md](BHUH_EXP88_SUPERTUX_PREREGISTERED_PROTOCOL.md) was committed **before the selected source PNG bytes were read**, as commit `b99312c2074ea9e5e5c187d2a9b71c0e89a18b69`. The independent source project is `supertux/supertux` — distinct from prior Pixelorama, Mindustry, Godot and Wesnoth. Its original root `LICENSE.txt` indicates GNU GPLv3, with possible separate attribution for individual artwork; the CI artifact retains original source Git paths/hashes, first-parent SHA, root license text and attribution notice. Reuse of the content must respect actual contributor rights.

Exactly **29 authentic Git status=modified PNG pairs** (58 historical files), frozen by three original Git commit messages/path families:
- `rock_fallblock_art`: 14 PNG pairs from commit `624d2b951320923eaae9a1b1e403d90c7218427e`, **14 of 14 genuinely changed decoded RGBA**.
- `autotile_relayout`: 7 PNG pairs from `2cdc40783befc7535a3d787a283876e5476d155b`, **7 of 7 genuinely changed decoded RGBA**.
- `penny_png_profile_cleanup`: 8 PNG pairs from `0362b47a9b39a44d1efe7ef05c9f5ac8aa6bd256`, **0 of 8 genuinely changed decoded RGBA**, despite Git's original PNG files being changed. This is the deliberate predeclared image-metadata/re-encoding negative-control group, not excluded after source audit.

The source-qualification gate passed: **21 of 29** source pairs genuinely changed RGBA, well above the ≥2 minimum. Source Git status `modified`, original per-file PNG bytes and pixel SHA256 were separately checked by CI.

## Exact lossless comparisons, full file bytes including shared metadata

All 14 unchanged Exp84–87 methods were run: 2 experimental complete sparse-mask/4-channel-replacement codecs and **12 established complete lossless references**, including original PNG bsdiff4, XOR/modulo WebP/Zstandard, atlas WebP/PNG/Zstandard and 16-pixel tile dictionary formats. Every method encoded original pre-edit pixels and complete metadata within its archive, no free side-information, and every actual saved file was independently reopened, SHA256 checked and decoded.

| Fixed source group | Changed new-image pixel fraction | Best sparse mode | Sparse bytes | Best of all 12 references | Reference bytes | Sparse difference |
|---|---:|---|---:|---|---:|---:|
| **Entire 29 original PNG edit bundle** | **36.7740%** | `sparse_webpold_zstd` | **430,620** | `tile16_zstd` | **374,887** | **+55,733 B (+14.8666%): LOSS** |
| `rock_fallblock_art`, 14 pairs | 80.8435% | `sparse_webpold_zstd` | 173,652 | `tile16_webp` | **144,883** | **+28,769 B (+19.8567%): LOSS** |
| `autotile_relayout`, 7 pairs | 3.3781% | `sparse_webpold_zstd` | 229,132 | `tile16_zstd` | **201,662** | **+27,470 B (+13.6218%): LOSS** |
| `penny_png_profile_cleanup`, 8 pairs | **0.0000%** | `sparse_webpold_zstd` | 29,088 | `mod_webp` | **27,548** | **+1,540 B (+5.5902%): LOSS** |

**Preregistered positive criterion:** full 29-pair bundle win against minimum of 12 references **AND** at least 2/3 source-commit subgroup wins, all original source and 56 persisted archives fully verified. Observed: **0/3 subgroup wins, no aggregate win**. All sources qualified and all file audits passed, so this is a **reproducible negative result**, not an inconclusive run.

## Interpretation and limitations

1. The best competitor depended on the source: exact **tile dictionaries** dominated the bundle and two of the three groups; simple modulo WebP excelled on the no-pixel-change subgroup. The sparse implementation cannot be claimed as best for general sprite revisions.
2. Importantly, **Git status=modified does not imply RGBA pixels changed**. The eight `penny_png_profile_cleanup` PNG original file revisions modified PNG encoding/metadata but decoded to byte-identical RGBA. The experimental RGBA-only codecs correctly meet their declared pixel-reconstruction goal, but **do not promise exact original PNG ancillary metadata**. The bsdiff4 baseline has a stricter source-byte reconstruction target, appropriately disclosed.
3. Changed-pixel fraction alone is insufficient to predict wins: the 0%-changed source group still lost to the best reference, and 3.38%-changed autotiles still lost badly to tile dictionary methods. Artist edit structure, repeated tiles, image geometry and header sizes matter. No new threshold or scaling law is validated.
4. These pairs are historically related files from one source repository with three commits, not 29 independent artist teams. Fixed sampling by filename family may not be representative. All actual archive byte totals include a shared historical source manifest for comparability; some fields are redundant for some codecs. The experiment is an equal-metadata **within-run** comparison, not fully optimized per-codec product packaging or a native Git packfile comparison.

## Accumulated evidence and release decision

- Exp84 Pixelorama: authentic source revisions, aggregate LOSS.
- Exp85 Mindustry: aggregate LOSS, one small subgroup win.
- Exp86 Godot: aggregate LOSS.
- Exp87 Wesnoth: **aggregate small WIN by 457 bytes (0.7003%)**, but preregistered subgroup transfer gate FAILED (1/3).
- **Exp88 SuperTux: aggregate LOSS by 55,733 bytes, all three groups lose**, strong exact reproducibility.

This is now **five independent source projects**, with **one small whole-corpus win** and **no successful preregistered transfer gate** in Exp84–88. It does not establish broad or universal improvement. A scientific next step is a *separate* audited study of **minimal per-codec metadata and native Git / bsdiff baselines**, plus another truly independent frozen project if the goal remains sparse-edit replication. Do not retune Exp88 or remove the metadata-only group post hoc.

**Public release HOLD**: PR #1 remains research-only/draft, `main` and public demo unchanged; no state-of-the-art/compression-breakthrough promotional claim.
