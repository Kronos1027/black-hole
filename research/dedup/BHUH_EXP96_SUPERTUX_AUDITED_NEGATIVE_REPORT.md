# BLKH Exp96 — audited SuperTux cross-project transfer (negative)

**Scientific decision: HOLD.** Cross-project retrospective source (previously used in Exp88), no blinded independence and no gain against strongest reference.

**Implementation:** [a2a3af5](https://github.com/Kronos1027/black-hole/commit/a2a3af547809fcf929dc342ff0e5265ebb8961ae)
**Workflow:** [79480a2](https://github.com/Kronos1027/black-hole/commit/79480a21a4987df0af2883205b8b7c3ed90c63f0)
**Verified run:** [43282238681](https://github.com/Kronos1027/black-hole/actions/runs/43282238681) (success).
**Audited archive artifact:** [13972358869](https://github.com/Kronos1027/black-hole/actions/runs/43282238681/artifacts/13972358869).

29 genuine Git parent/child PNG pairs in `supertux/supertux`, frozen in Exp88; 4 overlapping groups, 14 reference codecs, two motion selectors, **64 full self-contained archives independently decoded and exact RGBA verified**. Source original PNG bytes are not the lossless guarantee: the guarantee is exact RGBA pixels, including hidden RGB.

| Group | Mismatch-min motion | Compressed-byte motion | Best full archive among 14 baselines | Motion penalty vs best |
|---|---:|---:|---:|---:|
| all_29_pairs | 430,782 B | 430,482 B | 374,887 B | +55,595 B (+14.83%) |
| rock_fallblock_art | 173,779 B | 173,480 B | 144,883 B | +28,597 B (+19.74%) |
| autotile_relayout | 229,151 B | 229,151 B | 201,662 B | +27,489 B (+13.63%) |
| penny_png_profile_cleanup | 29,097 B | 29,097 B | 27,548 B | +1,549 B (+5.62%) |

All four groups **lose** to the strongest same-fidelity competitor. The compressed-byte selector gains only 300 bytes in aggregate relative to mismatch-min, and the subgroups overlap the aggregate. The PNG-profile cleanup is an important negative control because the Git changes may be metadata-only with unchanged RGBA.

Limitations: retrospective, previously seen source, no CPU/RSS overhead measured, no quality-matched lossy RD claim. No public compression breakthrough; no demo or LinkedIn superiority claim.

**Next phase:** Exp97 on the existing frozen Godot history; treat it as a second cross-project retrospective stress test, not as an unseen independent holdout. Do not reinterpret a CI pass alone as a positive result.
