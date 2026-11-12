# BLKH Exp98 — preregistered retrospective joint-stream ablation

**Status at protocol freeze:** implementation and GitHub Actions workflow published; outcome **unknown**. No independent holdout and no release claim.

## Question

Exp95's independent per-pair Zstandard patch-size selector beat the Exp93 mismatch-minimizing motion selector on Wesnoth but lost to the best existing codec. Exp97 showed that local patch-size minimization can make the **aggregate archive larger**. Does optimizing the *whole jointly compressed patch stream* improve actual archive size?

## Fixed sources and candidates

Re-use the same 21 original Wesnoth Git PNG parent/child pairs from Exp87/93, with identical original source bytes, metadata and archive format. This is retrospective; it cannot be promoted as unseen evidence. For each pair, precompute at most three distinct records:
1. shift minimizing mismatched RGBA pixels;
2. shift minimizing individually Zstandard-compressed patch size (Exp95);
3. zero shift.

Initialize from the best *whole joint Zstandard stream* of three uniform selection strategies. Then run at most three deterministic coordinate-descent passes, in frozen pair order, replacing each pair's candidate only if the compressed **joint** patch stream strictly shrinks. Deterministic tie-break by candidate order. Keep exact old source WebP, metadata, header and all patch bytes in the complete archive.

## Audits and decision

For each of four overlapping groups, persist and separately decode complete archives from mismatch, local compressed-patch and global joint-stream selectors. Check source RGBA exactness including hidden RGB under alpha zero, metadata equality, byte counts and SHA-256. Compare full bytes to the frozen strongest prior archive: all_21_pairs 64,802 B; thea_shadow_fix 43,783 B; dune_firetrooper_cleanup 13,119 B; ant_sprite_distinction 9,916 B.

Report all results including negatives, no tuning on new projects, no independent claim. These four overlapping groups are **not** four independent replicates. Joint Zstandard scoring is CPU-expensive and needs time/RSS measurement before production suitability can be claimed. Any apparent win requires new preregistered independent source families and full same-fidelity reference comparisons. **Demo/LinkedIn HOLD** until that stage.
