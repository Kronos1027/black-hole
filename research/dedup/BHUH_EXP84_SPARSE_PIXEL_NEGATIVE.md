# BLKH Exp84 — changed-pixel masks vs binary patches on real Git PNG history (NEGATIVE)

**Status: complete and independently revalidated by official GitHub Actions.** **0/4** actual 19-pair and three per-commit comparisons beat the strongest real baseline when full source metadata and pixels are counted. **No hypothesis win**, although the new codecs faithfully reproduced every RGBA pixel, and the bsdiff4 reference reconstructed both old and new original source PNG byte streams exactly.

- Frozen before Exp84 byte measurements: [BHUH_EXP84_SPARSE_PIXEL_PROTOCOL.md](BHUH_EXP84_SPARSE_PIXEL_PROTOCOL.md).
- Reproducible source: `research/dedup/experiment_84_sparse_pixel_patches.py`; CI: `.github/workflows/exp84.yml`.
- [Official successful workflow](https://github.com/Kronos1027/black-hole/actions/runs/40829855325) and complete artifact `exp84-pixelorama-real-sparse-patches-vs-bsdiff`: all **56 actual saved files** (14 codec variants × 4 source groups), provenance and exact source hashes, decoded-file audits, per-file byte counts and checksums.
- The inputs are the **same preexamined 19 genuinely RGBA-modified Pixelorama PNG version pairs** pinned to three real historical Git commits (13 pan/cursor pairs, four curve/tool pairs, two isometric pairs), not new independent held-out images. Old/new source PNG byte hashes and original RGBA byte hashes are recorded, with hidden RGB under zero alpha preserved.

## Frozen four-group results

Both new variants stored original old RGBA values plus a bit-packed **one bit per changed pixel** mask and four replacement-channel bytes for each changed pixel. Candidates used a shared old-image lossless WebP atlas or the *original historical old PNG bytes* as source, then one Zstandard9 compressed patch stream. All candidate file metadata, source keys, original PNG hashes and indices were accounted for. Comparators were the eleven preexisting exp83p methods plus **bsdiff4(old PNG, new PNG)** patches, encoded into one Zstd9 archive, which restores the *original old and new PNG files byte-for-byte*.

| Real historical group | Pixels replaced in new-source domain | Best custom archive (bytes) | Best of twelve reference archives (bytes) | Excess custom size |
|---|---:|---:|---:|---:|
| Complete 19 image revision pairs | 5,141 / 15,676 = **32.795%** | 38,320 (`sparse_webpold_zstd`) | **38,276** (`tile16_webp`) | **+44 B** (+0.115%) |
| Pan/cursor, 13 pairs | 4,477 / 12,772 = **35.053%** | 26,455 | **26,278** (`tile16_webp`) | **+177 B** (+0.674%) |
| Curve and 3D, 4 pairs | 250 / 1,936 = **12.913%** | 9,028 | **8,904** (`atlas_webp`) | **+124 B** (+1.393%) |
| Isometric, 2 pairs | 414 / 968 = **42.769%** | 5,562 | **5,377** (`atlas_webp`) | **+185 B** (+3.441%) |

**Prespecified success criterion** was aggregate win + at least two of the three individual commit groups; result **0/4, failed**. Full-file values are not directly comparable to Exp83P because Exp84 adds the original old-atlas layout metadata to *every* format. Only the matched within-Exp84 rows support the stated differences.

**Quality verification:** saved files were re-opened and all 38 image versions reconstructed exactly to their original uint8 RGBA pixels including transparent hidden RGB, source geometry and ordering. The bsdiff reference also independently reconstructed the complete actual source PNG bytes. The latter is a materially stronger file-preservation property than an exact pixel codec. Python implementation runtime and decoder software size were not included in archive sizes for any arm.

## Interpretation and next decision

These 19 sprite edits affect roughly 13–43% of new image pixel positions. A per-pixel replacement mask + 4-byte changed pixels is not especially sparse on these sources, and can lose to a single WebP atlas or an explicit tile dictionary despite exact old/new reconstruction. A successful decompression proof does **not** imply a compression advantage.

The high source correlation, already inspected set, tiny absolute differences and four within-source comparisons do **not** justify a broad scientific theory claim. Further parameter tuning on the same 19 pairs would be post-hoc. The better next experiment is **independent authored PNG version histories from a different real project** with new commits and file paths frozen before source-pixel reading, using these **same fixed codecs**, plus a legitimate binary-delta reference and exact source provenance. That experiment should disclose source licenses and reject unverified "modified" files whose decoded pixels do not change rather than silently select other versions.

**Release gate:** PR remains draft. `main`, public demo and unqualified LinkedIn breakthrough claims unchanged.
