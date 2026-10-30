# BLKH Exp91 — audited original PNG source-byte preservation: Zstandard list beats native Git PACK and bsdiff on both frozen populations

**Official independently verified GitHub Actions run:** [40999678657](https://github.com/Kronos1027/black-hole/actions/runs/40999678657), SUCCESS. Artifact `exp91-original-png-byte-exact-git-pack-zstd-bsdiff-24` ID `12953026821`, 9,542,181 byte ZIP, 30-day Actions retention. All **24 actual complete original PNG source-byte-exact archives** were written, reopened, SHA256 checked, and decoded; **100 genuine original immutable Git PNG source files** independently refetched and compared byte-for-byte and exact uint8 RGBA. The Git PACK method was decoded into a newly initialized, isolated SHA1 Git repository with **no access to the encoding repository**, verifying native Git `index-pack --stdin` and `cat-file blob` functionality. The method restored all original PNG ancillary metadata because it restores **original source file bytes**, not a newly encoded RGBA image.

Frozen before new pack-size comparisons: [Exp91 native Git PNG original-file-byte protocol](BHUH_EXP91_NATIVE_GIT_PNG_BYTES_PROTOCOL.md), commit `1e9e2094e7a52a1e5d8d25c825d023b9f6613b48`. Previously known Exp87 Wesnoth and Exp88 SuperTux source revisions reused *without changing source paths, input selection or library method parameters*; this is a **retrospective file-byte baseline study**, NOT a new independent-source holdout.

## Explicit equal-fidelity and equal-manifest design

**Required fidelity: exact entire original PNG file bytes**, including original PNG metadata, embedded ICC profiles, image source encoding and hidden RGB, not merely decoded RGBA. All three independently recover original old and new Git PNG byte streams, no external old source. Each complete saved format is one `BL91P` fixed header + identical source-order manifest (per original PNG **20B native Git SHA1 blob OID + 32B SHA256 of source file bytes**) + exact complete compressed method payload. No original path/commit/license fields are embedded; this **weaker source provenance** choice is uniform among these three methods. The original upstream attribution remains in the source research inventory and CI artifact; original art must not be redistributed without appropriate contributor rights. Source image content is full source bytes and exact original PNG chunks, unlike the Exp90 RGBA-only arms.

Registered three methods under identical complete archive byte-count contract:
1. `pnglist_zstd9`: Zstandard level 9 over a size-prefixed sequence of every old and new original PNG.
2. `pngpair_bsdiff_zstd9`: per original Git edit pair original old PNG + `bsdiff4.diff(old,new)` (source byte patches), then Zstandard9, exact original new PNG reconstructed using bsdiff4.patch.
3. `git_pack_level9_window50_depth50`: original PNG bytes written as genuine native Git SHA1 blob objects; native Git `git pack-objects --stdout --window=50 --depth=50 --compression=9 --delta-base-offset`; **PACK** bytes saved inside archive; independent temporary-repository decode using `index-pack` and `cat-file`. Neither external pre-populated Git objects nor the Git .idx file are necessary at decode time; .idx is regenerated from packfile. Git version saved to CI `results.json`.

## Independently audited complete archive byte counts

| Authentic frozen Git history | Original full PNG bytes | `pnglist_zstd9` | `git_pack_level9_window50_depth50` | `pngpair_bsdiff_zstd9` | Winner |
|---|---:|---:|---:|---:|---|
| **Exp87 Wesnoth 21 edit pairs (42 files)** | **187,335** | **70,694** | 78,315 | 87,956 | `pnglist_zstd9` |
| Wesnoth Thea shadow 14 pairs | 166,880 | **56,302** | 61,420 | 70,881 | `pnglist_zstd9` |
| Wesnoth Dune 4 pairs | 8,832 | **3,887** | 4,930 | 4,459 | `pnglist_zstd9` |
| Wesnoth Ant 3 pairs | 11,623 | **10,713** | 12,055 | 12,876 | `pnglist_zstd9` |
| **Exp88 SuperTux 29 edit pairs (58 files)** | **1,374,919** | **1,212,441** | 1,311,965 | 1,271,973 | `pnglist_zstd9` |
| SuperTux Rock/fallblock art 14 pairs | 495,866 | **450,567** | 485,350 | 464,499 | `pnglist_zstd9` |
| SuperTux Autotile relayout 7 pairs | 600,177 | **574,801** | 581,305 | 579,412 | `pnglist_zstd9` |
| SuperTux Penny source-PNG metadata cleanup 8 pairs | 278,876 | **190,027** | 245,400 | 231,987 | `pnglist_zstd9` |

`pnglist_zstd9` had the smallest complete output in **all 8** original natural groups. This is a credible outcome of a frozen experiment under the specified reference methods, NOT evidence that Git can never beat Zstandard under alternative object sets or settings.

## Verification coverage and interpretation

- **24 source-byte-exact fully self-contained saved files** = 8 cases × 3 methods. Every original reconstructed PNG file matched upstream raw bytes byte-for-byte and independent SHA256; native Git OIDs independently recomputed using the actual Git `blob NUL rawdata` hash construction.
- Verified **100 source PNG byte files** from true parent-and-commit pairs in their pinned Git histories and their original `modified` path status; original source RGB/A from PNG decoding separately audited.
- All 3 codec implementations selftested with metadata-only PNG edits: different original PNG raw bytes containing different ancillary chunks, yet identical decoded RGBA. Native Git PACK decode used an isolated *fresh* object database; absent encoding object DB does not give the Git method privileged access to side data.
- Native Git `pack-objects` compression level 9, delta search window 50, depth 50 and delta-base-offset were **registered before execution** and Git version is saved for reproducibility. The choice of these settings may not be optimal for all repositories; original Git commits' author/filename history is not stored in our benchmark archive (equal for all methods).
- This file-byte requirement is **stronger** than Exp90's exact-RGBA sparse modes, so it is a **scientific category error** to compare their 15,127-byte Wesnoth RGBA-only archive directly against this 70,694-byte PNG-file-byte-preserving Zstandard baseline and claim BLKH 4.7× better at *the same work*. The PNG metadata, palette/layout/source chunks, file-byte fidelity and complete source bytes define different product requirements.
- The benchmark's shared SHA1+SHA256 manifest gives robust source validation but does not store filename, copyright and attribution; embedding those **equally for all methods** would change byte totals and should be counted in future archive product decisions.

## Scientific/engineering next step

A defensible BLKH product roadmap must offer **two declared fidelity tiers**:
1. **RGBA exact sprite history**: output exact uint8 RGB/A array, optional decoder-critical minimal metadata, with a clear policy on whether source provenance lives inside the file. Strongest still-unbeaten source baselines include `atlas_webp`, `xor_webp`, `tile16_zstd` across Exp84–90.
2. **Original PNG file-byte exact revision history**: original source PNG binaries including metadata preserved; baselines now **include empirically superior `pnglist_zstd9`** in both studied sources, as well as tested native Git pack and bsdiff. A future BLKH design claiming this tier must handle original PNG metadata/lossless file byte recovery, not just RGBA.

A subsequent truly **independent frozen upstream** test is needed before any broad superiority claim. Do not retune historical groups or use retrospective source populations as independent verification. Test peak memory, CPU encode/decode throughput, original source license attribution and format versioning before any real release.

**Release remains HOLD**: research-only branch and draft PR #1, no changes to `main` or public demo, no performance-breakthrough advertising. This result primarily strengthens scientific baselines and exposes precise information-service requirements.
