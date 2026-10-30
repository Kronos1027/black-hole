# BLKH Exp91 — PRE-REGISTERED exact original PNG-byte native Git-pack benchmark

**Reproducible, retrospective file-byte-compression engineering study**, registered before the Exp91 pack-size measurements. Historical source populations Exp87 and Exp88 have been studied previously; do **NOT** construe this as newly independent BLKH image-content algorithm validation. The target is **bit-for-bit original PNG file bytes** (including ancillary chunks, ICC and other metadata), **not merely decoded exact RGBA**. Only compare methods meeting this stronger target. Never compare Git's exact source bytes to a smaller BLKH pixel-only archive and claim an equal-fidelity win.

## Precommitted populations

The 21 original Wesnoth PNG revision pairs and 29 original SuperTux PNG revision pairs, exactly the Git commits/path-groups in frozen Exp87/Exp88 protocols; total **50 old/new original PNG pairs, 100 historical original PNG files**. In each project test 4 cases: whole original source and each of its three unchanged original commit subgroups. Source files must be refetched from original immutable upstream Git commit/parent, Git `status=modified` verified independently and raw PNG SHA256 independently checked. Any missing/oversized image invalidates that case, never replace.

## Fixed three exact-original-PNG-file codecs

Three existing/frozen compression concepts, **implemented inside the same explicitly identical fully self-contained source-order, hash and framing contract**:
1. `pnglist_zstd9` — concatenate for **every old and every new image** exactly `<I original PNG file size> + original PNG file bytes`, compress complete bytes with **Zstandard level 9**, no source pixels/images external.
2. `pngpair_bsdiff_zstd9` — concatenate for **each original Git edit pair** exactly `<I old PNG bytes length><I bsdiff patch length> + old original PNG bytes + bsdiff4.diff(old,new) bytes`, compress full bytes with **same Zstandard level 9**. Must decode with bsdiff4.patch and verify **new original PNG file bytes**, *including metadata*, not merely RGBA.
3. `git_pack_level9_window50_depth50` — create a temporary isolated local Git repository in SHA-1 object format, store each source PNG as a genuine Git **blob** with `git hash-object -w --stdin`; pass all object IDs to **native git pack-objects --stdout --window=50 --depth=50 --compression=9 --delta-base-offset**. Save the COMPLETE resulting standalone `PACK` byte stream inside the benchmark archive; do not count Git's separately constructed `.idx` file as required input because a decoder can recreate it using `git index-pack --stdin` on a fresh isolated repo from the pack bytes. Decode via native `git cat-file blob <object id>` and recover all original PNG bytes in original old/new order (including duplicate blob references). Include Git version in report.

The THREE method variants must receive an **identical common source-order manifest**:
- Fixed `<5sBBHI>` header (5-byte `BL91P` magic; version=1; method ID byte; uint16 number of source PNG files; uint32 complete compressed payload size).
- Exactly **20 raw bytes SHA-1 original Git blob OID + 32 original file-byte SHA256 bytes per each source PNG** in old/new order. No filenames, source paths or Git commit SHA inside any method archive (these attribution fields remain in separately archived research source inventory, NOT included in compared complete archive sizes, equally to all three methods). Explicit reduced provenance semantics, exact original PNG bytes retained, source-order recoverable by numbered OIDs.
- Then complete exact compressed payload stream.
- The Git archive must carry its full native pack bytes; all PNG originals and the binary-delta old side must be within each file, no outside Git object database at **decode** time. The temporary Git object database used for **encoding** and decoder's regenerated index are not counted as archive bytes, as they are rebuildable working state. Document time, Git version, memory constraints.
- All variants verify SHA256 on every reconstructed PNG and Git blob SHA1 hash; original PNG byte hashes equal against upstream originals. Verify all streams actually persisted on disk. Baseline differences in source functionality, framing, counting and OID list are eliminated across three methods. Source copyright/license provenance not embedded in these *storage-format* archives, and original authors must be credited separately for redistribution.

**Total:** 8 fixed cases × 3 methods = **24 saved fully standalone original-PNG-byte archives**, each independently audited from saved file. 100 upstream raw PNG source files independently refetched/audited. Check one source corpus per group and all 3 groups; no partial successes substituted for complete test.

## Frozen technical and scientific gate

- Technical success: independent Git-first-parent, Git-status source qualification, all 100 original source PNG file byte hashes; all 24 source-byte-accurate on-disk compressed archives reconstruct original historical PNG bytes **exactly in order**, including PNG ancillary metadata; verify SHA256 and SHA1 Git blob OID; execute Git pack decoding in an isolated fresh repo **without access to encoding object directory**; reject corrupt frames, truncated payloads, unexpected length, missing blobs and source changes.
- Report full bytes, time to encode/decode, Git version, winner each source group, original PNG-source total bytes and number of distinct Git object hashes, no missing provenance cost assumption.
- Do not claim a new neural codec advantage, or compare to sparse RGBA-only compression as if preserving original file bytes. The minimum of these 3 is the **same-fidelity exact-PNG-source-byte reference** under a codec-agnostic manifest.
- No changes to `main`, public demo or draft-PR status until independent transferable scientific evidence. If Git tooling/algorithm options unsupported on runner, treat as **execution qualification failure**, repair only tooling, not source selection or tuned parameters.

Direct reproducibility: `research/dedup/experiment_91_native_git_pack_png.py` and `.github/workflows/exp91.yml`.
