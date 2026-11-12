# BLKH Exp90 — decoder-minimal manifest, source reconstitution verified; metadata-sensitive sparse wins

**Research/engineering result: SUCCESSFUL TECHNICAL IMPLEMENTATION; NOT NEW GENERAL COMPRESSION SUPERIORITY.** Official [GitHub Actions run 40999090023](https://github.com/Kronos1027/black-hole/actions/runs/40999090023) completed and independently verified **448 real persisted archives** (112 original + 336 treatments), **100 authentic historical source PNG file versions** (50 source revision pairs), all codec-specific direct decode and exact uint8 RGBA hashes. Artifact: `exp90-full-vs-decoder-minimal-448-self-contained-archives` GitHub Actions artifact ID `12952905505`, 72,391,325 bytes in the compressed ZIP, retention 30 days. Frozen [protocol](BHUH_EXP90_DECODER_MINIMAL_METADATA_PROTOCOL.md) committed at `561f77bee6296afa55211cf483b6a3230db7c82e` **before Exp90 measurements**.

## Comparison target and retention semantics

The two source datasets are reused **retrospectively** (known positive Exp87 Wesnoth, known negative Exp88 SuperTux). This is **NOT** fresh independent confirmation or tuning on an unseen test set. Both original paired PNG source histories are refetched and independently verified by immutable Git first-parent/commit, Git status=modified, original raw PNG byte SHA256, and original full exact decoded RGBA. The **same existing 14 methods** are reproduced unchanged: 2 experimental sparse modes and 12 comparison lossless methods.

All three treatments preserve **exact original compressed image payload bytes**, including the necessary encoded old images; none receives a separate old-source sidecar or omits image content:
- **FULL-ZSTD**: the existing Exp89 `BL89R` Zstd9 envelope wraps complete original historical JSON + payload; decoder can reconstruct the **full original old archive bit-for-bit including provenance**, then decode exact image pixels. This preserves full research attribution.
- **LEAN-RAW**: an original-family `BL84S`/`BL83P` version=1 archive with its JSON reduced to only **existing decoder-required per-method** layout geometry, source image RGBA SHA256, total atlas SHA256, old atlas geometry/checksum only when needed, original PNG source-byte SHA256 only for methods whose legacy decoder checks the original bytes, codec method and two legacy count fields. All historical provenance strings, original path names, Git commits, source license, and other non-decode-critical metadata are intentionally **lost**. Original image payload is not modified. **Directly decoded by original unmodified decoder**; exact original RGBA preserved. LEAN-RAW is not equivalent to a provenance-preserving archive; source filename/attribution may require independent storage for lawful republishing.
- **LEAN-ZSTD**: same loss-of-provenance lean manifest, additionally compressed with Exp89 self-contained Zstd9 envelope. Reconstructs lean-raw bit-for-bit, not the full historical source archive. Pixel reconstruction remains exact.

### Audited aggregated sizes, full actual bytes

| Frozen source and comparison | Best experimental sparse | Best of all 12 fixed references | Sparse-minus-best-reference |
|---|---:|---:|---:|
| Exp87 Wesnoth (21 genuine Git PNG source pairs), original FULL-RAW | 64,802 B | 65,259 B, `xor_webp` | **−457 B (sparse wins)** |
| Exp87 FULL-ZSTD complete provenance | **18,201 B** | 18,661 B, `xor_webp` | **−460 B (sparse wins)** |
| Exp87 LEAN-RAW decoding only | 19,701 B | **19,240 B, `xor_webp`** | **+461 B (sparse LOSES)** |
| Exp87 LEAN-ZSTD decoding only | **15,127 B** | 15,467 B, `xor_webp` | **−340 B (sparse wins)** |
| Exp88 SuperTux (29 genuine Git PNG source pairs), original FULL-RAW | 430,620 B | **374,887 B, `tile16_zstd`** | **+55,733 B (sparse LOSES)** |
| Exp88 FULL-ZSTD complete provenance | 371,953 B | **316,223 B, `tile16_zstd`** | **+55,730 B (sparse LOSES)** |
| Exp88 LEAN-RAW decoding only | 374,265 B | **317,241 B, `tile16_zstd`** | **+57,024 B (sparse LOSES)** |
| Exp88 LEAN-ZSTD decoding only | 367,863 B | **311,921 B, `tile16_zstd`** | **+55,942 B (sparse LOSES)** |

**Methodologically crucial:** Exp87's tiny 457-byte sparse victory from the older verbose format **reverses** into a **461-byte deficit in LEAN-RAW** when equally removing unnecessary manifest fields for every competitor. Applying equal header compression to LEAN-ZSTD yields a **340-byte positive gap**; FULL-ZSTD also holds a narrow 460-byte sparse win. This makes the **archive header representation** an important confounder of the small raw-byte advantage, not proof of a broadly stronger sparse image codec. Header choices did NOT make BLKH competitive with SuperTux; the 55–57 KB deficit persists.

## Why we report service semantics honestly

The new direct lean archive is fully self-contained for **exact decoded RGBA** with integrity hashes and original old images still included, but it intentionally **does not retain** original filenames, upstream commit history, copyright/attribution strings, or uncompressed full JSON. An archive whose product promise includes *source attribution and exact archival provenance* **must** separately store those fields (and count their bytes) or use FULL-ZSTD; LEAN formats are not an equal-feature overall size replacement for FULL formats. Likewise, the older pixel decoders reconstruct original **PNG ancillary chunks/file bytes** only in byte-preserving formats (e.g. `original_png_zstd`, `bsdiff_png_zstd`), while other methods promise exact decoded RGBA, not exactly original PNG source metadata.

The large numerical decrease from 64,802 B to 15,127 B on Wesnoth is a combination of **losing verbose provenance** and **compressing the lean metadata**, not an improved image-content compression algorithm. FULL-ZSTD offers provenance-preserving 18,201 B without this trade-off; LEAN-ZSTD loses that extra information for another 3,074 B. The strongest reference `xor_webp` similarly shrinks 18,661→15,467 B (3,194 B extra), so the gap between sparse and reference becomes **120 B less favorable** to sparse under lean compression.

## Independent verification of 448 actual persisted complete archives

1. **112 original** source-data-embedded archives (14 × 8), verified full saved size, SHA256 and identical original decode to all 50 old/new PNG pairs. Frozen original aggregate sizes checked (64,802/65,259 and 430,620/374,887) to prevent accidental codec parameter drift.
2. **112 FULL-ZSTD** archives, independent bit-for-bit **whole original archive restoration** and source PNG RGBA hash validation.
3. **112 LEAN-RAW** archives, verified **exact same original compressed image payload** bytes and direct decode with *existing, unchanged codec decoder*. The auditor independently checks the exact allowlisted **per-codec field set**, per-image checksum and original historical sources.
4. **112 LEAN-ZSTD** archives, exact byte-for-byte restoration of the verified lean-raw archive and same exact RGBA reconstruction.
5. CI refetches **100 actual Git-stored original PNG file bytes**, verifies corresponding first-parent Git `modified` statuses and original file/pixel SHA256s and declared source categories; keeps source-license/attribution files alongside time-limited archive artifacts.
6. Ran synthetic malformed/truncated header and pixel-integrity selftests and independently recomputed all per-arm rankings, without comparing repacked BLKH against old un-repacked references.

The registered engineering test requires LEAN-RAW to reduce 10 or more of the 14 complete original legacy archives in **both** aggregate datasets. Both passed **14/14** compared to original FULL-RAW historical files. Again, that saves bytes by **dropping old provenance**, and is not a content codec victory.

## Research boundary and next defensible experiment

- Neither historical Wesnoth nor SuperTux source selection is independent of prior Exp87/88 analysis. This is an explanatory **format ablation**, not transfer validation.
- The lean output reuses the prior `BL83P`/`BL84S` on-disk magic/version and therefore is **decoder-compatible for the tested implementations**, but its missing historical provenance makes it *semantically not backward-compatible with software relying on source-history fields*. A deployable format must have a new feature/version identifier, clear rights and metadata policy, and a stable migration strategy rather than silently overwriting a preservation standard.
- No claims of better native Git packing or native PNG file-byte compression are supported here; native Git `pack-objects` and other PNG byte-preserving binary delta/optimized metadata **remain an independent comparator task**. That comparison must specify whether exact PNG original file bytes or only decoded RGBA are required, since those are different workloads.
- Header format, metadata constraints, Zstandard version, times and maximum RSS can be studied under frozen conditions; adding another two-source retrospective header improvement cannot replace a truly independent source test.

**Release: HOLD.** Research changes only in draft PR #1; leave `main`, public demo and breakthrough claims untouched. This experiment provides actionable design insight: **make metadata semantics explicit, and benchmark codec content only under equal format/service requirements**.
