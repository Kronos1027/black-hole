# BLKH Experiment 90 — preregistered decoder-minimal metadata versus complete historical provenance

**Registered prior to Exp90 measurement; Exp87 and Exp88 original image data, source files and results already known.** This is a retrospective **format-engineering** ablation, **not** a newly independent compression benchmark or an algorithmic breakthrough.

## Frozen input: same unchanged 14 lossless codec payloads

Use the exact historical Git PNG pairs and fixed subgroups of:
- Exp87 `wesnoth/wesnoth`: 21 original PNG pair revisions (42 PNG files), 3 natural historical commit groups + whole corpus.
- Exp88 `supertux/supertux`: 29 original PNG pair revisions (58 PNG files), 3 fixed historical groups + whole corpus.
- Exactly **8 cases × 14 codecs = 112 original historical archive streams** from the same 2 sparse methods + all 12 original reference methods. All source paths, commits, decoding semantics, WebP/PNG/Zstandard settings, tile dimensions and old image input content unchanged. Original source Git SHA, file bytes, source license files and decoded full exact RGBA are independently refetched and audited as in Exp89.

## Three comparison arms, each applied to **every** method

Use old `<5sBI>` original prefix and exact old legacy codec binary payload **unchanged**:
1. **FULL-ZSTD:** Exp89 `BL89R` self-contained Zstandard-9 compressed entire original unmodified provenance header. Round-trips the **whole original archive byte-for-byte** (including full source metadata). This remains the strongest full-provenance standard within the current study.
2. **LEAN-RAW:** replace original verbose JSON manifest with exactly the codec-required dictionary `meta` below, canonical compact sort_keys JSON, retain old MAGIC/old version=1 and unchanged binary payload. The original decoders must decode this self-contained new file directly without any external original source reference, preserving exact original uint8 RGBA and required hash checks. This arm **deliberately discards historical provenance** such as Git commit SHA, original file path/name, license, upstream author and full original JSON, so it is **NOT** the same archival information service. Must be clearly labeled weaker metadata-retention semantics. Must not misrepresent as bit-identical original archive or original-PNG bytes recovery for non-PNG methods.
3. **LEAN-ZSTD:** apply the **same** Exp89 `BL89R` Zstandard-9 wrapper to the *lean-raw* archive for equal header coding. Reconstructs exactly the lean-raw archive, not the provenance-rich original. This has **same pixel reversibility** but deliberately **less historical provenance retained** than FULL-ZSTD.

The three arms are not equivalent in provenance, so never claim that LEAN-* beat provenance-preserving FULL-ZSTD on an apples-to-apples archival requirement. Compare BLKH sparse and all 12 reference methods **within each arm** only.

## Frozen required decoder manifest schema (candidate-minimal, JSON)

For each original codec:
- Always preserve: `codec`, `atlas_shape`, `atlas_sha256`, `source_sprites`; each sprite retains ONLY `x`, `y`, `width`, `height`, `pixel_rgba_sha256` in original order. These are required by current decoders for dimensions/placement and output data-integrity validation.
- Four legacy delta, 7 old baseline methods need `source_png_count` and `historical_pair_count` because existing legacy decoder validates them (11 legacy modes).
- If method `original_png_zstd`, each source sprite retains `source_png_sha256` as the native decoder verifies historical PNG file bytes. If method `sparse_pngold_zstd`, retain `source_png_sha256` on **old** image sources; if `bsdiff_png_zstd`, retain `source_png_sha256` on **both** old and new PNG files (required stronger actual Git source PNG byte integrity).
- If method `sparse_webpold_zstd`, additionally retain `old_atlas_shape`, `old_atlas_rgba_sha256`, and `old_atlas_positions` (only their `x,y,width,height` tuples), required for reversible old reference decoding.
- No other original full provenance fields are retained in LEAN. Minimize decoder-required fields, **not image-content bytes** and do not strip SHA256 checksums to fabricate size results. No custom encoder retuning, no changes to original exact lossless payload bits. Track exact full JSON manifest lengths and removed bytes for every codec.

Security/limits: Existing decoder shape and SHA256 checks remain active; enforce maximum number of sprites, source file/header bounds, unchanged old MAGIC/version, nonempty binary payload, source PNG hash integrity for the relevant codecs, and corruption/truncation self-test. All real files must be independently written, reopened, hashed and decoded (no hypothetical byte subtraction).

## Audit scope and frozen decision

Across 8 cases × 14 unchanged codecs:
- **112 original whole archives** + **112 FULL-ZSTD** + **112 LEAN-RAW** + **112 LEAN-ZSTD** = **448 saved, complete, auditable file streams**.
- Original source Git path `status=modified`, first parent, all 100 original source PNG bytes/hashes (50 source pairs, 2 versions each), decoded exact uint8 RGBA including invisible RGB, codec-specific original PNG byte integrity verified as before.
- For FULL-ZSTD, restore **exact original full original archive**; for LEAN-ZSTD, restore exact lean archive; LEAN-RAW decode with the unmodified correct codec. SHA256-check every actual disk file, complete bytes (header + manifest + payload), exact RGBA all arms.
- Produce aggregate and three subgroup method rankings per source in **each** of FULL-ZSTD, LEAN-RAW and LEAN-ZSTD, comparing BLKH to strongest of all 12 valid references **using only sizes from the same arm**. Freeze existing Exp87 / Exp88 original archive size guards.
- A **technical success** requires all 448 files and source identities passing independent CI. Engineering success, separately, requires ≥10/14 codecs in **both aggregate sources** to have LEAN-RAW strictly smaller than original legacy full-JSON complete archive. Smaller LEAN-ZSTD vs FULL-ZSTD may be reported as *provenance/metadata feature tradeoff*, **not a same-functionality compression win**.

No claims of broad SOTA, positive independent transfer, or world-leading algorithms. No public demo or `main` changes. PR remains draft. Follow-up needed for native Git/PNG-file byte delta and production codecs with optimized metadata; this study does not include those external baselines.
