# BLKH Experiment 89 — registered metadata-envelope packaging ablation, two known historic source populations

**Study type:** retrospective *engineering comparison*, not a new independent artistic-source holdout and not a newly invented compression algorithm. The datasets and original method standings of Exp87 (Wesnoth) and Exp88 (SuperTux) are already known. This record freezes the implementation and reporting protocol **before running the new header-repack measurements**. It cannot establish broad compression superiority regardless of a positive result.

## Test material, frozen upstream selection

1. Exp87: exact `research/dedup/experiment_87_wesnoth_real_maintenance.py` frozen 21 original Git PNG revision pairs, aggregate plus its 3 original commit groups.
2. Exp88: exact `research/dedup/experiment_88_supertux_png_history.py` frozen 29 original Git PNG revision pairs, aggregate plus its 3 original commit groups.
3. Recompute original archives with **all 14 unchanged existing Exp84 encoder methods** and exact original source commits and pixel hashes, keeping source-qualification and original PNG-byte checks. No data or codec tuning or source-group trimming after results.

There are **8 evaluation cases × 14 old formats = 112 complete original archival bitstreams**; and **two exact reversible metadata-envelope transforms** applied to every stream = **224 fully saved wrapped archives**. Total 336 actual persisted artifacts, all audited. This is not a meaningful replication of Exp87's small win: those data were previously studied.

## Treatments — full original bytes, no sidecar or decoder-free old data

Every original archive is `OLD_HEADER(10 bytes) || full canonical original JSON metadata || original exact compressed payload`, where `OLD_HEADER` is the historical `<5sBI` (magic `BL83P` or `BL84S`, old version=1, raw metadata byte count). The source PNGs and old pixels are encoded **inside** original original archives; no external model, shared dictionary or PNG file is provided to the decoder.

Define a new fully self-contained BL89R envelope:
- Fixed header `<5sB5sII` = wrapper magic `BL89R`, **algorithm 1=Zstandard level 9 or 2=zlib level 9**, original 5-byte historical format magic, uncompressed exact original JSON metadata size, and compressed metadata byte count.
- Exactly follows the compressed **original JSON metadata bytes**, then the **unchanged original payload bytes**. No removed provenance, no re-encoding of image content, no shared header sidecar, and no hidden bytes excluded from the total.
- Decoder independently decompresses the metadata and reconstructs the exact original historical archive **bit-for-bit** including old 10-byte header. Original codec then independently reconstructs byte-perfect RGBA. The wrapper is intentionally semantically neutral: only redundant full provenance-header bytes are entropy-coded.
- Bound uncompressed metadata to original format safety limits (<=220k BL83P, <=250k BL84S), old archive at most 25 MB, and all lengths checked; refuse truncated, overlong and malformed envelope frames. Save wrapper streams on disk and confirm their actual byte count and SHA256, restore original SHA256 and original RGBA hash list.
- Zstandard and zlib are **two packaging candidates**, not a new neural/BLKH content codec. Report both; select the shorter of these *two* for each original method as an explicitly retrospective descriptive comparison only, with no independent holdout success claim.

## Prespecified reporting questions

1. What percentage of original whole-file bytes is full JSON metadata, in each method and source group? How many bytes can general-purpose meta-compression save after counting the new header?
2. Original vs both treatments: full complete stream size, original-format source SHA, restored original SHA, restored exact original RGBA, time to unpack, independent Git source status and original bytes, updated per-method rank. **No partial-payload sizes**.
3. Does the narrow original Exp87 aggregate sparse lead of 457 bytes against 12 references persist **under identical metadata-envelope treatment for every method**? Does the Exp88 aggregate 55,733-byte sparse deficit persist? Treat loss/tie/win and absolute difference separately.
4. Distinguish common packaging/header improvement from new sparse image compression. If *every* method becomes smaller, it is an archive-container improvement for all baselines, NOT a BLKH content-only breakthrough. Do not manufacture a winner by comparing repacked sparse to un-repacked competitors.
5. Claims must note the re-packaged archive is fully self-contained for **RG**BA reconstruction but, except the bsdiff baseline, does not promise exact original PNG ancillary metadata. The independently reconstructed original `.bl83p/.bl84` is bit-for-bit identical for all methods.
6. Log Python, Zstd, Pillow, Numpy and zlib versions and methods' complete list. Save results and original/repacked actual bitstreams as time-limited Actions artifact, with source-rights notices. Do not upload external artwork into BLKH Git.

## Success/failure decisions

- **Technical success:** all **112 original** and **224 wrapper** files verified from disk, including bitwise original archive reconstruction and exact full decoded RGBA, with no source-pair qualification failures.
- **Useful size engineering result:** at least one treatment saves >0 whole-file bytes for both 21-pair and 29-pair aggregates in **at least 10 of 14 methods** on each. Even then this is an equal-metadata *generic header wrapper* finding, not sparse algorithm superiority.
- If any required source is unavailable or a safety bound fails, report inconclusive source qualification, never substitute source paths or art to obtain a favorable score.
- Result cannot independently validate the Exp87 win, so keep `main`, public demo, public breakthrough narratives and draft PR unchanged until properly replicated.

Source reference files and sample selection are the original protocols at `BHUH_EXP87_WESNOTH_REAL_MAINTENANCE_PROTOCOL.md` and `BHUH_EXP88_SUPERTUX_PREREGISTERED_PROTOCOL.md`.
