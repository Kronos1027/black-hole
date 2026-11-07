# BLKH Experiment 84 — frozen sparse exact-pixel patches vs verified historical sprite formats

**Preregistered BEFORE Exp84 patch-bitstream size measurements.** Motivated by Exp83P's negative results on 19 genuine Pixelorama Git-edited PNG pairs. This is a **deliberately exploratory re-analysis of the SAME 19 pairs**, not independent external holdout. No source change, pairing, block sizing or parameter tuning based on results.

## Historical original input

Exact 19 pinned, fully verified historical modified-PNG Pixelorama Git pairs and three original commits from [Exp83P protocol](BHUH_EXP83_PIXELORAMA_HISTORICAL_PROTOCOL.md). The images consist of old-parent and new-commit **actual** PNG blobs; 19/19 changed RGBA values. Reuse unmodified `experiment_83_pixelorama_git_history.py` `discover()` and `manifest_for()` methods for deterministic order and identical complete provenance records, original source PNG byte hashes and decoded RGBA hashes. Test all 19 pairs together and separately in three frozen natural Git-commit groups (13,4,2), each with its own complete archive. Treat the single application's four correlated groups as one source family, NOT four independent studies.

**All 8-bit straight RGBA bytes including RGB under alpha=0 must round-trip exactly**. Different old/new dimensions are preserved: map old top-left onto a zero-valued canvas of new dimensions (RGBA=0 outside old crop), then invert the new-image patch. Original old PNG source blob must be included as real encoded pixels (no uncounted decoder-side cached base).

## Frozen two sparse-change candidate formats

For each pair, store a **1-bit changed-pixel mask** for every pixel in the *new* image, bit-packed in row-major little-bit order with zero-filled trailing bits. Changed pixel means **any of four RGBA channels differs** from the old image's mapped pixel. Store, sequentially, the **full new 4-byte RGBA** pixel values for precisely those changed mask positions. The new image is reconstructed from the decoded old image plus replacement values. Both mask and replacement-color arrays are included in one globally compressed, length-prefixed patch stream; store all source metadata and individual patch lengths/counts in the file, no external state.

Candidate A, `sparse_webpold_zstd`: old PNG-version images packed as an **exact straight-RGBA WebP lossless** (`exact=True`) single shelf atlas using the same 1024px width/16px padding scheme, plus mask/changed RGBA payload globally compressed by **Zstandard9**. The old atlas geometry is serialized and validated.

Candidate B, `sparse_pngold_zstd`: the *original historical old PNG file bytes* (each length prefixed, ordered, then Zstandard9-compressed) plus the identical bit-mask/full-RGBA patch stream compressed by Zstandard9. This method **recovers original old PNG file bytes** exactly, while only requiring RGBA equivalence for new source files (new originals remain available as a reference baseline).

No adaptive dictionary, learned network, tuning, per-case color transform or individual source acceptance filters. Zstandard level fixed at 9; source SHA256, patch content complete lengths, atlas layout and source pixel hashes counted.

## Strong references

Re-run **all 11 actual Exp83P formats**, with the identical full source provenance metadata: pixel XOR/modulo WebP/Zstd and seven baselines (whole RGBA WebP/PNG atlas, raw atlas Zstd, original 38 PNG source bytes+Zstd, optimized individual PNGs+Zstd, exact 16×16 block dictionary WebP/Zstd). Compare new methods **to the smallest of all 11**, not merely the weakest or conventional PNG reference. Additionally include `bsdiff_png_zstd`: for each pair store **exact original old PNG bytes** and actual **bsdiff4.diff(old_png,new_png)** binary patches (length-prefixed together, globally Zstd9-compressed), reconstruct **both original old and new PNG byte streams exactly** using `bsdiff4.patch` and decode to RGBA. This binary-delta method is a **strong external reference**, not a new BLKH invention. Count its complete stream/manifest and validate the byte-identical old and new original PNG files.

**All 14 methods per group, 56 full archives**, must be saved, SHA256-checked and independently decoded to every original old/new RGBA byte; bsdiff must additionally reconstruct both old/new original PNG byte streams. Encoders may have different decoder-code footprints, deliberately not part of file-size counts; note this limitation.

## Frozen outcome and evidence standard

The best of **two** sparse candidates must beat the minimum of **twelve** references (all 11 Exp83P methods + bsdiff), including full file metadata, both for the **aggregate** 19 pairs AND at least **2 of the 3 per-commit groups** to meet the narrow *exploratory* win criterion. Otherwise mixed or negative. Report changed-pixel fraction and total per-source dimensions for each group; do not cherry-pick assets with few changes. A positive within-source finding is **not** scientific generalization and must be checked against a new source repository and Git packfiles/optimal binary deltas before being marketed.

**Research PR draft only**; `main`, demo and broad public claims remain unchanged regardless of this experiment.
