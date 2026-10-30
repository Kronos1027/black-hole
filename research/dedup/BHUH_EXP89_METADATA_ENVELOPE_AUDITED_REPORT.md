# BLKH Exp89 — full self-contained Zstandard/zlib metadata envelopes: SUCCESSFUL generic engineering improvement, NOT a sparse-content breakthrough

**Official CI:** [40944170960](https://github.com/Kronos1027/black-hole/actions/runs/40944170960) — technical success. **Reproducible artifacts:** `exp89-two-source-compressed-metadata-complete-336-archives` artifact ID `12929070438` (55,036,092 compressed ZIP bytes; retention 30 days at GitHub Actions). Report authored after official test; prior frozen [protocol](BHUH_EXP89_METADATA_ENVELOPE_PROTOCOL.md) committed as `ab2472738dede88f12c71f20ba4a9bfcfb065b69` **before** new wrapper measurements. The Exp87/88 source datasets and old rankings were already known: this is a *retrospective packaging experiment*, **not independent confirmation of source transfer**.

## What physically changed

The original 14 codec output streams (2 sparse, 12 non-sparse references) carry full, uncompressed provenance manifests in a 10-byte old-format header + exact JSON + compressed image source/payload bytes. New `BL89R` envelope is a fully self-contained 19-byte header + Zstandard level-9 (or zlib level-9) compressed **original JSON metadata bytes** + **unaltered original compressed image payload bytes**. All original old image data, source filenames, SHA256s, PNG-source metadata and original paired source manifest remain **inside each file** — no omitted external sidecar, no free upstream decoder side information or loss of original original JSON fields. A BL89R reader must restore the original `.bl83p/.bl84` archival bytes **bit-for-bit** before the original decoder independently reconstructs original RGBA pixels byte-perfectly.

**All formats receive the SAME common packaging improvement.** This makes the new values a codec-neutral storage engineering improvement, NOT a novel BLKH pixel or sparse codec advantage. The original compressed image content streams are identical before and after packaging. Other native deployments may compress metadata differently or omit unnecessary fields.

The complete implementation is [experiment_89_metadata_envelope.py](experiment_89_metadata_envelope.py) and independent auditor/workflow is `.github/workflows/exp89.yml`. No changes to sparse masks, old codec settings, sources, source groups, 12 reference methods or original images.

## Official observed aggregate compressed *whole archive* lengths

| Source | Original BLKH sparse | Same BLKH + Zstd9 manifest | Original best of 12 references | Same best reference + Zstd9 manifest | Original sparse gap | Repacked sparse gap |
|---|---:|---:|---:|---:|---:|---:|
| **Exp87 Wesnoth (21 Git PNG pairs)** | 64,802 B | **18,201 B** | 65,259 B (`xor_webp`) | **18,661 B (`xor_webp`)** | **457 B smaller** | **460 B smaller** |
| **Exp88 SuperTux (29 Git PNG pairs)** | 430,620 B | **371,953 B** | 374,887 B (`tile16_zstd`) | **316,223 B (`tile16_zstd`)** | **55,733 B larger** | **55,730 B larger** |

The packed Wesnoth sparse archive saves **46,601 B, 71.9129% of its original entire 64,802 B file**, while the strongest baseline `xor_webp` itself saves **46,598 B** from the same header-repack technique. On SuperTux, packed sparse saves **58,667 B / 13.6238%** and the strongest `tile16_zstd` baseline saves **58,664 B**. Nearly identical absolute savings show that redundant/shared provenance metadata — **not changed image compressibility** — accounts for essentially all of these improvements. The 457 B advantage observed earlier in Wesnoth remains approximately the same (**460 B**) when both competitors are treated fairly.

The **alternative zlib9** wrapper also saves substantial header bytes but is larger than the Zstd9 wrapper in these aggregate files: Wesnoth sparse 18,733 B vs ref 19,193 B; SuperTux sparse 373,891 B vs ref 318,160 B. Zstd9 is a measured superior packaging choice for the current two populations, not a proof for all possible sources.

## Prespecified *all group* controls: unchanged winner identity

| Preregistered original Git group | Original sparse vs best-ref gap | Zstd9 sparse vs best-ref gap | Result after identical repacking |
|---|---:|---:|---|
| Exp87 Wesnoth aggregate | −457 B | **−460 B** | Narrow sparse win persists |
| Exp87 Thea shadow fixes | −181 B | **−185 B** | Narrow sparse win persists |
| Exp87 Dune firetrooper edits | +173 B | **+171 B** | Sparse loses |
| Exp87 Ant revisions | +23 B | **+19 B** | Sparse loses |
| Exp88 SuperTux aggregate | +55,733 B | **+55,730 B** | Sparse loses |
| Exp88 Rock/fallblock artwork | +28,769 B | **+28,764 B** | Sparse loses |
| Exp88 Autotile relayout | +27,470 B | **+27,467 B** | Sparse loses |
| Exp88 PNG-metadata-only Penny changes | +1,540 B | **+1,535 B** | Sparse loses |

The new header improves every candidate and every reference at substantially similar byte-count scale in each group. **There is no reversal of any one of eight source group outcomes.** Thus header compression is an excellent engineering improvement and is **not** evidence that the sparse algorithm transfers to more sources.

## Independent audit

CI explicitly:
1. Refetched and compared **all 100 immutable original Git PNG source bytes** across Wesnoth and SuperTux, including corresponding Git `modified` path status and first-parent commits. It also checks 21- and 29-pair source populations and RGBA pixels.
2. Persisted and independently reopened **all 112 original exact lossless archival streams** (14 codecs × 8 source groups), checking actual SHA256, metadata and original decoded RGBA.
3. Persisted and independently reopened **all 224 self-contained BL89R streams** (two compression wrappers × those 112 archives), checked exact file length, SHA256, rebuilt **bit-identical original archives**, and decoded byte-exact RGBA including hidden RGB beneath alpha=0. The `bsdiff4` historical PNG method additionally enforces original PNG source-byte SHA.
4. Checked strict codec identities and original known aggregate sizes (Wesnoth 64,802/65,259 and SuperTux 430,620/374,887) as drift guards; emitted all 8 outcomes and alternative zlib9 values.
5. Passed corruption/truncation/adversarial header-self-tests, including both original archive magic families and both wrapper formats.

**The frozen engineering threshold passed**: at least 10 of 14 codecs became smaller in both aggregate sources under one of the two fixed metadata-envelope modes, after including full new-wrapper overhead. The complete per-codec byte counts are in the saved CI results; the audit directly recomputed the frozen ≥10/14 threshold and returned true. This must not be interpreted as an independently verified compression state-of-the-art.

## Scientific limits and next step

- Exp87+Exp88 were already measured earlier: this is intentionally a **retrospective ablation**, not independent source replication, statistical confirmation, or a new image compression algorithm.
- These records often include repeated long source path strings and verbose Git provenance; the ~72% size reduction in small Wesnoth archives reflects highly compressible uncompressed manifest fields. An optimized production encoder might have used **only the decoding-required fields** or compressed the entire old container, reducing the baseline in another way.
- The new BL89R is an **external archival wrapper**: it must first reconstruct the older original stream in memory. Running times and peak memory should be benchmarked before any claim it is the best production deployment.
- Original format still reconstructs exact RGBA only unless the `bsdiff4` method; ordinary PNG ancillary chunks are not promised to round-trip.
- The win/loss gaps changed only 2–5 bytes because almost all provenance is codec-invariant. Sparse vs best baseline is **not** newly improved on image content itself.

**Next scientifically grounded work**: evaluate lean **per-codec actually necessary metadata** with a backward-compatible standalone decoder and measure source-native `git pack-objects` / binary PNG delta baselines under an independently frozen protocol. Compare complete end-to-end artifacts and integrity; do not compare new BLKH wrapper to unoptimized competitors.

**Release decision:** PR#1 remains draft; `main` and the public demo unchanged. No claim of universal BLKH compression, neural breakthrough, or a product-level image-compression record.
