# BLKH Experiment 87 — original Wesnoth sprite-maintenance Git history: MIXED, aggregate SMALL WIN, preregistered transfer FAIL

**Official status:** GitHub Actions passed the full independent source and artifact audit: https://github.com/Kronos1027/black-hole/actions/runs/40931215958. All **56 actual persisted, complete lossless archives** were reopened, SHA256-checked and decoded back to their exact source uint8 RGBA arrays (including hidden RGB under 0 alpha). All **21 original historical Git PNG old/new pairs** were also refetched, original PNG file hash checked and Git status verified as `modified`; the bsdiff4 reference additionally reconstructs its actual original PNG bytes. This report does not treat small numeric advantages as a universal compression breakthrough.

## Source and locked methodology
- Frozen *before accessing any selected source PNG pixels or compression sizes*: [BHUH_EXP87_WESNOTH_REAL_MAINTENANCE_PROTOCOL.md](BHUH_EXP87_WESNOTH_REAL_MAINTENANCE_PROTOCOL.md), commit `30cf25bfbc3b8595775a3a6de4ca9fd4d2af5d69`.
- Upstream: `wesnoth/wesnoth` https://github.com/wesnoth/wesnoth, new unrelated source repository relative to Pixelorama, Mindustry and Godot. Source original root COPYING (GPL-2.0-or-later) and contributor `copyrights.csv` retained in the successful CI artifact. Individual artwork ownership and any exceptions should be inspected before republishing those images. Original artwork not committed to this BLKH source repository.
- Source groups frozen from Git commit *messages and filename/status metadata*, not from measured changed-pixel fraction or compressed size:
  - `thea_shadow_fix`: 14 status-modified PNG pairs, source commit `ee27cba746684bd72fb0499226db381bf85db5bd`; **14/14 genuinely changed RGBA**.
  - `dune_firetrooper_cleanup`: 4 PNG pairs, `5d0c0ec39384c364d716fa5039d53913da4b1e20`; **2/4 genuinely changed RGBA**. Retained the two metadata-only PNG source changes in the test; no favorable exclusion.
  - `ant_sprite_distinction`: 3 PNG pairs, `811a2c5b05df2d7e48c57f49c4f76c1c3f3923fc`; **3/3 genuinely changed RGBA**.
  - Aggregate: 21 original Git PNG pairs, **19/21 genuinely changed RGBA**.
- Code: [experiment_87_wesnoth_real_maintenance.py](experiment_87_wesnoth_real_maintenance.py), CI: `.github/workflows/exp87.yml`. Both sparse candidates and all 12 stronger lossless reference methods reused without retuning from Exp84–86; equal complete old source pixels, origin metadata, indices and byte-counted archive framing across the fixed within-run comparison. No decoder receives the old image for free.

## Official complete-archive results

| Frozen original source group | Changed RGBA target pixel fraction | Best sparse (bytes) | Best of all 12 references (bytes) | Strongest reference | Sparse difference |
|---|---:|---:|---:|---|---:|
| **All 21 authentic history pairs** | **34.3184%** | **64,802** | **65,259** | `xor_webp` | **−457 B (−0.7003%): WIN** |
| `thea_shadow_fix` (14) | 49.0465% | **43,783** | **43,964** | `xor_webp` | **−181 B (−0.4117%): WIN** |
| `dune_firetrooper_cleanup` (4) | 5.1312% | 13,292 | **13,119** | `atlas_webp` | **+173 B (+1.3187%): LOSS** |
| `ant_sprite_distinction` (3) | 2.7755% | 9,939 | **9,916** | `xor_webp` | **+23 B (+0.2319%): LOSS** |

Winning experimental mode for all four groups: `sparse_webpold_zstd`. Every reported byte count is the length of a **saved actual compressed file**; the independent verification reopens and decodes every file. No in-memory hypothetical compression ratios, no PSNR-based claims, no lossy concessions.

**Pre-registered transfer criterion:** strictly beat all 12 references in the complete 21-pair bundle **AND** in ≥2 of 3 original commit subgroups, with ≥2 genuinely changed RGBA files and all 56 archives independently verified. **Outcome: FAIL** because aggregate wins, but only **one of three** subgroups wins. This is a legitimate one-corpus positive result, *not* a confirmed independently replicated, broadly transferrable result. Relative improvements <1% are particularly sensitive to full metadata serialization, encoder library versions and exact dataset composition.

## What this adds to the evidence

- Exp84 on 19 genuine Pixelorama old/new edits: aggregate LOSS; 0/3 group wins.
- Exp85 on 30 genuine Mindustry edits: aggregate LOSS by 35 B; one genuine `texture_cleanup` subgroup win by 429 B.
- Exp86 on 14 genuine Godot sprite edits: aggregate LOSS by 45,242 B; 0/2 group wins.
- **Exp87 on 21 genuine Wesnoth sprite edits: aggregate WIN by 457 B; 1/3 subgroup wins.**
- Combined, those separately prespecified source tests do **not** establish a general algorithmic advantage. No single changed-pixel-fraction cutoff was validated: within this run 49.05%-changed `thea_shadow_fix` wins slightly, while 2.78%-changed `ant_sprite_distinction` loses slightly. Source composition and competing codecs matter more than a guessed monotone threshold.

## Scope and next test gate

1. Treat the real, reproducible 457-byte aggregate advantage as **hypothesis-generating** and narrowly conditional. Do not promote it as a novel state-of-the-art compressor or product breakthrough.
2. Prior before next source selection, specify a new **unrelated fourth/fifth artist-source project** and immutable Git source list; test aggregate and natural subgroups without tuning codec parameters after reading pixels. The exact same 14-codec benchmark can be used to assess independent corpus replication.
3. Consider a future independent industrial baseline using source-minimal metadata for each codec and native Git packfile compression; the existing manifest contains shared provenance fields some codecs may not need, so the present ratios are a controlled specific representation comparison, not fully optimized production competitiveness.
4. Work on source-independent throughput, memory limits, and robustness while avoiding changing compression parameters during scientific validation.

**Release decision remains HOLD**: research branch/PR #1 only, no merge to `main`, no change to public demo, no unqualified LinkedIn breakthrough claim. Small result is scientifically worth preserving precisely because the prespecified gate still failed.
