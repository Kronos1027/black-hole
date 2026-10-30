# BLKH Experiment 85 — independent Mindustry real Git sprite history (MIXED; preregistered gate FAILED)

**Official reproducible status:** ✅ successful GitHub Actions; all **56 saved complete lossless archives** and the genuine old/new **30 PNG file pairs** were independently re-opened, SHA256 verified and decoded with all source uint8 RGBA channels including RGB under fully transparent pixels. This is an **independent source repository** from previously analyzed Pixelorama, but only *one* new project, not a broad artist population.

- [Protocol frozen before inspecting the PNG pixel data](BHUH_EXP85_MINDUSTRY_GIT_SPRITES_PROTOCOL.md) and [experiment implementation](experiment_85_mindustry_history_holdout.py).
- [Official successful workflow and independent source audit](https://github.com/Kronos1027/black-hole/actions/runs/40924630193); its `exp85-genuine-mindustry-source-edits-vs-bsdiff` artifact contains the saved original Git PNG bytes, actual compressed bitstreams, checksums, experimental parameters and `results.json`.
- Source original repo: `Anuken/Mindustry` at https://github.com/Anuken/Mindustry, root GPL-3.0. Historical file paths/commits and exact first-parent hashes are pinned in the protocol and embedded within each source-data manifest, with license attribution.
- Frozen source selection used historical Git `status=modified` filenames and commit provenance (not observed decoded RGBA similarity or compression results), followed by actual source validation. All **30/30 modified PNG pairs also had genuine decoded RGBA changes**.
- Four fixed source groups: one complete 30-pair bundle; natural commit subgroups `defense` (2 pairs), `texture_cleanup` (15 pairs) and `walls` (13 pairs). These groups have shared creator/development context and are **not independent repositories**.

## Same unchanged experimental codec methods from Exp84

The two experimental sparse approaches are *identical in representation and fixed settings to Exp84*: (A) exact old RGBA WebP atlas plus globally Zstandard9-compressed 1-bit mask and four new RGBA values per changed pixel, or (B) original old PNG file bytes+Zstandard9 plus the same complete patch. The old image's real pixels are not available free to the decoder; the archive includes them.

Exactly **12 strong byte-counted reference methods** were run without retuning: four established XOR/modulo RGBA delta methods with WebP lossless or Zstandard; exact RGBA atlas WebP and PNG; complete raw atlas Zstandard; exact original PNG source bytes jointly compressed with Zstandard; individually optimized PNGs+Zstandard; exact 16px tile dictionary+WebP/Zstandard; and genuine `bsdiff4` binary differences on original PNG file pairs followed by Zstandard.

All methods encode the same complete original Git commit/parent paths, source file hashes, old/new RGBA hashes, independent image geometry and shared source manifest. Every saved archive is fully decoded and verified from its **actual persisted bytes**, not an in-memory estimate or test made only against PNG.

## Official complete-file measurements

| Real source group | Actual changed RGBA pixel fraction | Best experimental sparse archive | Size | Best of 12 existing references | Size | Win? |
|---|---:|---|---:|---|---:|---|
| All **30** genuine PNG revisions | **2.894%** | sparse_webpold_zstd | **85,390 B** | **xor_webp** | **85,355 B** | **No** (+35 B / 0.041%) |
| `defense` **2** pairs | **2.856%** | sparse_webpold_zstd | **8,146 B** | **xor_webp** | **7,830 B** | **No** (+316 B / 4.04%) |
| `texture_cleanup` **15** pairs | **0.742%** | **sparse_webpold_zstd** | **43,995 B** | xor_webp | 44,424 B | **Yes** (−429 B / −0.966%) |
| `walls` **13** pairs | **8.218%** | sparse_webpold_zstd | **35,791 B** | **atlas_webp** | **34,864 B** | **No** (+927 B / 2.659%) |

**Prespecified independent-source transfer criterion:** aggregate 30-pair win **and** at least 2 of 3 natural subgroup wins. **FAILED**: aggregate custom loses by **35 bytes** and only `texture_cleanup` succeeds, **1/3** subgroups.

The small 429-byte advantage in a genuine independently sourced repository is meaningful as a *bounded engineering observation*: a sparse exact edit representation can save bytes when nearly all source pixels are unchanged. Yet it **does not** support the claim that this technique is universally competitive: on these same real source histories it loses the full-bundle comparison by 35 bytes and two additional subgroups. Exp84 on 19 real Pixelorama PNG edits also lost in every one of its four groups. This means there is no replicated whole-corpus advantage yet.

## Limitations, statistical integrity and product decision

- One new independent source repository, not 30 independent artist teams. The natural subgroups are correlated; repeated testing same source cannot substitute for out-of-repository validation.
- Changed-pixel proportion was measured as an exploratory explanatory variable, **not an input used to preselect successful examples**. No post-hoc edited-path exclusion, compression quality relaxation or selector tuning.
- The snapshot file sizes include complete source metadata and all required bytes; minor size differences may be sensitive to the shared encoding library version, source selection, filename manifests and atlas packing.
- The bsdiff reference guarantees byte-identical original PNG file blobs, whereas WebP/RGBA methods guarantee exact decoded RGBA channels (including hidden RGB), not PNG metadata/ICC identity. Both targets were disclosed and verified appropriately; do not treat exact file bytes as a feature of the RGBA-only codecs.
- No source-independent throughput/working-set profiling, no native Git packfile/delta representation comparison beyond bsdiff, and no production distribution testing. No new neural network or novel information-theoretic principle is claimed.

**Scientific decision: narrow mixed result, preregistered hypothesis not confirmed.** Exp85 improves external validity over purely synthetic image edits but cannot justify describing BLKH as world-leading universal compression. The genuinely favorable `texture_cleanup` group is a good *future preregistered application hypothesis* to test on another unrelated authoring repository: small-area real sprite modifications. Avoid inspecting and retuning on these same images to manufacture wins.

**Public release: HOLD.** Keep `main`, public demo and any scientific breakthrough marketing unchanged; draft research PR #1 remains unmerged.
