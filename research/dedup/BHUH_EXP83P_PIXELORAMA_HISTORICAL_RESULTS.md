# BLKH Exp83P — **19 real Git-edited PNG pairs**: honest mixed exact-block reuse, no delta-codec win

**Official status: GitHub Actions PASSED twice, including an independent source/download/status audit.** Results below are from the **second, final named-file run**, with all saved files re-opened from the released Actions artifact and their byte length/SHA256 verified again. **This is not a broad compression breakthrough or evidence of a new neural model.**

- **Frozen protocol, committed before this experiment downloaded/decoded the selected historical PNG pixels:** [BHUH_EXP83_PIXELORAMA_HISTORICAL_PROTOCOL.md](BHUH_EXP83_PIXELORAMA_HISTORICAL_PROTOCOL.md).
- **Reproducible source-aware benchmark:** `research/dedup/experiment_83_pixelorama_git_history.py`.
- **Independent validation workflow:** `.github/workflows/exp83-pixelorama-history.yml`.
- **Final successful GitHub Actions run:** https://github.com/Kronos1027/black-hole/actions/runs/40797786412.
- **Evidence:** `exp83p-pixelorama-historical-19-png-pairs` official artifact with original old/new Git PNG source bytes, `real_git_source_inventory.json`, `results.json`, four source atlas PNGs and 44 exact `.bl83p` full-size archives. The first successful run, prior to a filename/protocol-string-only correction, is https://github.com/Kronos1027/black-hole/actions/runs/40797209864; results remained numerically unchanged.

## Provenance: authentic historical source file edits, not synthetic assets

Source: public `Orama-Interactive/Pixelorama` pixel-art application, root MIT LICENSE, from three **fixed Git commits** each compared with its actual first parent. The source filenames were first chosen from GitHub's `status=modified` comparison **before examining original RGBA pixels or encoding results**, to avoid choosing only compressible modifications. Only built-in `assets/graphics` paths were selected (no external addons or splash illustration art):

| Real commit group | Pinned commit SHA | Matched original PNG paths | Changes in decoded RGBA |
|---|---|---:|---:|
| Pan/navigation icons in different themes | `9f8c9807e789bf476f95ac7045409e943b5fc328` | 13 | **13** |
| Curve/3D shape tool icons | `9611bf72631c3eaed8e4f0281a6628e069ecaa63` | 4 | **4** |
| Isometric box tool icon changes | `13c2af3a2a53bfa1439d7a6ca0e3ddbd136a91b5` | 2 | **2** |

All **19/19 original PNG files truly changed pixel content**; no merely re-encoded-but-identical PNG slipped through. Pixelorama is one project, and theme-specific icons/commits are correlated, **not 19 independent artist families**. The pooled all-19 archive is an **experimental union of three commit-specific old/new edit cohorts**, not a historical full-repository snapshot diff between one pair of project versions.

For each changed path, the workflow obtains the actual Git first-parent SHA via GitHub's commit API, confirms `modified` file status in GitHub compare, downloads **both** original PNG contents from the exact refs and records raw PNG byte SHA256, exact decoded RGBA byte SHA256, dimensions, commit and file path. No images were manufactured, recolored, resized or retuned, and any differing source dimensions were handled by an explicitly reversible zero-outside-old spatial rule.

## Lossless comparison with real original-source file formats

Every codec receives **the same full counted source metadata** and the same exact ordered first-fit 16px-aligned RGBA atlas. Target losslessness means reconstruct every original **uint8 RGBA pixel including RGB under alpha=0**, without claiming bit-identical PNG metadata/ICC preservation for re-encoded formats.

New four source-aware predictors (all established mathematical transforms, not novel): old image stored unchanged; new version represented as per-channel **XOR(old,new)** or **modulo-256 (new−old)**, with RGB+alpha transformed, and full atlas encoded as WebP lossless `exact=True` or ZSTD9.

Seven strong unmodified references: single WebP lossless RGBA atlas, PNG atlas, raw RGBA atlas+ZSTD, original length-prefixed source PNG files+ZSTD, independently optimized PNG files+ZSTD, existing **16px exact SHA256 tile dictionary** compressed with WebP lossless and counted uint16 index, and the equivalent dictionary with ZSTD9. This experiment compares the **new** differential method with the **best of all seven** and separately checks the preexisting tile dictionary against **non-indexed** standards.

All **44 complete saved archives** (11 methods × [the combined cohort and each of the 3 individual commit cohorts]) were independently loaded and decoded in GitHub Actions and passed source-by-source equality and SHA256. The final Actions ZIP was also inspected in the container: **38 exact historical PNG source blobs, 44 size/hash-verified compressed archives, 4 complete original source atlas images**.

## Official final results — all sizes include metadata/index/headers

| Cohort | Actual edited PNG pairs | Best new reversible delta | Bytes | Best of all 7 prior methods | Bytes | New delta wins? |
|---|---:|---|---:|---|---:|---|
| All three commits pooled | 19 | xor_webp | 27,215 | **tile16_webp** | **26,492** | **No** (+2.73% larger) |
| Pan/navigation edits | 13 | xor_webp | 19,119 | **tile16_webp** | **18,188** | **No** (+5.12%) |
| Curve and 3D tools | 4 | xor_webp | 6,361 | **atlas_webp** | **6,329** | **No** (+0.51%) |
| Isometric tool edits | 2 | xor_webp | 4,079 | **atlas_webp** | **3,979** | **No** (+2.51%) |

**Precommitted gate:** the new delta format must beat best prior format on pooled set AND ≥2/3 individual commit groups. **Result 0/4 tested comparisons, 0/3 commit-group wins: FAILED**. The simple file-version XOR/modular residual does not provide a measured advantage over the strongest honest alternatives here.

### A very small positive *pre-existing BLKH indexed tile* observation

Compared **only against the strongest standard non-dictionary reference, WebP lossless whole atlas**, the already existing `tile16_webp` dictionary performs:

| Cohort | Indexed 16px dictionary | Full lossless WebP atlas | Dictionary savings |
|---|---:|---:|---:|
| All 19 historical pairs | **26,492 B** | 26,591 B | **99 B (0.372%)** |
| 13 pan/navigation pairs | **18,188 B** | 18,527 B | **339 B (1.83%)** |
| 4 curve/3D pairs | 6,388 B | **6,329 B** | **−59 B (dictionary larger)** |
| 2 isometric pairs | 4,032 B | **3,979 B** | **−53 B (dictionary larger)** |

Thus an actual historical multi-asset source cohort **does** show a narrowly reproducible indexed-dictionary advantage over a full WebP atlas. However **only 1/3 individual commit cohorts wins**, and both positive differences are tiny and susceptible to file-format/manifest/packing choices. Because the pooled set contains the pan cohort, it is **not** a second independent replication.

A separate analysis of the original decoded PNGs showed:
- Pan: **13/13** same-path PNG pairs changed pixels; approximately **35.05% of corresponding spatial pixel locations** changed in padded comparisons, and **0/52** aligned 16×16 blocks between each old/new image remained identical in the same position.
- Curve/3D: **4/4** files changed; about 12.91% of pixel locations changed, 4/16 corresponding aligned 16px blocks remained identical.
- Isometric: **2/2** files changed; one pair also changed physical pixel dimensions; about 42.77% of padded pixel locations changed, 0/8 corresponding aligned blocks matched.

Therefore, the limited pan/cohort saving is plausibly driven by **reuse among related theme icons and padded/empty tile structures across assets**, rather than by large unchanged old→new tile patches at the same coordinate. A follow-up should measure source-independent cross-asset dictionary contribution separately from chronological-revision persistence, but must not reinterpret the current tiny effect as strong proof.

## Reproducibility and limitations

- Three **actual** commit histories in **one** MIT-licensed project, no independent game creators or representative industry sample.
- New delta codec is distinct from the previous SHA256 dictionary and loses all tested groups.
- Small file-size effects do not establish encoder/decoder speed superiority, modern video/sprite compression leadership, a new compression principle, patent novelty or broad neural-learning gain.
- No native Git packfile incremental compression, rsync/binary delta archives, xdelta3, Zstandard dictionary training on old files, AV1 intra or screen-content standards were measured. Source original PNG bytes+ZSTD9 and strong spatial WebP are included, but do **not** exhaust all competing versioning technologies.
- File formats count the common complete historical source metadata for every arm, so the absolute byte differences are small and may vary with changed metadata conventions. No test of native original PNG ancillary metadata bit-exact preservation for custom methods; original source PNG bytes remain an available baseline.

## Decision: no public release yet

**Negative for proposed new XOR/modulo version compression**, with a limited **99 B/339 B existing dictionary** win for related authored icon families. The completed test is a significant scientific-methodology advancement because it uses real commit-preserved, decoded, image-modifying source changes and honest complete-file baselines. However the result is neither broad nor robust enough for a BLKH worldwide breakthrough claim. Keep research PR #1 **draft**, do not modify `main`, public demo or LinkedIn marketing claims.

A meaningful next study would sample **independent historical revisions from additional game/editor artists**, explicitly compare native version-delta tools such as xdelta3 or Zstd trained dictionaries, and precommit the cross-asset vs temporal-persistence hypotheses separately.
