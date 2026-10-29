# BLKH Exp85 — independently authored real Mindustry sprite history (MIXED: 1/3 commit groups win, aggregate loses)

**Status:** ✅ official GitHub Actions SUCCESS, including independent re-fetch of original sources and commit-diff metadata, decode+SHA256 verification of **all 56 complete binary archives**. **Only a 429-byte (0.97%) within-group improvement** on one natural 15-pair historical texture-cleanup commit; the complete 30-pair archive does **not** beat its strongest reference, so **the precommitted independent-project transfer criterion FAILED**. No general superiority or novelty claim.

- Frozen before inspection of original sprites or codec sizes: [BHUH_EXP85_MINDUSTRY_GIT_SPRITES_PROTOCOL.md](BHUH_EXP85_MINDUSTRY_GIT_SPRITES_PROTOCOL.md).
- Code `research/dedup/experiment_85_mindustry_history_holdout.py`; workflow `.github/workflows/exp85.yml`.
- [Official successful source-backed independent audit](https://github.com/Kronos1027/black-hole/actions/runs/40876173234).
- Original artifact `exp85-genuine-mindustry-source-edits-vs-bsdiff` includes `results.json`, **60 actual historical parent/child PNG files**, all 56 complete compressed files, original-art attribution and original project **GPLv3 license** text. Original PNGs are *not committed to the BLKH Git repository*.

## New independent project provenance and exact pinned sample

Different project and creators from Pixelorama Exp83P–84: original open-source game **Mindustry** by Anuken/contributors. Repository https://github.com/Anuken/Mindustry, root LICENSE GPL-3.0, retaining upstream links and file/commit provenance. Do not infer that every artwork file has no independent third-party rights or exceptions.

Three pinned commits, each compared against its exact parent SHA in the original repository. Every source path was in real Git diff `status=modified`. All selected **30/30** old vs new PNG pairs had genuinely different decoded RGBA pixel arrays, not merely a changed PNG encoding. The source PNG bytes and fully decoded RGBA arrays have independent per-image SHA256. All source paths were locked in the preregistration before pixel inspection or compression.

- **defense (2 pairs)** `bae93299c7721a35431daa33e3af8b1f6c596c8d`
- **texture_cleanup (15 pairs)** `916a00b6ff8f1bc16427ef430df9223b3855d908`
- **walls (13 pairs)** `0aaa77a5eae7c3e3474bc869d6b622253f729575`

These are three correlated commits from **one new repository**. The 30 pairs are not 30 independent artist-project histories, and commit group sizes vary.

## Unchanged codec evaluation from Exp84

No new predictor optimized based on Mindustry source content. Reused the **same existing 14 complete self-contained encodings and exact decoders**:
- `sparse_webpold_zstd`, `sparse_pngold_zstd` — exact one-bit masks for changed source RGBA pixels, four replacement RGBA bytes per changed pixel, old image encoded in lossless WebP atlas or exactly original historical PNG byte archive, changed-mask/color stream compressed with Zstandard9;
- **12 references**, including four previous XOR/modulo RGB(A) difference schemes stored in WebP or ZSTD, whole-lossless RGBA WebP/PNG atlas, raw RGBA+ZSTD atlas, byte-exact original PNGs+ZSTD, optimized PNGs+ZSTD, 16px SHA256 exact-tile dictionaries+WebP/ZSTD and actual `bsdiff4` old→new binary PNG patch archive (reconstructs both historical PNG file byte streams exactly).

Same shelf packing max 1024px, 16px tile alignment, full historical file and image hashes/paths, image dimensions, full archive metadata, source provenance and indexed references are included in every arm. The target image fidelity is **bit-exact decoded RGBA including hidden transparent RGB**, while the bsdiff and original-source-PNG baseline preserve the original PNG file bytes themselves.

## Official full-file compressed results (not partial payload estimates)

| Natural historical group | Truly changed pixel positions / new-source positions | Best indexed sparse RGBA source-patch (bytes) | Best of twelve reference formats (bytes) | Candidate overhead/savings |
|---|---:|---:|---:|---:|
| **All 30 real pairs** | 7,520/259,840 (**2.894%**) | 85,390 (`sparse_webpold_zstd`) | **85,355** (`xor_webp`) | **+35 B**, +0.041% (LOSE) |
| defense, 2 pairs | 936/32,768 (**2.856%**) | 8,146 (`sparse_webpold_zstd`) | **7,830** (`xor_webp`) | **+316 B**, +4.036% (LOSE) |
| **texture_cleanup, 15 pairs** | 1,198/161,536 (**0.742%**) | **43,995** (`sparse_webpold_zstd`) | 44,424 (`xor_webp`) | **−429 B**, −0.966% (**WIN**) |
| walls, 13 pairs | 5,386/65,536 (**8.218%**) | 35,791 (`sparse_webpold_zstd`) | **34,864** (`atlas_webp`) | **+927 B**, +2.659% (LOSE) |

**Precommitted success gate**: the whole 30-pair archive must beat all twelve reference formats **AND** the sparse method must beat all twelve in at least two of the three natural commit groups. Actual **aggregate loss, 1/3 group win → GATE FAILED**.

The best 15-pair sparse patch is only 429 bytes smaller, and those source edits affect fewer than 1% of pixel positions. This is a plausible mechanism, not a causal proof: geometry, alpha semantics, WebP encoding, metadata and the exact original images also affect file size. Other groups with 2.86% and 8.22% changed pixels lose to stronger references. This is an **independent project test** from Pixelorama, but not a statistically powered population sample, and the small win can easily change in a new collection.

## Technical verification

Official CI first tested the unchanged Exp83P/Exp84 codecs, then downloaded the 60 original historical PNG source versions, captured source exact hash/dimensions and Git first-parent references, and saved 56 actual full archives. A **separate verification step** independently queried source Git commit diffs, re-downloaded each old/new PNG byte stream, compared original source SHA256 and RGBA pixels, opened every saved compressed archive, and reconstructed all original pixels/hashes, including alpha and hidden RGB. Actual original Mindustry LICENSE text and full attribution were bundled in the Actions artifact, not copied into main.

No end-to-end decoder speed/memory comparison, no Git object pack delta comparison or learned entropy model, and no independent human visual assessment is necessary for exact-lossless pixel equivalence. Original PNG metadata preservation applies to bsdiff/source PNG baselines only, not to the BLKH RGBA pixel codec.

**Scientific conclusion:** sparse edit representations are promising **only in sufficiently localized real source-edit regimes**, and even this Mindustry test yields just a 0.97% within-group win while failing its independently preregistered aggregate standard. This is meaningful falsifiable engineering research, **not** a compression breakthrough or a generalized dominance claim.

**Release decision:** PR #1 remains draft; `main`, demo, and broad public/LinkedIn breakthrough publication remain unchanged. Further claims require multiple new independent project histories and stronger source archive/Git packfile alternatives without tuning on this already observed corpus.
