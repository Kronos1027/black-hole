# BLKH Exp83P — authentic Pixelorama Git PNG edits: source-verified negative result

**Status: official GitHub Actions SUCCESS**, including separate source re-download, Git parent and diff-status verification, all historical old/new PNG source-byte hashes, and complete independent reconstruction of **44** saved lossless RGBA archives, 11 codecs across 4 source groups. **Scientific result: 0/4 groups beat the strongest matched lossless reference.**

- Protocol (before pixel measurements): [BHUH_EXP83_PIXELORAMA_HISTORICAL_PROTOCOL.md](BHUH_EXP83_PIXELORAMA_HISTORICAL_PROTOCOL.md).
- Code: `research/dedup/experiment_83_pixelorama_git_history.py`.
- CI: `.github/workflows/exp83-pixelorama-history.yml`, [official successful run](https://github.com/Kronos1027/black-hole/actions/runs/40797209864).
- Reproducibility artifact: `exp83p-pixelorama-historical-19-png-pairs` with `results.json`, **38 actual historical source PNG bytes**, all 44 complete compressed files and reconstructed atlas PNGs, full source checksums and per-group results.

## Independent application/source selection

First real Git-authored sprite **revision** dataset examined in this track, from open-source Pixelorama, `Orama-Interactive/Pixelorama`, repository MIT license. All 19 selected PNG paths were independently confirmed as Git `status=modified` rather than just added/renamed. All **19/19 pairs had genuinely changed decoded RGBA pixels**, not merely changed source PNG encoding bytes.

The protocol froze three commit SHAs and exactly 13/4/2 same-path icon before/after pairs *before fetching their pixel data*. It captured each pinned historical first-parent commit, original old/new PNG file SHA256, decoded RGBA SHA256, dimensions and both versions. Related assets in one source repository across three commits are correlated observations: NOT 19 independent creators, not a random representative sample of all asset workflows. Art files reside in the MIT-licensed project; source attribution retained.

## Frozen comparison (all complete bytes)

Four source-aware reversible XOR/modulo pixel-RGBA temporal differences, atlas encoded as WebP lossless `exact=True` or Zstandard level 9. Both historical old and new sprites are stored in the same file; when dimensions differ, the old image is treated as zero outside its bounds, with all new source values recovered exactly. Seven baseline methods: whole WebP and PNG lossless RGBA atlas, full atlas RGBA raw+Zstd, **original exact PNG source bytes** concatenated and Zstd, optimized individual PNGs+Zstd, exact 16×16 RGBA block dictionary + WebP/Zstd. Metadata and source hashes counted on *every* arm.

| Group (paired historical file count) | Smallest source-aware differential archive | Bytes | Smallest baseline | Bytes | Result |
|---|---|---:|---|---:|---|
| All 19 actual edited pairs | xor_webp | 27,215 | **tile16_webp** | **26,492** | custom **+2.73%** bytes |
| Pan/cursor commit (13 pairs) | xor_webp | 19,119 | **tile16_webp** | **18,188** | custom **+5.12%** bytes |
| Curve/3D tool commit (4 pairs) | xor_webp | 6,361 | **atlas_webp** | **6,329** | custom **+0.51%** bytes |
| Isometric tool commit (2 pairs) | xor_webp | 4,079 | **atlas_webp** | **3,979** | custom **+2.51%** bytes |

**0/4** win against all seven fixed references. The preregistered criterion — full bundle win and at least **2 of 3 individual commits** winning — was decisively **NOT met**. No claim of a competitive general source-version compression codec.

## Strong limitations and interpretation

These actual editor UI sprite revisions are small and three commits from one repository. Change extent, alpha sparsity and optimized output sizes may vary in other asset-version histories, but no unmeasured improvement can be assumed. Standard WebP spatial coding and explicit exact-tile dictionaries already represent similarity effectively; storing a separate XOR image does not automatically reduce complete file size. The best baseline itself changed by group, proving that results should not be described as merely a comparison against isolated PNGs.

The tested archive target is **exact decoded straight RGBA pixels**, including RGB hidden under alpha=0; source byte identity is separately used as a baseline and recorded, not automatically reproduced by WebP. No claims on original PNG metadata preservation, Git packfiles/bsdiff performance, production memory/cold-cache speed, independently authored version histories beyond this repository or neural compression.

**Next decision:** keep this fully negative result. Further experimentation on this same data is only *exploratory*. A meaningful new hypothesis would be sparse changed-pixel patch coding with complete old-source cost and an explicit conventional binary delta baseline; or expand to independent genuine authored Git repositories before claims. Do not tune transform parameters after seeing compression outcome or call partial within-sample wins a scientific breakthrough.

**Main/demo and broad public breakthrough claims remain HOLD; PR #1 stays draft.**
