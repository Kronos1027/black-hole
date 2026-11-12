# BLKH Experiment 82 — source-qualification audit on genuinely historically edited OpenGFX2 PNG sprites

**Audit-only preregistration before inspecting candidate PNG pixel differences or size-compression results.** Exp81 yielded zero qualified same-path sprite revisions in its fixed NinjaAdventure history; Exp82 DOES NOT retrospectively repair that failed experiment, or run compression.

## Immutable real project and terms

OpenTTD/OpenGFX2, https://github.com/OpenTTD/OpenGFX2, game graphics under **GPL-2.0** per https://www.openttd.org/content-creators and upstream repository `LICENSE`. Creator/project identification and original asset Git SHAs must be recorded, no source images committed to BLKH. Freeze source main commit SHA **`18f21b51a19df0f36635917e03aa84eb6e10e5b7`**, traverse exactly the first **120 first-parent commits**, in newest-first order; do not substitute moving HEAD. If fewer available, fail.

## Frozen source selection, audit only

For each of the 119 consecutive parent→child transitions, extract `git diff --diff-filter=M --name-only <parent> <child> -- '*.png'`. Consider only paths under `graphics/`, excluding `.import` files, generated atlas previews and renames (same-path modified only). Decode actual PNG Git blobs from each commit. Record all candidates, rejection reasons, parent/child SHA, path, original PNG SHA256, original decoded RGBA uint8 SHA256, dimensions, pixel-channel mismatch count, exact alpha channels and final dimension-change status. **No PNG output resizing or premultiplication**; preservation of RGB under zero alpha is mandatory in future compression but only audited here.

Qualification groups are deliberately broader than Exp81: **equal before/after dimensions between 16 and 1024 pixels in either axis**, source byte length ≤12 MB, and original RGBA arrays really differ (not only PNG metadata). For each qualify, record actual dimensions and estimated pair RGBA bytes. No cherry-picking or result-guided filtering. For a future coding study, deterministic order is transition index then exact path, with first 24 eligible pairs, and nested counts 4,8,16,24 IF actually present.

## Decision and limitations

This audit does not benchmark compression or show any BLKH superiority. If ≥4 qualified pairs under planned per-asset bound and viable aggregate dimensions are available, a **separate frozen Exp83 protocol** can use them in full-file exact RGBA comparison against WebP and original PNG+Zstd. If not, report a source limitation and find another independent project instead. There must be no synthetic frame generation or source replacement inside this audit.

**No main/demo modifications or breakthrough announcements.**
