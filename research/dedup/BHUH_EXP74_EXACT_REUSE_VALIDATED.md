# BLKH Experiment 74 — VERIFIED specialized cross-frame exact-block deduplication

**Date:** 2026-10-28 UTC. **Status:** ✅ Official GitHub Actions passed all 140 full-file decodes, 20 scenarios, seven codecs each. **First strong *narrow engineering win* in this branch against genuine lossless WebP animation and APNG baselines under deterministic tile-reuse workloads.** This is NOT a novel compression theorem, a neural result, or a win over naturally captured video.

- [Precommitted protocol](BHUH_EXP74_PROTOCOL.md) (before results).
- [Official successful GitHub Actions](https://github.com/Kronos1027/black-hole/actions/runs/40697115889).
- Full implementation / CI: `research/dedup/experiment_74_cross_frame_exact_blocks.py`, `.github/workflows/exp74.yml`.
- Reproducible `exp74-true-lossless-animation-vs-crossframe-dedup` Actions artifact, with `results.json`, all 140 actual encoded archives including their byte/manifest SHA256, image and source hashes, and exact decoded verification in CI.

## Dataset and honest scope

Use 5 independent real *source photographs* previously examined in Exp73: astronaut, coffee, chelsea (cat), hubble_deep_field and immunohistochemistry. Each source contributes one **simulated sequence of 8 frames at each edit intensity**. Start from a central original 128×128 RGB crop, then at every transition replace **0, 2, 8 or 32** unique 16×16 tile positions with real photographic tile crops sampled (with replacement) from the *next source photo* in the fixed cycle. Deterministic seeds were precommitted. The 20 scenarios are strongly correlated: not twenty independent video recordings, and not natural temporal observations. Block duplication is built into the simulation; a maximum of 128 distinct tiles is possible because the pool contains only 64 original source +64 donor tiles, even when 32 tile positions change at each transition. This is an intentionally favorable workload for exact dedup, *clearly disclosed*.

All 8×128×128 RGB×3=393,216 uncompressed color bytes per case were exactly reconstructed. Each saved archive contains a manifest with source+donor original image SHA256, 8 frame SHA256, 100-ms timing, tile/sequence structure and source edit rule. Conventional formats received the same complete manifest.

Custom methods:
- **raw_zlib**: exact SHA256 dictionary of unique 16×16 RGB tiles plus 512 uint16 references and one zlib9-compressed dictionary payload;
- **webp_dictionary**: same exact dictionary and reference table, but pack unique tiles into one lossless WebP atlas. All dictionary atlas, reference IDs, headers and manifest bytes count. This method was the smaller custom variant in all reported scenarios.

Strong standard alternatives, every one actually encoded and decoded for exact RGB and frame ordering: lossless animated WebP, APNG, full 4×2 frame atlas WebP lossless, atlas PNG lossless, and raw eight-frame zlib. Native animation frame durations were verified from actual WebP ANMF RIFF chunks; static animations may be coalesced by the native encoder only when the counted manifest verifies all eight source frame hashes and durations.

## Official complete byte-accounted per-case results

Here, reference is the **smallest of all five valid non-dedup alternatives**, not individually saved PNG files. Eight frames per scenario, 128² RGB each, lossless.

| Source | Changed 16² tiles / transition | Dictionary + WebP bytes | Best standard bytes | Best standard codec | Relative byte savings |
|---|---:|---:|---:|---|---:|
| astronaut | 0 | 27,403 | **26,036** | animated WebP | -5.25% |
| astronaut | 2 | **32,785** | 33,382 | animated WebP | 1.79% |
| astronaut | 8 | **39,643** | 50,230 | animated WebP | 21.08% |
| astronaut | 32 | **50,102** | 108,731 | animated WebP | 53.93% |
| coffee | 0 | 24,763 | **23,520** | animated WebP | -5.28% |
| coffee | 2 | **30,581** | 31,112 | animated WebP | 1.71% |
| coffee | 8 | **38,223** | 47,276 | animated WebP | 19.15% |
| coffee | 32 | **49,486** | 110,331 | animated WebP | 55.15% |
| chelsea | 0 | 26,948 | **25,607** | animated WebP | -5.24% |
| chelsea | 2 | **32,244** | 32,799 | animated WebP | 1.69% |
| chelsea | 8 | **42,378** | 50,879 | animated WebP | 16.71% |
| chelsea | 32 | **52,443** | 111,700 | animated WebP | 53.05% |
| hubble_deep_field | 0 | 28,133 | **26,834** | animated WebP | -4.84% |
| hubble_deep_field | 2 | **32,407** | 33,256 | animated WebP | 2.55% |
| hubble_deep_field | 8 | **41,195** | 48,012 | animated WebP | 14.20% |
| hubble_deep_field | 32 | **49,662** | 99,303 | **whole WebP atlas** | 49.99% |
| immunohistochemistry | 0 | 23,335 | **21,846** | animated WebP | -6.82% |
| immunohistochemistry | 2 | **26,685** | 28,034 | animated WebP | 4.81% |
| immunohistochemistry | 8 | **39,363** | 48,010 | animated WebP | 18.01% |
| immunohistochemistry | 32 | **48,490** | 111,677 | animated WebP | 56.58% |

**Win counts by change intensity:** 0 changed tiles **0/5 wins**, 2 changed **5/5**, 8 changed **5/5**, 32 changed **5/5**. The predefined `≥4/5` limited engineering criterion was met in each nonzero intensity. **15/20** overall, but don't present 20 observations as independent source video samples.

This result is strongly dependent on **reuse from a fixed 64-tile donor pool**. At 32 changed positions, the unique dictionary remains only about **126–127 tiles** across all eight frames, far below 512 total tile positions. Under actual continuously moving or noisily captured video, exact tile reuse may be nearly absent and this scheme may lose badly. Storing no source photons and reusing exact patches is not a model of real motion.

## Scientific limitations and safeguards

1. **Not genuine animation footage**: each sequence is a controlled composite from real photos; frame edits deliberately produce identical tiles across time. No temporal noise, transforms, subpixel shifts, real animation codec optimization beyond saved native WebP, nor motion vectors.
2. **Known technique**: content-addressed block dictionaries, WebP lossless and frame-index arrays are established ideas. No new information-theoretic method or neural compression is established.
3. **Few independent source scenes**: only five original photos, each reused in four strongly correlated scenarios. 140 encoded files are *verification artifacts*, not 140 independent examples.
4. **Comparators**: animated WebP/APNG and joint image atlases included; no FFV1, AV1/HEVC lossless, FLIF/JXL animation, palette-specific sprite codec, general dedup package (tar+zstd), optimized screen-content codecs, image sequence formats or platform cache. A claim over all animation compression is unjustified.
5. **Actual bytes**: full dictionary and 1024-byte index per file are counted. The source SHA256 and duration metadata are counted across every scheme. Decoder software and execution runtime are not part of file bytes for any method.
6. **Timing**: encode/decode elapsed times are retained in `results.json` but not a controlled dedicated hardware study; no speed advantage is asserted.
7. **APNG/WebP robustness**: a compatibility issue with Pillow v13's duration-info exposure was corrected by parsing actual WebP ANMF duration bytes and preserving collapsed all-identical sequences only when all eight manifest checksums match. All 140 archives then independently validated in CI. Original failed runs remain in Actions history for audit.

## Engineering decision

**This is a meaningful positive engineering result in a precisely restricted reuse domain**; it is the strongest size advantage we have found so far under the project's intended exact-block-sharing mechanism. It is suitable to develop a *prototype for repeated sprite cells/UI states/asset variations*, but **not yet** to update the public demo with global compression leadership or publish a LinkedIn breakthrough claim. The release gate needs genuine time-varying frames and stronger motion/lossless-video baselines.

**Next falsifiable experiment:** reuse this **unchanged algorithm and block size**, then benchmark real microscopic temporal sequences from publicly available `skimage.data.protein_transport()` and `skimage.data.nickel_solidification()` (do not artificially replace tiles), with deterministic published bit-depth conversion where required to satisfy current 8-bit RGB prototype. A failure must be published as clearly as the success here. Keep `main` untouched and PR #1 draft.
