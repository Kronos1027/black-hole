# BLKH Exp79 — actual Kenney authored colorways (NEGATIVE)

**Status:** GitHub Actions PASSED and separately verified all 21 saved archives. This is an exploratory follow-up prompted by Exp78, **not independent holdout validation**: all image variants come from the same already inspected Kenney UI Pack.

- Pre-committed protocol: [BHUH_EXP79_REAL_COLORWAYS_PROTOCOL.md](BHUH_EXP79_REAL_COLORWAYS_PROTOCOL.md).
- Successful official verification: https://github.com/Kronos1027/black-hole/actions/runs/40707860237
- Code and CI: `research/dedup/experiment_79_authored_colorways.py`, `.github/workflows/exp79.yml`.
- Exact original ZIP fixed at SHA256 `a8a14a234911eb648c062622915c93e79e94e97cb7f9f375a70f6617f1174318`; source Kenney UI Pack, CC0, mirrored on OpenGameArt.
- 8/16/24 lexicographically selected real original **same-shape four-color variants** (Blue, Green, Red, Yellow), totaling nested 32, 64, 96 RGBA PNGs. No synthetic recoloring or chronological file revisions.

**Actual complete-file bytes (including full metadata/source hashes and all index overhead):**

| Number of actual colorway sprites | Best indexed 16px dictionary / bytes | Best conventional lossless reference / bytes | Relative BLKH overhead |
|---|---|---|---:|
| 32 (8 shapes ×4) | tile16_webp / **19,960** | original_png_zstd / **19,346** | +3.17% |
| 64 (16 shapes ×4) | tile16_webp / **41,031** | original_png_zstd / **39,322** | +4.35% |
| 96 (24 shapes ×4) | tile16_webp / **63,512** | atlas_webp / **58,839** | +7.94% |

**0/3** subsets passed the prespecified size criterion. The CC0 pack's actual authored color variants share forms, but explicit 16×16 hashing/ID overhead outweighs the best packed source/atlas strategy on these nested examples. Each archive was decoded and its original PNG source pixels **including hidden RGB under alpha=0** and alpha were checked exactly by SHA256. These 3 nested subsets are strongly correlated, not 3 independent artists.

**Honest limits:** no true authored version-control progression was tested; two best references preserved original PNG bytes with zstd and one was an exact RGBA WebP atlas. No evidence of a new compressor or neural model. Actual unknown sprite workflow may contain different cross-version reuse.

**Next falsifiable test:** the same frozen 4-color shape groups could be compared under a **different reversible representation**, e.g. per-pixel differences relative to each shape's Blue RGBA sprite, with source-shape geometry identical; encode all actual bytes plus atlas/metadata using zstd/WebP. Precommit predictor and sample sizes *before reading compression outcomes*; compare against ALL already validated strong Exp79 formats. A negative outcome should be published without retuning. This is an engineering representation change and remains within one already seen pack.

**Public action:** keep main/demo unchanged and the research PR draft. No LinkedIn claim of universal compression breakthrough.