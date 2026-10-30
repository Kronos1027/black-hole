# BLKH Experiment 73 — cross-source learned residual probabilities and rANS (NEGATIVE)

**Date:** 2026-10-28 UTC. **Status:** ✅ official GitHub Actions complete, all independently decoded full archives and RGB source hashes checked. **No lossless size advantage over WebP**, despite truly trained and correctly serialized probability tables. This is *not a neural model*: a fitted histogram is a learned probability table, and the entropy coder is standard rANS.

**Preregistered before the new source measurements:** [BHUH_EXP73_PROTOCOL.md](BHUH_EXP73_PROTOCOL.md).  
**Successful reproducible run:** https://github.com/Kronos1027/black-hole/actions/runs/40693576965  
**Code and CI:** `research/conditional/experiment_73_cross_source_rans.py`, `.github/workflows/exp73.yml`.  
**Evidence:** the official `exp73-learned-static-context-rans-lossless` artifact contains `results.json`, the actual 158,798-byte rANS complete archive, the actual 157,725-byte residual-zlib complete archive, 18 conventional saved image archives and the five source crops. All raw saved bytes and decoded RGB crops were rechecked with SHA-256 by an independent CI step.

## Scientific question and method

Exp72 found that a shared coordinate/sine neural model on four independent small RGB source photos was much larger than a quality-matched WebP atlas. Exp73 tests a **different mechanism**: deterministic lossless Paeth spatial prediction followed by a **static cross-source entropy probability model** and standard byte-renormalized rANS.

- Original natural source images: `astronaut`, `coffee`, `chelsea` (previously used, fit **only** the probability tables), plus `hubble_deep_field` and `immunohistochemistry` (newly selected for evaluation, **not used in the frequency training loop**).
- One deterministic center crop of **128×128 RGB** pixels per source, 5 real source photographs in total; source full-image and crop SHA-256 fixed and recorded.
- Reversible Paeth prediction (left, up, upper-left), modulo-256 residual, context = RGB channel × threshold `abs(left-up) > 24`, for **6** context tables of **256** symbols.
- Frequencies initialized with add-one smoothing, learned on only the first three original photos, normalized to 4,096 units per context. Model table serialized exactly **3,072 bytes** (six × 256 × uint16), **counted once** in the five-image bundle.
- Actual 32-bit rANS encoded source streams, each length-prefixed in a single complete archive with header and shared manifest. Independent decoder reloaded all bytes, decoded all five RGB crops and verified exact source SHA-256. The model has no uncounted source-specific learned state.
- Conventional references use the **identical complete source manifest** and the same five source crops, encoded using PNG lossless, WebP lossless and reversible JPEG2000, as *individual files* or **one spatially packed 128×640 RGB atlas**. The smaller total from these fair valid organizations was chosen for each format. They were all actually reloaded and decoded, not merely quoted as API estimated sizes.
- Ablation: exactly reversible Paeth residuals compressed with ordinary zlib level 9, including its own five-stream lengths and full source manifest.

## Official complete serialized sizes

| Method | Complete five-image encoded bytes | Difference vs smallest WebP |
|---|---:|---:|
| **WebP lossless, single RGB atlas** | **118,184** | Baseline |
| JPEG2000 lossless, single atlas | 147,737 | +25.0% |
| PNG lossless, single atlas | 153,305 | +29.7% |
| Paeth + zlib9, full bundle | 157,725 | +33.5% |
| **Trained static 6-context probabilities + rANS**, full model/bitstream bundle | **158,798** | **+34.37%** |

**WebP is strictly smaller** than the learned-rANS approach on the complete five-photo lossless corpus. Even the ordinary Paeth+zlib ablation is **1,073 bytes smaller** than the model/rANS bundle (~0.68% relative difference).

The aggregate rANS payload sizes (before the shared model, manifest and file headers, **not standalone file sizes**) were:

| Source | Role | rANS payload bytes |
|---|---|---:|
| astronaut | probability training | 30,416 |
| coffee | probability training | 26,701 |
| chelsea | probability training | 33,577 |
| hubble_deep_field | **unseen by fitted frequency tables** | 31,473 |
| immunohistochemistry | **unseen by fitted frequency tables** | 31,032 |

Total payload 153,199 bytes + 3,072-byte probability table + 2,527 bytes of remaining length/manifest/header and ancillary material = **158,798 bytes in the complete archive**.

**Losslessness:** all five decoded RGB crops exactly equal their sources; zero reconstructed channel error. A passed corruption test detects truncated entropy streams. Source hashes, model table checksum, full archive SHA-256, all baseline archive hashes and exact decoded equality were independently checked by the CI validation step.

## Timings and scope

The complete rANS bundle *including training probability counts and writing* took about **0.507 seconds** in the hosted runner, and its five-photo pure Python decoder took about **0.333 seconds**. The zlib control encoding/decoding times are in `results.json`. No comparable optimized native WebP timing study was performed, so these times **must not be used to claim faster or slower end-to-end production decoding** than other optimized codecs.

This is 5 actual source photos, not thousands of independent examples. The two newly added source images are new with respect to this research history, but the corpus-level size result includes the three probability-training sources. This is a *trained source-family static histogram*, not neural or adaptive entropy modeling. The fact that rANS is a near-entropy coder does not make the six-class predictor probabilistically optimal: richer spatial dependencies, filters, context trees, hyperpriors and color transforms can outperform it. The protocol deliberately did **not** tune the context threshold or model size after inspecting the new two sources.

## Scientific decision and next falsifiable research direction

**Exp73 is a negative result**, and an informative diagnostic: on these five small real photographs, merely replacing zlib with a trained static rANS frequency table does not beat even the zlib residual baseline, much less WebP lossless. This suggests the modeling side (predictor/spatial context/color relationship), not merely the final entropy-coding primitive, is insufficient.

Do **not** claim learned/neural compression leadership. Potential next rigorous track: evaluate a *different predictor with a real entropy model* and frozen source selection, or investigate the original BLKH block-deduplication capability over **large genuinely repeated visual assets**, where exact repeated blocks and indexes may provide a functionally useful advantage. Any next study must benchmark modern image atlas compression/deduplication baselines and account for every byte plus efficient random reads.

**Public demo, main branch and LinkedIn breakthrough publication: HOLD.** Keep all result files in the protected draft PR and distinguish research progress from demonstrated product superiority.
