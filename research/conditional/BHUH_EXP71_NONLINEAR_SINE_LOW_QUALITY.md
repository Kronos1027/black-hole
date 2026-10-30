# BLKH Experiment 71 — real nonlinear shared sine decoder with counted latents

**Status: official GitHub Actions run PASSED. Scientific conclusion: NO usefully high-fidelity compressed model was demonstrated.** A very small nonlinear shared model beat the *lowest quality* tested WebP atlas in complete bytes on four fonts, but reconstructed glyphs at unacceptable fidelity. Do not present this as a practical compression breakthrough.

- **Frozen protocol:** [BHUH_EXP71_PROTOCOL.md](BHUH_EXP71_PROTOCOL.md), written before training.
- **Official successful run:** https://github.com/Kronos1027/black-hole/actions/runs/40620600080
- **Implementation / CI:** `research/conditional/experiment_71_conditioned_sine_fonts.py`, `.github/workflows/exp71.yml`.
- **Evidence artifact:** `exp71-nonlinear-conditioned-sine-glyphs`, including original atlases, complete codeable compressed model/atlas files, SHA256, PSNR and per-glyph SSIM metrics, and source face hashes.

## Genuine nonlinear conditional decoder, not a linear SVD basis

The model uses one **shared nonlinear sine-activated network** per font face, plus one learned **per-glyph embedding** for each of 48 rendered 32×32 masks. Continuous normalized `(x,y)` inputs are concatenated with the glyph embedding, followed by two sine-activated dense layers and one sigmoid output. This is an actual conditioned coordinate network, but **not canonical SIREN initialization** and not a new architecture.

The same deterministic Adam configuration was used for all cases: 1,600 steps, 2,048 training pixel coordinates per step, `lr=0.003`, seed 710071. Shared layers, biases and **all 48 per-glyph embeddings were serialized as float16**, then zlib-compressed with a complete source manifest. A **pure NumPy decoder**, independent of PyTorch training inference, reloaded the saved file and reconstructed every pixel; the recorded quality is computed solely on these actual decoded bytes.

Four real public font faces were tested, with the same 48 glyph masks as Exp70. These faces were **already used in the previous experiment** and were trained and evaluated on the **same corpus**. This is a source-corpus compression test, not independent generalization.

Comparators were 48-glyph **joint** atlases in PNG lossless, WebP lossless, JPEG2000 lossless and WebP lossy quality 1/5/10/20/30/40/50/60/70/80/90/95/100. Per model, the minimum-sized actual decoded atlas meeting **at least the same PSNR and mean per-glyph SSIM** was selected. All sizes include common manifest and full format headers. Source and stored output SHA256, image dimensions, grayscale output, and all comparison metrics were independently verified from files in CI.

## Actual post-float16 byte-accounted results

| Font | Network size / per-image latent | Model bytes | Best equal-or-higher-fidelity WebP atlas bytes | Decoded PSNR dB | Mean glyph SSIM | Model smaller? |
|---|---|---:|---:|---:|---:|---|
| DejaVu Sans | hidden16 / latent4 | **1,562** | 2,807 | 13.46 | 0.284 | Yes, unusable quality |
| DejaVu Sans | hidden32 / latent8 | 3,862 | **2,807** | 15.10 | 0.472 | No |
| DejaVu Sans | hidden64 / latent8 | 10,300 | **2,807** | 17.80 | 0.624 | No |
| DejaVu Serif | hidden16 / latent4 | **1,567** | 3,034 | 14.27 | 0.366 | Yes, unusable quality |
| DejaVu Serif | hidden32 / latent8 | 3,874 | **3,034** | 15.66 | 0.468 | No |
| DejaVu Serif | hidden64 / latent8 | 10,347 | **3,034** | 16.38 | 0.506 | No |
| Lato Regular | hidden16 / latent4 | **1,562** | 2,767 | 13.82 | 0.302 | Yes, unusable quality |
| Lato Regular | hidden32 / latent8 | 3,878 | **2,767** | 15.46 | 0.473 | No |
| Lato Regular | hidden64 / latent8 | 10,322 | **2,767** | 17.45 | 0.582 | No |
| Liberation Sans | hidden16 / latent4 | **1,572** | 2,807 | 13.67 | 0.269 | Yes, unusable quality |
| Liberation Sans | hidden32 / latent8 | 3,880 | **2,807** | 15.28 | 0.447 | No |
| Liberation Sans | hidden64 / latent8 | 10,345 | **2,807** | 17.33 | 0.567 | No |

**Summary:** `4/12` models have fewer bytes than the best higher-quality atlas. These are **only the 16-hidden-unit models, with poor reconstructed quality**: PSNR about 13–14.3 dB; average SSIM 0.269–0.366. *None* of the 12 models meets the **preregistered usability guardrails PSNR ≥30 dB AND mean glyph SSIM ≥0.90**, and `0/12` qualify as a useful compression release candidate. The four small models total **6,263 bytes**, versus **11,415 bytes** for the comparison atlases, but this apparent 45.1% space savings comes with severe fidelity loss. More accurate networks are larger than WebP q1 while still visually inferior in absolute quality.

The model param count and training time are retained per case in `results.json`; decoding was measured only for quality in this experiment, not benchmarked as an optimized inference runtime. No statistical significance or generalization should be claimed from four correlated font-family corpora.

## Failure analysis and next action

- The linear shared-basis control in Exp70 was `0/16` at matched quality. This genuine nonlinear experiment improves compactness **only where reconstruction is too poor to be useful**.
- Increasing width to 32 and 64 helped PSNR/SSIM modestly, but the serialized float16 weights increased much faster than fidelity, under this fixed training regime.
- A font atlas packs repeated strokes and edges very efficiently. An explicit glyph-specific embedding and shared network can be a useful generative representation, but such capabilities were not established as superior to scalable vector fonts, which are an even more direct alternative for glyphs.
- Comparing quality-matched WebP q1 alone should **not** be mistaken for a robust rate-distortion frontier near 30 dB. We intentionally require high-quality criteria before entertaining an application release.
- More steps/better initialization/capacity could improve the neural model, but repeatedly tuning on the same font sets would not be independent proof. A next experiment should either register a high-quality training protocol and new actual held-out visual families or focus on **functionality beyond image size**, such as continuous-res coordinates or editable shared latent transitions, with quality/cost measured against vector or parametric baselines.

**Decision:** these tests do **not** justify updating the BLKH public demo, merging the draft research PR, or publishing claims of competitive neural compression. The result is a rigorously recorded negative outcome for this architecture/dataset/quality regime, with a narrow but unusable byte-saving observation.
