# BLKH Experiment 72 — four independent real RGB photos, neural conditional codec vs conventional color atlases

**Status: GitHub Actions complete / passed. Scientific finding: NEGATIVE for practical rate–distortion.** All 12 independently decoded shared nonlinear RGB models were larger than the smallest actual conventional encoded *spatial atlas* meeting or exceeding BOTH their decoded RGB PSNR and mean RGB tile SSIM. Zero models qualified for the preregistered quality/useful-size gate.

**Precommitted protocol:** [BHUH_EXP72_PROTOCOL.md](BHUH_EXP72_PROTOCOL.md), committed **before Exp72 source-corpus measurements**.  
**Successful reproducible workflow:** https://github.com/Kronos1027/black-hole/actions/runs/40622623178  
**Code:** `research/conditional/experiment_72_real_photo_families.py` and `.github/workflows/exp72.yml`.  
**Complete evidence artifact:** `exp72-real-public-photo-neural-vs-atlas`, with actually encoded `.bl72` model and conventional image files, source RGB crops, independent decoded scores, every SHA256 and manifest. This artifact is for scientific reproducibility, not a product release.

## New real source photographs, clearly defined independence

Sources from scikit-image v0.25.2's standard image data; licensing and provenance: [official scikit-image sample image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html).

- `astronaut`: NASA astronaut photograph, public domain.
- `coffee`: cup photo, CC0 by Rachel Michetti.
- `chelsea`: cat photo, CC0 by Stefan van der Walt.
- `rocket`: public-domain rocket-launch photograph; predeclared as external-source confirmation relative to `astronaut/coffee/chelsea`.

For each source, use the predeclared central **128×128 RGB original-pixel crop** without resize, split as **16 adjacent nonoverlapping 32×32 tiles**. Save the identical original layout as a single continuous **128×128 RGB atlas** for every baseline. Thus there are **four independent source photographs** and not 64 independent visual assets. The model is trained and measured on the **same target photo** to encode that source (legitimate within-corpus compression, not prediction/generalization across unseen photos). Original full source and actual crop SHA256 are included in every output's manifest.

## Model and fair strong baselines

One true nonlinear shared coordinate decoder per source photo, with separate embedding/latent for each of 16 tiles, two sine-activated hidden layers, 3-channel sigmoid output. Fixed architecture variants: `hidden32 latent8`, `hidden64 latent8`, `hidden96 latent12`; 1,500 Adam steps, lr .003, deterministic seed 720072, 2048 samples per step. **Every shared weight, bias, and all 16 latent vectors** are serialized as float16 and zlib-compressed, with the same counted source manifest used by conventional baselines. Scores are computed **only by independently decoding the actual stored bytes in NumPy**, not the FP32 PyTorch training network.

The 22 baseline encodings per source are PNG lossless, WebP lossless, reversible JPEG2000, **13 WebP lossy quality settings**, and **6 JPEG lossy quality settings**. All 88 baseline files are actually serialized, decoded and checked. The best comparator for each network is **the smallest valid conventional saved file with no worse global RGB MSE/PSNR and no worse mean per-tile RGB SSIM**, not merely PNG or the best quality-independent file. The lossy-atlas baselines **preserve the original spatial layout**, avoiding any penalty from artificially breaking their coherent source image into isolated blocks.

All recorded source hashes, source-photo 128×128 crops, complete encoded file sizes/hashes, mean RGB SSIM, PSNR/MSE, network configuration and selected best conventional codec were **independently checked by GitHub Actions against saved outputs**.

## Official measurements from saved-file decoding

All file sizes below **include source manifest/header bytes**, and the quality numbers are measured after float16 parameter quantization and zlib decompression. No model passed the **frozen usable-quality gate** PSNR ≥30 dB AND mean RGB SSIM ≥0.90.

| Source | Shared network hidden/latent | Model bytes | Best matching conventional RGB atlas bytes | Neural PSNR dB | Mean RGB tile SSIM | Smaller? |
|---|---|---:|---:|---:|---:|---|
| astronaut | 32/8 | 3,530 | **996** (WebP q1) | 23.974 | 0.6804 | No |
| astronaut | 64/8 | 10,090 | **1,136** (WebP q5) | 25.993 | 0.7626 | No |
| astronaut | 96/12 | 21,295 | **1,262** (WebP q10) | 26.640 | 0.7903 | No |
| coffee | 32/8 | 3,535 | **855** (WebP q1) | 26.948 | 0.7683 | No |
| coffee | 64/8 | 10,088 | **967** (WebP q5) | 28.585 | 0.8129 | No |
| coffee | 96/12 | 21,295 | **1,033** (WebP q10) | 29.117 | 0.8192 | No |
| chelsea | 32/8 | 3,511 | **927** (WebP q5) | 27.271 | 0.6559 | No |
| chelsea | 64/8 | 10,065 | **1,337** (WebP q20) | 28.590 | 0.7388 | No |
| chelsea | 96/12 | 21,291 | **1,601** (WebP q30) | 29.243 | 0.7725 | No |
| rocket | 32/8 | 3,532 | **704** (WebP q5) | 28.440 | 0.8363 | No |
| rocket | 64/8 | 10,063 | **840** (WebP q20) | 29.584 | 0.8567 | No |
| rocket | 96/12 | 21,333 | **840** (WebP q20) | 30.102 | 0.8603 | No |

**Precommitted quality-and-size outcomes: 0/12 size wins, 0/12 useful-quality models, 0/12 release candidates.** The `rocket` h96/z12 model exceeded 30 dB PSNR but its mean RGB SSIM remained 0.8603 (<0.90), and the resulting ~21 KB model was about **25.4×** larger than its 840-byte matched WebP atlas.

For the cheapest neural configuration, the model used 3.5–5.0× as many bytes as the matched WebP atlas despite relatively low quality. Expanding from 32 to 96 hidden units increased neural bytes roughly sixfold but gained only 1.7–2.7 dB across these small natural image crops. The dominant factor is the overhead of shared weights relative to the compact 128×128 WebP representation; this architecture did not amortize those bytes at this corpus size.

## Methodology caveats

- Although four sources are distinct, the 16 tiles within a source share spatial content, and the models are trained to encode the same photo being evaluated. This is **not** a four-way generalization test of frozen weights transferring to a new photo; the hyperparameter settings were frozen, and each photo was independently trained.
- RGB PSNR, mean tile SSIM and a threshold do not guarantee perceptual utility or color-management correctness in every downstream display; no independent human visual study. The JPEG/JPEG2000/WebP features are hardware/library dependent, and the benchmark is small, CPU-trained, and not representative of all photo distributions.
- WebP/JPEG image decoder implementation size was excluded for all codecs, just as PyTorch training tooling is excluded. The neural NumPy decoder is stored as algorithmic code rather than bundled in each output; codec software footprint would matter for product deployment.
- Four fixed original photographs are too small to claim a global benchmark or statistical significance. No independent dataset expansion, entropy bottleneck, better learned prior, perceptual-weighted training, established CompressAI framework or optimized neural bitstream was evaluated.
- These results reject **this explicit conditional sine network on 16 adjacent photo tiles**, NOT all neural image compression. Modern learned codecs often use analysis/hyperprior/entropy models rather than one small coordinate MLP.

## Next scientific decision

Blindly increasing hidden units on the same four photos is unlikely to establish a useful competitive codec: network bytes grew much faster than visual quality. The next experiment must be **new architecture or new application affordance**, with trained and compressed weights honestly counted, strong whole-image/atlas baselines and a fixed high-fidelity threshold. Possible falsifiable approaches: learned **entropy-constrained latent coding** with real quantized bitstream, or a use case requiring multi-resolution continuous synthesis/editability that conventional raster codecs do not offer; both must be compared with vector or multi-resolution standard baselines. Any scientific claim should be conditioned on independently sourced test families and exact retrieved outputs.

**Demo, `main`, LinkedIn breakthrough claim: HOLD.** The draft research PR remains the sole destination.
