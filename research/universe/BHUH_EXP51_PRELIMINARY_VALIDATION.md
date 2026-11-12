# BHUH / BLKH Experiment 51 — preliminary independent validation (2026-10-27)

**Status: experiments 51/52 reproduced by GitHub Actions for the correlated synthetic test (2 seeds); still not a general compression breakthrough.**

## Provenance

These measurements were produced in an isolated Python environment by an **independent reimplementation** of the Exp 51 architecture and BLK51 float16 + zlib serializer, **not** by executing this repository's exact `experiment_51_fair_sharing.py` script. The reference script on this branch supports `--dataset smooth` and `--compare-codecs`. **The exact repository implementation has now reproduced the two-seed synthetic experiment on GitHub Actions.** The natural-image full-size comparison in Experiment A still refers to the independent local checker. The local replication sources, SHA-256s, raw bytes and output records are preserved separately.

All reported neural quality measurements are calculated **after decompressing, loading quantized float16 model weights and reconstructing the output**, not before quantization. Sizes are total bytes of saved payloads for the **entire six-image set**, including metadata, shared backbone, per-image latent values and indices.

## Experiment A: unrelated natural grayscale images — general-purpose result is negative

Six distinct scikit-image subjects (astronaut, camera, cell, coins, moon, page), 32 × 32 pixels, 300 epochs; layer width=16.

| Method | Seed 42: size / PSNR | Seed 123: size / PSNR |
|---|---|---|
| Independent SIREN, six networks | 4,066 B / 28.63 dB | 4,067 B / 27.99 dB |
| Shared SIREN, K=1 | 1,017 B / 22.27 dB | 1,023 B / 22.09 dB |
| Shared SIREN, K=3 | 2,540 B / 26.03 dB | 2,542 B / 24.90 dB |
| WebP quality=25, separate files | 988 B / **31.55 dB** | same deterministic baseline |
| WebP quality=80, separate files | 1,740 B / **38.28 dB** | same deterministic baseline |

**Conclusion:** This SIREN configuration does not beat WebP for these natural images.

## Experiment B: six related analytic smooth surfaces — promising restricted case

Six correlated 64 × 64 surfaces from the same phase-shifted sinusoidal generator, 600 epochs, layer width=16, latent dimension=4, float16 weight packing + zlib.

| Method | Seed 42: size / PSNR | Seed 123: size / PSNR |
|---|---|---|
| Individual SIRENs | 4,075 B / 40.47 dB | 4,094 B / 39.22 dB |
| Shared SIREN, K=1 | **1,023 B / 47.56 dB** | **1,023 B / 48.07 dB** |
| Shared SIREN, K=3 | 2,538 B / 46.85 dB | 2,532 B / 48.23 dB |
| WebP quality=25, six files | 1,026 B / 38.08 dB | identical baseline |
| WebP quality=80, six files | 1,384 B / 43.77 dB | identical baseline |
| WebP quality=95, six files | 2,296 B / 48.94 dB | identical baseline |
| Animated WebP quality=95 (6 frames) | 2,304 B / 48.99 dB | identical baseline |

Two-seed K=1 shared SIREN mean: **47.81 dB / 1,023 B**. At roughly matching bytes, WebP quality=25 achieves 38.08 dB / 1,026 B. This is an encouraging domain-specific rate–distortion result, **not evidence of superiority for general natural images**. Sampling a correlated analytic family is favorable to sharing model parameters. Additional seeds and held-out real-world datasets are necessary.

## Reproduction using exact repository code

```sh
python research/universe/experiment_51_fair_sharing.py --self-test
python research/universe/experiment_51_fair_sharing.py --dataset natural --count 6 --size 32 --epochs 300 --widths 16 --k 1,3 --seeds 42,123 --compare-codecs --out /tmp/exp51-natural
python research/universe/experiment_51_fair_sharing.py --dataset smooth --count 6 --size 64 --epochs 600 --widths 16 --k 1,3 --seeds 42,123 --compare-codecs --out /tmp/exp51-smooth
```

The third command was run by the repository GitHub Actions job (two seeds, K=1) and the printed neural sizes and post-decode PSNR matched the independent check. The natural-image full-size command remains a proposed repository reproduction step. The exact workflow includes the 64px Exp 52 adaptation with two seeds.

## Methodological limitations and next hypothesis

- Only two complete seeds were retained for the smooth set; more seeds and uncertainty intervals needed.
- WebP/JPEG/PNG are evaluated on 8-bit rounded grayscale images; neural models train on float32 surfaces. All decoded outputs are evaluated against the same float32 reference, so the 8-bit precision ceiling applies to those codecs.
- The corpus is very small (N=6). Evaluate larger independent sets and **real contiguous texture/terrain tiles** with a separate held-out corpus.
- Report compression/decompression time and model-training cost before discussing practicality.
- Compare other multiframe or corpus-level codecs, WebP animation, AVIF, JPEG XL, and lossless tile predictors where supported.
- Some planned large-width runs did not complete in the allotted runtime and must not be silently excluded when publishing aggregate claims.
- Evaluate terrain RMSE in meters as well as PSNR for real geographic elevation data.

**Working hypothesis for Experiment 52:** parameter sharing can exploit a low-dimensional *family* of structurally related surfaces, but has insufficient inductive bias to compete broadly on unrelated natural images. Test under controlled matched-byte and held-out generalization conditions, not just cherry-picked codec settings.

## Follow-up Experiment 52: frozen shared backbone, new latent only

An independent local replication trained a conditional SIREN backbone on the first **five** of the six correlated synthetic textures (64 × 64, 600 training epochs), then froze **all shared network weights** and optimized **only the sixth texture's 4-value latent vector** for another 600 epochs. Serialized float16 weights were decoded before quality measurement.

| Seed | Mean PSNR for first five | Sixth (held-out for backbone) PSNR | Total decoded recipe size |
|---|---:|---:|---:|
| 42 | 47.86 dB | 41.49 dB | 1,021 B |
| 123 | 43.91 dB | 42.42 dB | 1,023 B |

The sixth image was not shown during **backbone training**, but its pixels *were* used during latent optimization; this is therefore encoder-side adaptation, **not zero-shot generalization**. The entire six-image recipe size is given, not a separately claimed per-image size. The code to reproduce with the repository's original serialization functions is `research/universe/experiment_52_holdout_latent.py`. The exact repository implementation has now reproduced the two-seed held-out values in GitHub Actions. Its separately serialized five-image base recipe allows direct measurement of the marginal size: **8 bytes for seed 42 and 10 bytes for seed 123** (header/index/zlib differences included). Evaluate leave-one-out across all six images, realistic procedural families, and new real-world scenes.

## GitHub Actions direct reproduction and verification

The dedicated workflow `.github/workflows/exp51.yml` executed the original repository scripts at revision `81c623d6909daaf7e5076bd43767c539ca5d24cd` on 2026-10-27. Workflow run: https://github.com/Kronos1027/black-hole/actions/runs/40453653259

- **Exp 51, synthetic 64×64, 600 epochs, width 16, seeds 42/123:** individual = 4,075 B / 40.47 dB and 4,094 B / 39.22 dB; one shared model = **1,023 B / 47.56 dB** and **1,023 B / 48.07 dB**. All values printed by the exact code after decoding.
- **Exp 52, same settings, freeze backbone after first five images:** held-out PSNR = **41.49 / 42.42 dB**, six-image model = **1,021 / 1,023 B**, marginal cost of last image = **8 / 10 B**.
- Self-test, PNG/JPEG/WebP decoding, JSON integrity checks, and checks that the saved bytes equal the measured sizes all passed.
- Job published `exp51-smoke-artifacts` containing serialized outputs and `results.json` for these runs.

**Scope warning:** all positive rate–distortion measurements here involve a small specially correlated synthetic family. Do not extrapolate to uncorrelated photos, larger systems, or unseen real-world textures without subsequent controlled experiments.
