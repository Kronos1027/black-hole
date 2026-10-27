# BHUH / BLKH Experiment 51 — preliminary independent validation (2026-10-27)

**Status: reproducible exploration, not a validated compression breakthrough.**

## Provenance

These measurements were produced in an isolated Python environment by an **independent reimplementation** of the Exp 51 architecture and BLK51 float16 + zlib serializer, **not** by executing this repository's exact `experiment_51_fair_sharing.py` script. The reference script on this branch now supports `--dataset smooth` and `--compare-codecs`; run its self-tests and new configurations before citing these measurements as repository-verified. The local replication sources, SHA-256s, raw bytes and output records are preserved separately.

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

The third command can take substantial CPU time. These commands are proposed reproduction steps for the repository implementation, **not claimed as already completed**.

## Methodological limitations and next hypothesis

- Only two complete seeds were retained for the smooth set; more seeds and uncertainty intervals needed.
- WebP/JPEG/PNG are evaluated on 8-bit rounded grayscale images; neural models train on float32 surfaces. All decoded outputs are evaluated against the same float32 reference, so the 8-bit precision ceiling applies to those codecs.
- The corpus is very small (N=6). Evaluate larger independent sets and **real contiguous texture/terrain tiles** with a separate held-out corpus.
- Report compression/decompression time and model-training cost before discussing practicality.
- Compare other multiframe or corpus-level codecs, WebP animation, AVIF, JPEG XL, and lossless tile predictors where supported.
- Some planned large-width runs did not complete in the allotted runtime and must not be silently excluded when publishing aggregate claims.
- Evaluate terrain RMSE in meters as well as PSNR for real geographic elevation data.

**Working hypothesis for Experiment 52:** parameter sharing can exploit a low-dimensional *family* of structurally related surfaces, but has insufficient inductive bias to compete broadly on unrelated natural images. Test under controlled matched-byte and held-out generalization conditions, not just cherry-picked codec settings.
