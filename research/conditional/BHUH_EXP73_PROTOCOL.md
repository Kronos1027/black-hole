# BLKH Experiment 73: frozen cross-source learned entropy model (rANS)

**Predeclared before measuring the newly selected source images.** This is an engineering test of *learned probability modeling* with a conventional lossless entropy coder. It is **not** neural network compression, not a new entropy-coding algorithm, and must not be portrayed as a scientific breakthrough.

## Data and independence

Use five standard 8-bit RGB photographs from `scikit-image==0.25.2`: training-probability sources `astronaut`, `coffee`, `chelsea` (already studied in Exp72) and two **new, never used in prior BLKH experiments** evaluation sources `hubble_deep_field`, `immunohistochemistry`. Each uses a deterministic central 128×128 RGB crop, without rescaling, exactly the same crop definition as Exp72. Only these five source photographs count toward independent image count; RGB channels and tile positions are correlated measurements. Preserve original source SHA256, crop SHA256, origin and dimensions. License/provenance follow their upstream scikit-image sample-data metadata; do not assign blanket license terms to the photographs or bundle their distribution dependencies in the repository.

## Frozen predictor, model and bitstream

Train a **static 6-context categorical residual probability model** only on the first three crop images. Reversible Paeth spatial predictor (upper, left, upper-left), per-channel uint8 difference modulo 256; context `3*edge_flag+channel`, where `edge_flag` is `abs(left-up)>24`, using reconstructed neighbor bytes and a zero-valued pixel border. Each context has a 256-symbol residual histogram, add-one smoothing, deterministically normalize the six histograms to **4,096 total frequency units**, minimum frequency 1. **No probability adaptation or retraining on either held-out photo.** Serialize all six 256-entry uint16 frequency tables in the final bundle and count them **once** across all five encoded RGB photos; no learned uncounted decoder-side state.

Entropy coding: standard 32-bit byte-renormalized **rANS** with 12 probability bits and state lower bound `1<<23`, with arithmetic-/bitstream-level independent decode and lossless pixel equality. The bundle serializes a magic/version, source IDs and hashes, RGB dimensions, predictor and model metadata, exactly one probability-model table, per-source compressed-stream length and each rANS byte stream. All bytes count. Per-source payload size alone **must not be claimed as a standalone compressed file size**.

## Strong fair baselines

Compare aggregate byte-accounted complete five-photo rANS bundle against: (1) independently encoded PNG lossless, WebP lossless and reversible JPEG2000 images with a common complete source manifest; and (2) a **single packed 128×640 RGB atlas** in the same lossless formats with that same manifest. Use the **smallest actual decoded complete** baseline; no lossy-vs-lossless comparisons. Include a Paeth residual + zlib level 9 ablation using the same source order, manifest and lossless RGB reconstruction, to identify whether the learned probability model improves beyond a conventional residual compressor.

Report exact decoded equality, archive SHA256, table model SHA256, source and crop SHA256, bytes, encode and decode times. Train on only the three preregistered images. Report the two previously unused sources **separately** (payload sizes and correctness); do not claim the entire bundle is held out. Both standalone model-complete 1-photo accounting (repeated table/header per source) and amortized 5-photo bundle accounting must be clearly distinguished.

## Scientific decision and release gate

Count everything, never tune predictor/context/normalization after viewing outcomes, include unfavorable cases. A useful bounded claim requires a reproducible advantage over the **smallest comparable WebP/PNG/JPEG2000 atlas** in actual complete lossless bytes at exact RGB reconstruction, including all model and metadata overhead. A probability-model or rANS win over an artificially weak compressor is not sufficient. No demo update, main branch merge or LinkedIn scientific breakthrough claim based solely on Exp73.
