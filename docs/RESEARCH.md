# Scientific Foundation: Black Hole Architecture

## Core Papers and References

### 1. SIREN — Sinusoidal Representation Networks

> Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, Gordon Wetzstein.  
> *Implicit Neural Representations with Periodic Activation Functions.* NeurIPS 2020.

**Why it matters:** SIREN introduced sine-activated MLPs that can represent high-frequency signals (images, audio, 3D shapes) with extraordinary fidelity. This is the backbone of the Black Hole Singularity.

Key insight: Sine activations with proper initialization allow a tiny MLP to learn complex continuous functions. A 3-layer MLP with 256 neurons can represent a high-resolution image.

---

### 2. COIN — Compression with Implicit Neural Representations

> Emilien Dupont, Adam Goliński, Milad Alizadeh, Yee Whye Teh, Arnaud Doucet.  
> *COIN: Compression with Implicit Neural Representations.* 2021.

**Why it matters:** Proved that INRs can beat JPEG at low bit-rates. Instead of storing pixels, store MLP weights. The encoding process is training; the decoding is inference.

Key metrics: A 200KB image compressed to ~2KB of neural weights with acceptable quality.

---

### 3. COIN++ — Meta-Learned Compression

> Emilien Dupont, Hrushikesh Loya, Milad Alizadeh, Adam Goliński, Yee Whye Teh, Arnaud Doucet.  
> *COIN++: Neural Compression Across Modalities.* 2022.

**Why it matters:** Introduced a meta-learned base network. Instead of storing full weights, store only lightweight modulations. 100x faster encoding than COIN. This is the future direction for Black Hole's recipe storage.

---

### 4. Siamese SIREN — Audio Compression

> Lazendorfer & Wattenhofer.  
> *Siamese SIREN: Audio Compression with Implicit Neural Representations.* 2023.

**Why it matters:** Demonstrated shared-layer architectures for parameter reduction. A 2x256 shared backbone + 1x128 head achieved 303KB quantized recipe size with 2.58 PESQ.

---

### 5. NeRV — Neural Representations for Videos

> Hao Chen, Bo He, Hanyu Wang, et al.  
> *NeRV: Neural Representations for Videos.* NeurIPS 2021.

**Why it matters:** Replaced coordinate-wise MLPs with convolutional decoders. This eliminates the inference bottleneck of querying millions of coordinates. Future Black Hole versions will incorporate NeRV-style decoders for video data.

---

### 6. D'OH — Hypernetworks for INRs

> Schwarz et al.  
> *D'OH: Decoder-Only Random Hypernetworks for Implicit Neural Representations.* 2024.

**Why it matters:** Explores weight quantization, pruning, sparsity, and entropy coding for INR weights. This directly enables the extreme compression ratios targeted by Black Hole.

---

## The Paradigm Shift

| Traditional | Black Hole |
|-------------|------------|
| Data is static bytes | Data is a living function |
| Compression = packaging | Compression = learning |
| Decompression = CPU spike | Decompression = always warm |
| Disk → RAM copy | Recipe → RAM inference |
| File system hierarchy | Singularity continuum |

---

## Open Problems

1. **High-entropy data:** Random data (encrypted files, keys) cannot be compressed by any means, including INRs. The Kolmogorov complexity equals the data length.
2. **Training time:** Current INRs need minutes to train per image. COIN++ and meta-learning are reducing this to seconds.
3. **Generalization:** INRs are overfit to a single instance. Each file needs its own network (or modulation).
4. **Hardware integration:** True zero-copy requires kernel-level drivers (io_uring, DMA, GPU DirectStorage).

---

## Glossary

- **INR (Implicit Neural Representation):** A signal represented as a neural network function f_θ(x) rather than discrete samples.
- **SIREN:** Sinusoidal Representation Network — a specific INR using sine activations.
- **Recipe:** The neural weights θ that define the compressed signal.
- **Singularity:** The core compression engine where data is transformed into recipes.
- **Horizon of Events:** The boundary where data transitions from stored state to pre-calculated warm state.
- **Ejection:** The instantaneous reconstruction and mounting of data into RAM.
- **Zero-Copy:** Moving data to memory without intermediate copies or disk round-trips.
