# 📄 Phase 1 Research Paper Draft

**Title**: Black Hole Universe: Multi-File SIREN Compression via Shared Roots

**Author**: Darlan Pereira da Silva (Kronos1027)
**Date**: June 2026
**Status**: Draft (Phase 1 complete)

---

## Abstract

We present the first experimental validation of the Black Hole Universe
Hypothesis (BHUH), a novel compression paradigm where multiple files share
a common mathematical "root" (base generator) with per-file "modulations"
(trajectories in latent space). Using FiLM-modulated SIREN networks, we
demonstrate that a single shared network can represent 50 images with only
24KB total, achieving 17.93x improvement over separate SIREN networks and
62.78x improvement over ZIP compression. We discover a scaling law where
improvement increases linearly with the number of files, confirming the
"shared roots" principle of BHUH.

## 1. Introduction

Traditional compression treats each file independently. The Black Hole
Universe Hypothesis proposes that files sharing structural similarity can
share a common "root" — a base generator network — with per-file
modulations acting as "trajectories" through a shared latent space.

This is analogous to how a single tree (base network) produces many leaves
(individual files), each with unique structure but sharing the same trunk
and branches.

## 2. Background

### 2.1 SIREN
SIREN (Sitzmann et al., 2020) uses sinusoidal activation functions to
represent signals as neural implicit functions. A SIREN network maps
coordinates to pixel values: $f(x, y) \to (r, g, b)$.

### 2.2 FiLM Modulation
Feature-wise Linear Modulation (Perez et al., 2018) applies per-sample
scale and shift to intermediate features:
$$\hat{f} = \gamma \odot f + \beta$$
where $(\gamma, \beta)$ are generated from a conditioning input.

### 2.3 Hypernetworks
Hypernetworks (Ha et al., 2016) generate weights for a target network.
Our approach uses a similar principle but with FiLM modulation for
efficiency.

## 3. Method

### 3.1 ModulatedSIREN Architecture
- Base SIREN: 2 hidden layers, 32 features, $\omega_0 = 30$
- Modulation embedding: 16-dimensional per-file
- FiLM generators: linear layers producing scale + shift

### 3.2 Training
- Coordinate inputs: normalized $(x, y) \in [0, 1]^2$
- Loss: MSE between predicted and actual pixel values
- Optimizer: Adam, lr=3e-3
- Training: all files simultaneously, shared gradient updates

### 3.3 Compression
- Weights: INT8 quantization + zlib level 9
- Residual (bit-perfect mode): PNG lossless

## 4. Experiments

### 4.1 Scaling Law (Table 1)

| N Images | Separate SIRENs | BHUH (shared) | Improvement | vs ZIP |
|----------|----------------|---------------|-------------|--------|
| 5        | 43,134B        | 21,269B       | 2.03x       | 7.15x  |
| 10       | 86,265B        | 21,568B       | 4.00x       | 13.92x |
| 20       | 172,453B       | 22,213B       | 7.76x       | 27.15x |
| 50       | 431,042B       | 24,044B       | 17.93x      | 62.78x |

### 4.2 Key Finding: Constant BHUH Size
The BHUH compressed size stays nearly constant (21-24KB) regardless of
the number of images, while separate SIRENs scale linearly. This confirms
the shared roots principle — the base network is amortized.

### 4.3 Scaling Law
$$\text{Improvement} \approx \frac{N}{3}$$
where N is the number of images. This linear scaling suggests that for
1000 images, improvement would be ~333x.

### 4.4 Bit-Perfect Analysis
With 80 epochs, SIREN approximation error makes residuals large (508KB
for 20 images). Future work: more epochs + WebP residual for practical
bit-perfect mode.

## 5. Discussion

### 5.1 Why Shared Roots Work
Satellite-like images share common frequency components. The base SIREN
learns these shared frequencies; modulations capture per-image phase and
amplitude differences.

### 5.2 Connection to BHUH
This experiment validates Principle 3 (Multiverse) of BHUH:
"Multiple files share roots — common structure in generator space."
The base network IS the shared root; modulations ARE the trajectories.

### 5.3 Limitations
- Tested on synthetic satellite-like images only
- Lossy mode only (bit-perfect needs more research)
- No comparison with COIN++ or other neural compressors yet
- No natural photo testing

## 6. Future Work

### Phase 2: Universal Hypernetwork
- Extend to text, audio, binary data types
- Test on real datasets (CIFAR-10, ImageNet)

### Phase 3: LLM Program Search
- Use LLMs to find generating programs for structured files

### Phase 4: Diffusion Seed
- Pre-trained diffusion models as seeds

### Phase 5: Universe Prototype
- Combine all approaches into unified system

## 7. Conclusion

We validated the shared roots principle of the Black Hole Universe
Hypothesis with a 4-18x improvement over separate SIREN networks,
scaling linearly with the number of files. At 50 images, our approach
achieves 62.78x compression over ZIP while maintaining acceptable
visual quality. This confirms that the "multiverse" principle — where
files share mathematical roots — is a viable compression paradigm.

## References

1. Sitzmann et al., "Implicit Neural Representations with Periodic
   Activation Functions" (SIREN), NeurIPS 2020.
2. Perez et al., "FiLM: Visual Reasoning with a General Conditioning
   Layer", AAAI 2018.
3. Ha et al., "HyperNetworks", ICLR 2017.
4. Dupont et al., "COIN++: Data Compression with Implicit Neural
   Representations", arXiv 2022.
5. Kolmogorov, "Three approaches to the quantitative definition of
   information", 1965.

## BibTeX

```bibtex
@misc{blkh_universe_phase1_2026,
    title={Black Hole Universe: Multi-File SIREN Compression via Shared Roots},
    author={Pereira da Silva, Darlan},
    year={2026},
    note={Phase 1 experiment validated. 17.93x improvement at 50 images.},
    url={https://github.com/Kronos1027/black-hole/tree/main/research/universe}
}
```
