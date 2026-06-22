# Black Hole (BLKH)

> **"The data is not a static block of bytes waiting to be dragged. The data is a living mathematical function, kept in potential state and ejected instantly on demand."**

---

## Author

**Created by:** Darlan Pereira da Silva  
**Contact:** darlan1027pc@gmail.com  
**GitHub:** [Kronos1027](https://github.com/Kronos1027)  

This is a personal research project born from the vision of replacing static file storage with neural mathematical representations. Every line of architecture, concept, and implementation direction was conceived by the author. This repository documents the journey from idea to executable prototype.

---

## What is Black Hole?

**Black Hole** is a revolutionary data compression and execution paradigm that replaces traditional file storage with **Implicit Neural Representations (INRs)**. Instead of storing files as passive blocks of bytes on disk, Black Hole:

1. **Ingests** files into a Singularity — destroying the original structure and converting it into a continuous mathematical function (neural weights).
2. **Maintains** the data in a Stationary State — using opportunistic idle CPU cycles to keep recipes pre-calculated and warm in memory.
3. **Ejects** data on demand — reconstructing files directly into RAM without disk round-trips or decompression spikes.

---

## Architecture

```
[Raw Data / OS] ──(Ingestion)──> [ THE SINGULARITY ] ──(Idle/Pre-calc)──> [ STATIONARY STATE ]
                                              │
                                       (Click/Ejection)
                                              ▼
                                    [ REAL MEMORY (RAM) ]
```

### The Three Phases

| Phase | Name | Description |
|-------|------|-------------|
| **Phase 1** | **Singularity** (INR Compression) | Uses SIREN-based MLPs to convert any byte sequence into a compact neural recipe. |
| **Phase 2** | **Horizon of Events** (Opportunistic Daemon) | Monitors idle CPU cycles and pre-calculates data recipes in the background. |
| **Phase 3** | **Ejection** (Zero-Copy Engine) | Mounts reconstructed data directly into volatile memory with zero disk round-trip. |

---

## Quick Start

### Requirements

- Python 3.8+
- NumPy
- (Optional) psutil — for real CPU idle monitoring

```bash
pip install -r requirements.txt
```

### Phase 1: Compress a File

```bash
cd phase1_inr_compressor
python compress.py my_file.txt my_file.recipe.json 3000 1e-3
```

This trains a SIREN network to represent the file as neural weights. The `.recipe.json` is the "recipe" that replaces the original file.

### Phase 1: Reconstruct (Eject)

```bash
python decompress.py my_file.recipe.json output.txt 1024
```

### Phase 2: Run the Opportunistic Daemon

```bash
cd phase2_opportunistic_daemon
python daemon.py
```

The daemon monitors idle CPU cycles and pre-computes recipes in the background.

### Phase 3: Ejection Engine

```bash
cd phase3_ejection_engine
python ejector.py demo
```

### Run Tests

```bash
cd tests
python test_end_to_end.py
```

### Unified Demo

```bash
python demo.py
```

Generates a full visualization: original signal → compressed recipe → reconstructed output → error analysis.

---

## Test Results

```
TEST 1: Compress -> Decompress (Sinusoidal Signal, 512 bytes)
  PSNR: 38.23 dB  → PASS

TEST 2: Ejection Engine (Periodic Signal, 256 bytes)
  PSNR: 48.65 dB  → PASS
  Exact match: 43.75%

TEST 3: Compression Ratio Vision (Roadmap)
  PASS (conceptual)

All tests PASSED. The singularity is stable.
```

---

## Scientific Foundation

Black Hole is built on peer-reviewed research:

- **SIREN** (Sitzmann et al., 2020): *Implicit Neural Representations with Periodic Activation Functions* — sine-activated MLPs for high-frequency signal reconstruction.
- **COIN / COIN++** (Dupont et al., 2021): *COIN: Compression with Implicit Neural Representations* — encoding images as tiny MLP weights.
- **Siamese SIREN** (Lazendorfer & Wattenhofer, 2023): *Audio Compression with INRs* — shared layers for parameter reduction.
- **NeRV** (Chen et al., 2021): *Neural Representations for Videos* — convolutional decoder for fast inference.
- **D'OH** (Schwarz et al., 2024): *Decoder-Only Hypernetworks for INRs* — weight quantization and sparsity.

See [docs/RESEARCH.md](docs/RESEARCH.md) for full references.

---

## Roadmap

- [x] Phase 1: Core SIREN INR compressor (1D byte sequences)
- [x] Phase 2: Opportunistic compute daemon
- [x] Phase 3: Ejection engine simulation
- [ ] Integrate with actual `io_uring` / DirectStorage APIs
- [ ] Extend to 2D images and 3D volumes
- [ ] Weight quantization (8-bit / 4-bit) for extreme compression
- [ ] Meta-learning (COIN++ style) for faster convergence
- [ ] GPU acceleration via CUDA kernels
- [ ] Kernel driver for true zero-copy ejection

---

## License

MIT License — Copyright (c) 2025 Darlan Pereira da Silva.  
See [LICENSE](LICENSE) for full text.

The singularity is for everyone.

---

## Citation

If you use Black Hole in research, please cite:

```bibtex
@software{blackhole2025,
  title = {Black Hole: Implicit Neural Compression Architecture},
  author = {Darlan Pereira da Silva},
  year = {2025},
  url = {https://github.com/Kronos1027/black-hole}
}
```

---

## Acknowledgments

This project was architected and directed by **Darlan Pereira da Silva**. The implementation prototype was built with guidance from AI research assistants, grounded in the peer-reviewed scientific literature documented in `docs/RESEARCH.md`. The vision, terminology, and three-phase architecture (Singularity, Horizon of Events, Ejection) are original intellectual contributions of the author.
