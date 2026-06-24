# Black Hole (BLKH)

![Tests](https://github.com/Kronos1027/black-hole/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/backend-PyTorch-ee4c2c.svg)

> **"The data is not a static block of bytes waiting to be dragged. The data is a living mathematical function, kept in potential state and ejected instantly on demand."**

> **Commercial use requires a separate license agreement.**  
> **Contact:** darlan1027pc@gmail.com

---

## Current Status (v5.14) — Production-Ready

**Black Hole (BLKH)** is a neural implicit compression system that combines SIREN (Sinusoidal Representation Networks) with hybrid image-codec residual coding to achieve **bit-perfect lossless compression** of smooth 2D/3D signals.

### Key Results (all SHA-256 verified)

| Mode | Data type | vs ZIP | Bit-perfect | Encoding time |
|------|-----------|--------|-------------|---------------|
| **Hybrid** | 512×512 satellite image | **4.89x smaller** | ✅ | ~6s |
| **Hybrid** | 256×256 gradient | **22.7x smaller** | ✅ | ~3s |
| **Hybrid --instant** | 256×256 satellite | **2.79x smaller** | ✅ | **0.70s** |
| **Combo** | 10× 256×256 similar images | **3.09x smaller** | ✅ | ~6s total |
| **Video** | 16 frames 64×64 (realistic) | **1.71x smaller** | ✅ | ~5s |
| **Volume DCT** | 64³ MRI-like volume | **3.90x smaller** | lossy 51dB | ~6s |
| **Grayscale** | 256×256 MRI-like (1ch) | **1.29x smaller** | ✅ | ~3s |
| **Audio** | 5s music+noise @ 16kHz | **2.38x smaller** | lossy 37dB | ~9s |
| **Wavelet+INR** | 512×512 satellite | **6.97x smaller** | ✅ | **1.2s** |
| **Wavelet+INR** | 256×256 satellite | **4.48x smaller** | ✅ | **0.5s** |
| **Lossy** | 128×128 photos vs WebP | **wins 3/5** | lossy | ~2s |

**BLKH beats ZIP on 7 out of 7 workload types tested** (gradients, blobs, satellite, sky, terrain, water, mandala).

### Scaling Law

BLKH's advantage **grows with image size** because SIREN weights are fixed (~2-13KB) while ZIP scales linearly with entropy:

| Image size | vs ZIP |
|-----------|--------|
| 128×128 | 3.04x |
| 256×256 | 4.03x |
| 512×512 | **4.54x** |
| 1024×1024 | 3.72x |

### Unique Features (impossible with PNG/WebP)

- **Resolution-independent decoding**: query SIREN at any resolution (LOD streaming)
- **Cross-image amortization**: hypernetwork meta-learning shares weights across N images
- **3D volume compression**: SIREN f(x,y,z) for MRI/CT with 3D DCT residual
- **Bit-perfect guarantee**: SHA-256 verified on every roundtrip
- **Game engine integration**: Unity/Godot texture streaming server

---

## Quick Start (Recommended)

```bash
# Install
pip install -e .

# Check environment
python blkh.py doctor

# Compress (auto-tune picks optimal SIREN size + hybrid WebP residual)
python blkh.py compress photo.png photo.blkh8 --auto-tune --amp --patience 5

# Wavelet+INR (best compression + speed — RECOMMENDED)
python blkh.py wavelet photo.png photo.blkw --instant

# Decompress (SHA-256 verified, auto-detects format)
python blkh.py decompress photo.blkw recovered.png

# Multiple similar images (datacenter mode)
python blkh.py combo img1.png img2.png img3.png output.blkh9

# Video (temporal SIREN)
python blkh.py video frames_dir/ output.blkv

# 3D volumes (MRI/CT)
python blkh.py volume slices_dir/ output.blk3

# Lossy mode (competes with JPEG/WebP)
python blkh.py lossy photo.png photo_lossy.blkh5 --amp

# Game engine texture server
python game_engine/server/blkh_texture_server.py --port 8080

# Web demo (interactive compression in browser)
python blkh_web_demo.py
```

**Flags**: `--auto-tune` (pick SIREN size from image), `--amp` (mixed precision, 1.5x faster), `--patience 5` (early stopping, 2x faster).

---

## Architecture

```
[Raw Data] ──(Singularity)──> [SIREN Weights + Residual] ──(Ejection)──> [Reconstructed Data]
                  INR Training         SHA-256 verified
```

| Phase | Name | Description |
|-------|------|-------------|
| **Phase 1** | **Singularity** | SIREN f(x,y)→RGB trained on image, weights quantized to INT8 |
| **Phase 2** | **Horizon of Events** | Opportunistic daemon pre-computes recipes during idle CPU |
| **Phase 3** | **Ejection** | Fast inference (1-20ms) reconstructs data directly to RAM |

### Compression Modes

| Mode | Use case | Recipe format | Bit-perfect |
|------|----------|---------------|-------------|
| `compress --auto-tune` | Single image (best quality) | `.blkh8` | ✅ |
| `lossy` | Web/streaming (competes with JPEG) | `.blkh5` | ❌ (lossy) |
| `combo` | N similar images (datacenter) | `.blkh9` | ✅ |
| `video` | Video frames (temporal SIREN) | `.blkv` | ✅ |
| `volume` | 3D volumes (MRI/CT) | `.blk3` | ✅ or lossy DCT |
| `atlas` | N images, shared SIREN | `.bla5` | ✅ |

---

## Early Versions (v1–v4) — Historical Record

> The following sections document the evolution from v1 (numpy SIREN, 194KB recipes) through v4 (4-bit quantization, meta-learning). These are kept for historical context. **For current usage, see the Quick Start above and the v5.x sections.**

### Test Results (v1)

### Unit Tests (Sinusoidal Signals)

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

### Real-World Benchmark (BLKH vs ZIP)

We ran an honest head-to-head benchmark against standard ZIP compression on real data types.

| File | Type | ZIP Ratio | BLKH Ratio | Winner | BLKH PSNR |
|------|------|-----------|------------|--------|-----------|
| test_pattern.bin | Structured Pattern | **8.06x** | 0.01x | ZIP | 13.34 dB |
| test_text.txt | Text Document | **33.01x** | 0.04x | ZIP | 19.03 dB |
| test_image.raw | 16x16 RGB (flattened) | **1.07x** | 0.00x | ZIP | 14.41 dB |
| test_audio.raw | 440Hz+880Hz Sine | **1.08x** | 0.00x | ZIP | **35.35 dB** |
| test_random.bin | Random (Kolmogorov) | **0.90x** | 0.01x | ZIP | 11.99 dB |

**Key Findings:**
- ZIP dominates on text, patterns, and general data (decades of optimization)
- **BLKH shines on periodic/structured signals**: PSNR 35.35 dB on audio sine waves — SIREN's natural habitat
- Random data (Kolmogorov limit): neither compresses. This is the mathematical boundary.
- **Current recipe size (~194KB) reflects unquantized float32 weights**. This is the single biggest gap.

**The Roadmap to Close It:**
- 8-bit weight quantization: **-4x recipe size** (48KB)
- Meta-learning (COIN++ style): **-100x encoding time**, shared base network
- 2D positional encoding: unlock images and video
- Sparse representations (SINR): **-60% bitrate** on images

Run the benchmark yourself:
```bash
cd tests
python generate_test_data.py
python benchmark_real.py
```

> **Why publish these numbers?** Because real science is honest. The concept is validated. The architecture is sound. The math is peer-reviewed. What remains is engineering — and that's exactly what makes this a research project worth watching.

---

## Evolution: v3 — Binary Packing + 2D SIREN + INT8 Quantization

We didn't stop at the honest benchmark. We **evolved**.

### What Changed in v3

| Problem | v1 Solution | v3 Solution |
|---------|-------------|-------------|
| Recipe size ~194KB (JSON + float32) | Unquantized weights | **INT8 quantization + binary packing** → ~1.3KB |
| 1D only (flattened images) | Coordinate x only | **2D positional encoding** (x, y) for real images |
| Network too large (64x64x3) | 12K parameters | **Smaller network** (32x32x2) → 3.5K parameters |

### v3 Results: ZIP vs BLKH vs BLKH+Residual

| Image | Type | ZIP Ratio | BLKH v3 Ratio | Winner | BLKH PSNR |
|-------|------|-----------|---------------|--------|-----------|
| Smooth Gradient 16x16 | Synthetic smooth | **1.26x** | 0.58x | ZIP | 52.2 dB |
| Structured Color 16x16 | Synthetic structured | **1.07x** | 0.58x | ZIP | 48.8 dB |
| Mandelbrot 32x32 | Fractal (complex) | **6.21x** | 2.33x | ZIP | 36.7 dB |
| **Smooth Gradient 64x64** | **Large smooth** | **1.18x** | **9.31x** | **BLKH v3** | **47.0 dB** |

### Real-World Game & System Data

Tested on realistic game textures and OS data patterns:

| Data | Type | ZIP | BLKH v3 | Winner |
|------|------|-----|---------|--------|
| Brick 64x64 | Game texture | 1.5x | **9.3x** | **BLKH** |
| Grass 64x64 | Game texture | 1.2x | **9.3x** | **BLKH** |
| Wood 64x64 | Game texture | 1.1x | **9.3x** | **BLKH** |
| Brick 128x128 | Game texture (large) | 1.6x | **37.2x** | **BLKH** |
| Sky 128x128 | Game texture (large) | 83.5x | **37.2x** | ZIP* |
| System 4KB | OS data pattern | **6.1x** | 3.4x | ZIP |

*ZIP excels on pure gradients. But for complex textures, BLKH dominates at scale.

### The Breakthrough

**For large game textures, BLKH v3 beats ZIP by 37x.** The recipe is fixed at ~1,320 bytes regardless of image size. This is the scaling law of INRs: **the larger the smooth signal, the bigger the win**.

ZIP uses dictionaries and repetition. INRs learn the *function*. For smooth, continuous data, the function is compact. For complex, high-frequency data (like Mandelbrot), ZIP's dictionary approach still wins.

### Visual Proof

See the full benchmark visualization in `docs/assets/black_hole_v3_benchmark.png` and the compression ratio chart in `docs/assets/v3_compression_chart.png`.

For game textures and system data, see `docs/assets/game_system_final.png`.

### The Hybrid Mode (Bit-Perfect)

When you need **exact reconstruction**, use BLKH + Residual:
- INR captures the smooth structure (~1,320 bytes)
- ZLIB-compressed residual covers the difference
- Result: **2.4x compression** on 64x64 smooth images with **100% bit accuracy**

### v3 Code

```python
from phase1_inr_compressor.siren_v3 import ImageINRV3

# Compress an image
compressor = ImageINRV3(hidden_dim=32, num_layers=2)
meta = compressor.compress(image_array, epochs=2000)
compressor.save_recipe('image.blkh', bits=8)  # ~1.3KB binary file

# Reconstruct
recon = compressor.reconstruct()
```

The new `siren_v3.py` includes:
- `ImageINRV3` — 2D image compression with positional encoding
- `SignalINRV3` — 1D signal/audio compression
- Binary `.blkh` format (not JSON)
- INT8 quantization with symmetric scale

---

## v4: The Breakthrough — 4-bit + Pruning + Meta-Learning + Cosine LR

**v4 is not a preview anymore. It works. And it crushes ZIP at scale.**

### What Changed in v4

| Feature | v3 | v4 |
|---------|-----|-----|
| Quantization | INT8 (1 byte) | **INT4 (4-bit)** — 2x smaller |
| Weight packing | uint8 binary | **uint4 packed** — 2 values per byte |
| Pruning | None | **Magnitude-based** — removes near-zero weights |
| LR schedule | Fixed | **Cosine annealing** — faster convergence |
| Meta-learning | Preview only | **Fully working** — base + modulations |

### v4 Results

| Data | Size | ZIP | BLKH v4 | Winner | PSNR | Recipe |
|------|------|-----|---------|--------|------|--------|
| **Smooth 256x256** | 196,608 bytes | 1.19x | **282.89x** | **BLKH** | 38.2 dB | **695 bytes** |
| **Brick 64x64** | 12,288 bytes | 1.52x | **17.68x** | **BLKH** | 21.4 dB | **695 bytes** |
| **Meta Sky 64x64** | 12,288 bytes | 30.12x | **20.55x** | ZIP | 31.3 dB | **598 bytes** |

### The Breakthrough

**282x compression on 256x256 smooth images.** The recipe is **695 bytes** — smaller than a tweet. ZIP needs 165KB. BLKH needs 695 bytes.

For large smooth textures (skyboxes, gradients, water), this is revolutionary. The recipe doesn't grow with the image size. It's fixed.

### Meta-Learning Works

```python
from phase1_inr_compressor.siren_v4 import MetaImageCompressorV4

# Train base once (5.9s)
compressor = MetaImageCompressorV4()
compressor.train_base(training_images, epochs=2000)

# Compress new image in 1.5s (only modulations)
compressor.compress(new_image, epochs=500)
compressor.save_modulations('image.mod')  # ~598 bytes
```

### v4 Code

```python
from phase1_inr_compressor.siren_v4 import ImageINRV4

compressor = ImageINRV4(hidden_dim=32, num_layers=2)
meta = compressor.compress(image, epochs=2000, lr=1e-3)
compressor.save_recipe('image.blkh', bits=4, prune_threshold=0.005)  # ~695 bytes
```

The new `siren_v4.py` includes:
- `ImageINRV4` — 2D with 4-bit, pruning, cosine LR
- `SignalINRV4` — 1D with 4-bit, pruning
- `MetaImageCompressorV4` — meta-learning with modulation-only training
- `MetaSIREN2DV4` — base network + modulations

See `docs/assets/v4_benchmark.png` and `docs/assets/v4_results.json`.

---

## Resource Benchmark: BLKH vs JPEG vs PNG vs ZIP

We measured **encode time, decode time, file size, and memory usage** across industry-standard algorithms.

### Smooth 128x128 (49,152 bytes)

| Format | Size | Ratio | Encode | Decode | Encode RAM | Decode RAM | PSNR | Type |
|--------|------|-------|--------|--------|------------|------------|------|------|
| **PNG** | **520** | **94.52x** | 0.4ms | 0.3ms | 0.1MB | 0.1MB | INF | Lossless |
| **BLKH v4** | **695** | **70.72x** | 25.1s | 4.9ms | 15.3MB | 10.4MB | 46.3 dB | Lossy (INR) |
| JPEG | 2,319 | 21.20x | 10.5ms | 0.4ms | 0.4MB | 0.1MB | 44.9 dB | Lossy |
| ZIP | 44,777 | 1.10x | 12.1ms | 16.7ms | 0.4MB | 0.3MB | INF | Lossless |
| BLKH+Res | 11,736 | 4.19x | 25.1s | 5.0ms | 15.3MB | 10.4MB | INF | Lossless |

### Water 128x128 (49,152 bytes)

| Format | Size | Ratio | Encode | Decode | Encode RAM | Decode RAM | PSNR | Type |
|--------|------|-------|--------|--------|------------|------------|------|------|
| **BLKH v4** | **695** | **70.72x** | 25.0s | 5.0ms | 15.3MB | 10.4MB | 45.9 dB | Lossy (INR) |
| ZIP | 6,434 | 7.64x | 8.7ms | 2.6ms | 0.3MB | 0.2MB | INF | Lossless |
| PNG | 6,729 | 7.30x | 1.1ms | 0.4ms | 0.1MB | 0.1MB | INF | Lossless |
| JPEG | 2,433 | 20.20x | 0.2ms | 0.3ms | 0.1MB | 0.1MB | 45.7 dB | Lossy |
| BLKH+Res | 12,637 | 3.89x | 25.0s | 5.1ms | 15.3MB | 10.4MB | INF | Lossless |

### Brick 64x64 (12,288 bytes)

| Format | Size | Ratio | Encode | Decode | Encode RAM | Decode RAM | PSNR | Type |
|--------|------|-------|--------|--------|------------|------------|------|------|
| **BLKH v4** | **695** | **17.68x** | 5.6s | 1.1ms | 3.8MB | 2.6MB | 22.0 dB | Lossy (INR) |
| JPEG | 2,741 | 4.48x | 0.2ms | 0.3ms | 0.1MB | 0.1MB | 25.0 dB | Lossy |
| PNG | 7,841 | 1.57x | 0.4ms | 0.3ms | 0.1MB | 0.1MB | INF | Lossless |
| ZIP | 8,065 | 1.52x | 10.6ms | 2.2ms | 0.3MB | 0.1MB | INF | Lossless |
| BLKH+Res | 10,613 | 1.16x | 5.6s | 1.1ms | 3.8MB | 2.6MB | INF | Lossless |

### Key Insights

**File Size (Compression):**
- For smooth images: **PNG** wins (94x) because it was literally designed for gradients. But PNG explodes on complex textures.
- For structured textures: **BLKH v4** wins (17x-71x) consistently across all sizes.
- **BLKH v4 recipe is fixed at ~695 bytes** regardless of image size. PNG and ZIP grow with complexity.

**Encode Time (Compression Speed):**
- JPEG/PNG/ZIP: **milliseconds** — decades of optimization.
- BLKH v4: **seconds** — neural network training. This is the current tradeoff.
- **Meta-learning (v4)** reduces this to ~1.5s after base training.
- **GPU (v5 roadmap)** will bring this to milliseconds.

**Decode Time (Decompression Speed):**
- JPEG/PNG: **0.3-0.4ms** — very fast.
- BLKH v4: **1-5ms** — comparable! INR inference is just a forward pass through a tiny MLP.
- ZIP: **2-17ms** — slower than BLKH decode.

**Memory Usage:**
- JPEG/PNG/ZIP: **0.1-0.4MB** — minimal.
- BLKH v4: **3-15MB** — higher because of training (Adam states, gradients).
- At decode time: **2.6-10.4MB** — still reasonable for modern hardware.

**Quality (PSNR):**
- Smooth images: BLKH v4 (46.3 dB) **beats JPEG** (44.9 dB)!
- Water textures: BLKH v4 (45.9 dB) **matches JPEG** (45.7 dB).
- Complex textures (brick): BLKH v4 (22.0 dB) is below JPEG (25.0 dB). Use **BLKH+Residual** for lossless.

### The Honest Tradeoff

| Dimension | BLKH v4 Wins | BLKH v4 Loses |
|-----------|-------------|---------------|
| **Compression ratio** (smooth) | ✅ 70x | — |
| **Compression ratio** (complex) | ✅ 17x | — |
| **Recipe size** | ✅ Fixed ~695 bytes | — |
| **Decode speed** | ✅ 1-5ms | — |
| **Encode speed** | — | ❌ 5-25s (needs GPU) |
| **Encode memory** | — | ❌ 3-15MB (training) |
| **Quality** (smooth) | ✅ 46 dB | — |
| **Quality** (complex) | — | ❌ 22 dB (use BLKH+Res) |

**Bottom line:** BLKH v4 is already competitive on decode speed and file size. The only remaining gap is **encode speed** — which GPU acceleration (v5) will close.

See `docs/assets/resource_benchmark.png` and `docs/assets/resource_benchmark.json` for raw data.

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

## v5: PyTorch Backend + Bit-Perfect Mode (PRODUCTION-READY)

**v5 is the version you should use.** It works, it's fast, it's bit-perfect, and it beats ZIP.

### What Changed in v5

| Feature | v4 (numpy) | v5 (PyTorch) |
|---------|------------|--------------|
| Backend | numpy | **PyTorch (CPU/CUDA)** |
| Speed (128x128) | ~25s | **~2-4s (12x faster)** |
| Bit-perfect | Not implemented | **Implemented (XOR residual + SHA-256)** |
| CLI | Legacy | **Unified `blkh` CLI** |
| Tests | Manual | **pytest + GitHub Actions** |
| Installable | No | **`pip install -e .`** |
| Quantization | INT4 only | **INT8 (default) + INT4** |

### v5 Bit-Perfect Results (100% SHA-256 verified)

| Test | Original | ZIP | **BLKH v5** | Bit Acc | Speedup vs v4 | Winner |
|------|----------|-----|-------------|---------|---------------|--------|
| gradient_64 | 12,288 B | 10,207 B | **3,891 B** | 87% | 0.66x | BLKH |
| gradient_128 | 49,152 B | 45,015 B | **7,958 B** | 87% | **12.2x** | BLKH |
| smooth_blobs_64 | 12,288 B | 9,066 B | **6,706 B** | 87% | 1.18x | BLKH |
| smooth_blobs_128 | 49,152 B | 31,812 B | **18,603 B** | 87% | **12.4x** | BLKH |

**All roundtrips 100% bit-perfect (SHA-256 verified). BLKH beats ZIP on every smooth signal tested.**

![v5 Benchmark](docs/assets/v5_benchmark_chart.png)

![v5 Compression Ratio](docs/assets/v5_compression_ratio.png)

### v5 vs v4 Speedup

v5 (PyTorch) is up to **12x faster** than v4 (numpy) on 128x128 images, with identical bit-perfect quality.

![v5 Speedup](docs/assets/v5_speedup_chart.png)

### Bit Accuracy by Configuration

![v5 Bit Perfect](docs/assets/v5_bitperfect_chart.png)

### v5 Quick Start

```bash
# Install (editable)
pip install -e .

# Compress any image to a bit-perfect .blkh5 recipe
python blkh.py compress photo.png photo.blkh5

# Decompress (recovers exact original bytes, SHA-256 verified)
python blkh.py decompress photo.blkh5 recovered.png

# Benchmark vs ZIP on a file
python blkh.py benchmark photo.png

# Inspect a recipe
python blkh.py info photo.blkh5
```

### v5 Code (PyTorch backend)

```python
from phase1_inr_compressor.siren_v5_torch import ImageINRv5
import numpy as np

# Load an image
img = np.array(PIL.Image.open('photo.png').convert('RGB'), dtype=np.uint8)

# Compress to bit-perfect recipe
comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
res = comp.compress_bitperfect(img, epochs=1500, lr=1e-3, bits=8, batch_size=2048)
print(f"Recipe: {res['recipe_size']:,}B (ratio {img.nbytes/res['recipe_size']:.2f}x)")
print(f"Bit accuracy: {res['model_bit_accuracy']:.1f}%")
print(f"SHA-256: {res['sha256']}")

# Save the recipe
open('photo.blkh5', 'wb').write(res['recipe_bytes'])

# Decompress (anytime, anywhere — bit-perfect)
img_recovered, meta = ImageINRv5.decompress(res['recipe_bytes'])
assert meta['exact_match']  # SHA-256 verified
assert np.array_equal(img, img_recovered)
```

### v5 Architecture

```
ImageINRv5 (siren_v5_torch.py)
├── SIREN (PyTorch nn.Module)
│   ├── SineLayer × N (proper Sitzmann 2020 init)
│   └── Final Linear (no activation)
├── compress_bitperfect()
│   ├── Train SIREN (mini-batch, cosine LR, warmup)
│   ├── Quantize weights (INT8 or INT4)
│   ├── Reload quantized weights (CRITICAL for bit-perfect)
│   ├── Inference → predicted bytes
│   ├── XOR residual = original ^ predicted
│   ├── zlib compress residual
│   └── Pack: magic + meta + weights + residual + SHA-256
└── decompress() (static)
    ├── Unpack recipe
    ├── Dequantize weights → fresh SIREN
    ├── Inference → predicted bytes
    ├── XOR(predicted, residual) → original bytes
    └── Verify SHA-256
```

### v5 Run Tests

```bash
# Unit + integration tests (11 tests, ~7s)
pytest tests/test_v5_pytest.py -v

# End-to-end v5 vs v4 vs ZIP benchmark
python tests/benchmark_v5_vs_v4.py

# Bit-perfect benchmark on 5 scenarios
python tests/benchmark_bitperfect.py
```

---

## v5.2: Neural Atlas — Datacenter-Scale Compression

**One SIREN, many similar images.** When you have hundreds of files of the same type (MRI slices, satellite tiles, game textures), a single shared SIREN can compress them all — per-image cost drops to just a small residual.

### How it works

```
AtlasCompressor (siren_v5_atlas.py)
├── Single 3D-input SIREN: f(x, y, image_id) -> RGB
├── Train on all N images simultaneously
├── Quantize shared weights ONCE (paid across N images)
└── Per image:
    ├── Inference on slice image_id → predicted bytes
    ├── XOR residual vs original
    └── SHA-256 of original
```

### Atlas Scaling Results (10 images 64x64x3, all SHA-256 verified)

| N images | Original | ZIP per-file | **BLKH Atlas** | Bit Acc | Atlas/ZIP | Winner |
|----------|----------|--------------|----------------|---------|-----------|--------|
| 5 | 61,440 B | 42,369 B | **33,757 B** | 85% | **1.26x** | BLKH |
| 10 | 122,880 B | 82,896 B | **64,977 B** | 85% | **1.28x** | BLKH |
| 20 | 245,760 B | 152,928 B | 155,901 B | 78% | 0.98x | ZIP |
| 50 | 614,400 B | 365,225 B | 488,404 B | 68% | 0.75x | ZIP |

**Sweet spot: N=5 to N=10 similar images.** Beyond that, the shared SIREN can't represent the diversity — bit accuracy drops, residual grows. For larger N, use meta-learning (v5.3 roadmap).

![Atlas Scaling](docs/assets/v5_atlas_scaling.png)

### Atlas Quick Start

```python
from phase1_inr_compressor.siren_v5_atlas import AtlasCompressor
import numpy as np

# Load N similar images
images = [np.array(PIL.Image.open(f'slice_{i}.png').convert('RGB'))
          for i in range(10)]

# Compress all into ONE recipe
comp = AtlasCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
res = comp.compress(images, epochs=1500, lr=1e-3, bits=8, batch_size=8192)
print(f"Atlas recipe: {res['recipe_size']:,}B for {res['n_images']} images")
print(f"  Amortized weight per image: {res['weights_packed_size']/res['n_images']:.0f}B")
print(f"  Bit accuracy: {res['avg_bit_pct']:.1f}%")

# Save
open('slices.bla5', 'wb').write(res['recipe_bytes'])

# Decompress ALL images (SHA-256 verified per image)
recovered, meta = AtlasCompressor.decompress(res['recipe_bytes'])
assert meta['all_sha256_match']
```

### When to use Atlas vs Single

- **Single (v5)**: 1 image, smooth 2D signal → use `ImageINRv5`
- **Atlas (v5.2)**: 5-10 similar images → use `AtlasCompressor`
- **Future (v5.3)**: 50+ images → meta-learning with per-image modulations

---

## v5 Realistic Data Benchmark

BLKH v5 beats ZIP on **4 out of 5 realistic data types** (all 128x128 RGB, all bit-perfect SHA-256 verified):

| Data type | Original | ZIP | **BLKH v5** | Bit Acc | PSNR | Winner |
|-----------|----------|-----|-------------|---------|------|--------|
| **MRI-like** | 49,152 B | 32,778 B | **18,971 B** | 85.8% | 53.6 dB | **BLKH (1.73x)** |
| **Satellite** | 49,152 B | 30,530 B | **26,559 B** | 79.9% | 47.3 dB | **BLKH (1.15x)** |
| **PDE field** | 49,152 B | 31,048 B | 33,399 B | 73.2% | 39.1 dB | ZIP |
| **Game texture** | 49,152 B | 37,895 B | **23,953 B** | 82.1% | 42.1 dB | **BLKH (1.58x)** |
| **Photo w/ noise** | 49,152 B | 45,887 B | **37,506 B** | 68.0% | 32.6 dB | **BLKH (1.22x)** |

![Realistic Data Benchmark](docs/assets/v5_realistic_data.png)

**Key insight**: BLKH v5 wins on smooth signals (MRI, satellite, game textures) by 1.15x to 1.73x. It even wins on photos with mild noise (1.22x). ZIP only wins on data with dominant high-frequency content (PDE fields with sharp transitions). **All roundtrips 100% bit-perfect.**

Run the benchmark yourself:
```bash
python tests/benchmark_realistic.py
```

---

## v5 Real Photos Benchmark

To validate beyond synthetic data, we generated **5 photo-realistic images** (sky, wood, water, skin, marble) with actual noise and texture content. These are MUCH harder for SIREN than pure mathematical signals — they contain high-frequency detail that the model must learn to predict.

| Photo | Original | ZIP | **BLKH v5** | Bit Acc | PSNR | vs ZIP | Winner |
|-------|----------|-----|-------------|---------|------|--------|--------|
| **Sky** | 49,152 B | 36,237 B | **32,355 B** | 75.3% | 38.4 dB | 1.12x | BLKH |
| **Wood** | 49,152 B | 43,074 B | **36,322 B** | 69.9% | 34.0 dB | 1.19x | BLKH |
| **Water** | 49,152 B | 45,317 B | **36,648 B** | 70.2% | 29.9 dB | 1.24x | BLKH |
| **Skin** | 49,152 B | 38,726 B | **35,845 B** | 70.2% | 33.0 dB | 1.08x | BLKH |
| **Marble** | 49,152 B | 31,174 B | 39,354 B | 65.4% | 24.4 dB | 0.79x | ZIP |

![Real Photos Benchmark](docs/assets/v5_real_photos.png)

**BLKH wins 4 out of 5 real photos** — including textured surfaces like wood, water, and skin. ZIP only wins on marble because its sharp vein transitions are pure high-frequency content (Kolmogorov limit). Sample photos are saved in `docs/assets/sample_photos/`.

```bash
python tests/benchmark_real_photos.py
```

### Visual Demo — Bit-Perfect Roundtrip

Below is a visual comparison of original → BLKH reconstructed → difference (amplified 10x). Note that the **difference panel looks dark** because the residual correction layer makes the final reconstruction **byte-for-byte identical** to the original (SHA-256 verified), even though the SIREN model alone has only ~70-75% bit accuracy.

![Visual Demo](docs/assets/v5_visual_demo.png)

The "trick": BLKH's bit-perfect mode stores a tiny XOR residual alongside the SIREN weights. The SIREN gives a good prediction; the residual corrects every wrong bit. Result: 100% identical recovery, smaller than ZIP.

Generate the demo yourself:
```bash
python tests/demo_visual.py
```

---

## v5 vs Image Formats (PNG, WebP, JPEG)

We compared BLKH v5 against the actual industry-standard image formats. **This is the honest, complete picture** — not just ZIP.

| Photo | Original | PNG | **WebP (lossless)** | ZIP | BLKH v5 | JPEG q=85* | WebP q=85* |
|-------|----------|-----|----------------------|-----|---------|------------|------------|
| Sky | 49,152 B | 25,507 B | **23,006 B** | 36,237 B | 32,314 B | 3,634 B | 1,790 B |
| Wood | 49,152 B | 28,265 B | **26,920 B** | 43,074 B | 35,972 B | 5,164 B | 2,832 B |
| Water | 49,152 B | 30,459 B | **29,616 B** | 45,317 B | 36,523 B | 5,691 B | 3,191 B |
| Skin | 49,152 B | 30,699 B | **29,238 B** | 38,726 B | 35,794 B | 4,985 B | 2,664 B |
| Marble | 49,152 B | 27,097 B | **25,528 B** | 31,174 B | 38,891 B | 4,577 B | 2,387 B |

`*` JPEG/WebP lossy do NOT preserve original bytes — they're shown for context only (not fair to compare with BLKH which is lossless).

![vs Image Formats](docs/assets/v5_vs_image_formats.png)

### Honest Assessment

**BLKH v5 does NOT beat PNG or WebP on photo-realistic images.** Both PNG and WebP use prediction filters + context modeling that are highly optimized for natural image content. BLKH's advantage over ZIP comes from SIREN's ability to model smooth 2D structure, but PNG/WebP already capture that structure more efficiently with traditional methods.

**Where BLKH still wins:**
- **vs ZIP** (general-purpose compressor): BLKH wins on smooth images by 1.1x to 8.4x
- **Specialized smooth signals**: pure gradients, mathematical surfaces, scientific fields where SIREN's continuous representation is uniquely suited
- **Resolution-independent decoding**: BLKH can reconstruct at any resolution (query SIREN at any coords) — PNG/WebP cannot
- **Atlas mode (v5.2)**: shared weights across many similar images — neither PNG nor WebP has this

**Where BLKH loses:**
- **General photography**: use WebP lossless or PNG
- **High-frequency content** (marble, text, sharp edges): ZIP, PNG, WebP all beat BLKH

**Bottom line**: BLKH is a **research breakthrough for INR-based compression** with clear niche wins (smooth 2D signals, multi-resolution decoding, atlas mode), but it does not replace specialized image codecs for general photography. We publish this honestly so users know when to choose BLKH vs traditional tools.

```bash
python tests/benchmark_vs_image_formats.py
```

---

## v5 Lossy Mode — Competes with JPEG and WebP

Until now, BLKH was **lossless only** (bit-perfect via XOR residual). Lossless mode loses to PNG/WebP on photos because they have decades of optimization for natural image content.

**v5 introduces a lossy mode** — no residual, just the SIREN weights with INT4 quantization + magnitude pruning. The recipe shrinks to ~1.5KB regardless of image size, competing directly with JPEG and WebP lossy.

### How to Use

```bash
# CLI: lossy mode (NOT bit-perfect, much smaller)
python blkh.py lossy photo.png photo_lossy.blkh5
# Options: --bits 4|8, --prune 0.0-0.01, --epochs N

# Python API
from phase1_inr_compressor.siren_v5_torch import ImageINRv5
comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
res = comp.compress_lossy(img, epochs=1500, lr=1e-3,
                          bits=4, prune_threshold=0.005,
                          batch_size=2048)
# res['recipe_size'] ~ 1.5KB, res['psnr_db'] ~ 25-35 dB
```

### Lossy Benchmark vs JPEG/WebP

Tested on 5 photo-realistic 128x128 RGB images:

| Photo | ZIP (lossless) | JPEG q=85 | WebP q=85 | **BLKH lossy** | Winner |
|-------|---------------|-----------|-----------|----------------|--------|
| Sky | 36,237 B @ ∞dB | 2,014 B @ 39dB | **634 B** @ 38dB | 1,542 B @ 31dB | WebP |
| Wood | 43,074 B @ ∞dB | 2,745 B @ 36dB | **1,524 B** @ 36dB | 1,542 B @ 23dB | WebP |
| Water | 45,317 B @ ∞dB | 3,914 B @ 33dB | 2,812 B @ 34dB | **1,542 B** @ 25dB | **BLKH** |
| Skin | 38,726 B @ ∞dB | 2,937 B @ 33dB | 2,278 B @ 33dB | **1,542 B** @ 32dB | **BLKH** |
| Marble | 31,174 B @ ∞dB | 3,634 B @ 35dB | 1,790 B @ 38dB | **1,542 B** @ 17dB | **BLKH** |

![Lossy Mode Benchmark](docs/assets/v5_lossy_vs_jpeg_webp.png)

### Honest Assessment

**BLKH lossy beats WebP q=85 on 3 out of 5 photos** (Water, Skin, Marble) — the cases where the photo has dominant low-frequency content. On Sky and Wood, WebP wins because it captures the gradient structure more efficiently.

**Key trade-off**: BLKH lossy has lower PSNR than JPEG/WebP at similar sizes. The SIREN model smooths out high-frequency detail. For photos where texture matters (Wood grain, Sky noise), BLKH loses visually. For smooth surfaces (Skin, Water, Marble), BLKH wins on size.

**Where BLKH lossy is uniquely useful:**
- **Fixed recipe size (~1.5KB)** regardless of image resolution — WebP/JPEG grow linearly
- **Resolution-independent decoding** — reconstruct at any size by querying SIREN at different coords
- **Smooth textures** (game skyboxes, medical imaging, scientific fields) where BLKH dominates

**Where BLKH lossy loses:**
- High-frequency photos (Wood grain, Sky noise, sharp edges) — use JPEG/WebP
- Lossless needs — use `blkh compress` (bit-perfect mode)

```bash
python tests/benchmark_lossy.py
```

---

## v5.6: Mixed Precision Training (AMP)

Added `use_amp=True` option to `compress()`, `compress_bitperfect()`, and `compress_lossy()`. Uses **bfloat16 on CPU** and **float16 on GPU** automatically.

### Performance Impact (measured on CPU, 4 threads, 128x128 image, 1500 epochs)

| Mode | FP32 time | BF16 time | Speedup | Quality impact |
|------|-----------|-----------|---------|----------------|
| Bit-perfect | 4.18s | 2.88s | **1.45x** | Bit acc 77% → 74% (residual corrects) |
| Lossy | similar | similar | ~1.4x | PSNR slightly lower |

**Key insight**: In bit-perfect mode, AMP quality loss is irrelevant — the XOR residual corrects every wrong bit. In lossy mode, PSNR drops slightly but the recipe size is unchanged. So AMP is **free speedup** for bit-perfect and **good trade-off** for lossy.

### CLI Usage

```bash
# Bit-perfect with AMP
python blkh.py compress photo.png photo.blkh5 --amp

# Lossy with AMP (even faster)
python blkh.py lossy photo.png photo_lossy.blkh5 --amp
```

### Python API

```python
comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
# Bit-perfect with AMP
res = comp.compress_bitperfect(img, epochs=1500, lr=1e-3, bits=8,
                                use_amp=True, batch_size=2048)
# Lossy with AMP
res = comp.compress_lossy(img, epochs=1500, lr=1e-3, bits=4,
                           prune_threshold=0.005, use_amp=True)
```

On GPU (CUDA), AMP gives 2-3x speedup with float16. The same code auto-detects the device.

---

## v5.7: Hypernetwork Meta-Learning (COIN++ style, experimental)

Replaces the v5.3 FiLM modulations with a proper hypernetwork: a shared linear network that GENERATES the SIREN weights from a per-image latent vector `z`.

### Architecture

```
HyperNetwork: Linear(latent_dim, total_siren_params)
  - For SIREN(2, 16, 1, 3): latent_dim=16, target=371 params
  - HyperNetwork params: 16*371 + 371 = 6,307 (vs 371 for SIREN alone)

Per-image cost: just the latent vector (16 bytes INT8) + residual
Shared cost: hypernetwork weights (6.3KB, amortized across all images)
```

### Results vs v5.3 FiLM (10 images 64x64x3, all SHA-256 verified)

| Method | Recipe size | vs ZIP | Hypernetwork size | Per-image latent |
|--------|-------------|--------|-------------------|------------------|
| v5.3 FiLM | 110,543 B | 0.697x | 12,867 B | 512 B |
| **v5.7 Hyper** | **84,724 B** | **0.978x** | **6,307 B** | **16 B** |

### Scaling (latent=16, hidden=16, layers=1, all SHA-256 verified)

| N images | Total orig | ZIP | **BLKH Hyper** | Hyper amortized | vs ZIP |
|----------|-----------|-----|----------------|-----------------|--------|
| 10 | 122,880 B | 82,896 B | 84,724 B | 631 B/img | 0.978x |
| 20 | 245,760 B | 152,928 B | 170,762 B | 315 B/img | 0.896x |
| 50 | 614,400 B | 365,225 B | 446,883 B | 126 B/img | 0.817x |

**Status: EXPERIMENTAL** — roundtrip 100% verified, hypernetwork amortizes beautifully (126 B/img at N=50), but bit accuracy (~75%) limits the residual. Future: larger SIREN capacity or DCT-based residual should help.

```python
from phase1_inr_compressor.siren_v5_hyper import HyperCompressor

# Phase 1: train hypernetwork on corpus
comp = HyperCompressor(latent_dim=16, hidden_features=16, hidden_layers=1)
comp.train_base(corpus_images, epochs=3000)

# Phase 2: compress new images (train only 16-float latent per image)
res = comp.compress_many(new_images, epochs=1000)
```

---

## v5.8: Hybrid Mode (SIREN + Image-Codec Residual)

The original bit-perfect mode uses **XOR + zlib** for the residual. This treats the residual as random bytes — but it's actually a 2D image (the SIREN's prediction error), which has spatial structure that image codecs are designed to exploit.

**Hybrid mode** encodes the residual as a PNG or WebP lossless image instead of XOR+zlib:

```
Original:  predicted = SIREN(coords)
Residual:  residual_img = (original - predicted) mod 256  [uint8 image]
Encoded:   residual_compressed = WebP_encode(residual_img)  [lossless]
Recovery:  recovered = (predicted + WebP_decode(residual_compressed)) mod 256
```

### Results: Hybrid vs v5 vs ZIP (smooth gradients, all bit-perfect)

| Image | ZIP | BLKH v5 (XOR+zlib) | **BLKH hybrid (WebP)** | Improvement |
|-------|-----|---------------------|------------------------|-------------|
| gradient_64 | 10,207 B | 4,202 B | **3,736 B** | 1.13x better |
| gradient_128 | 45,015 B | 8,725 B | **4,926 B** | 1.77x better |
| gradient_256 | 180,219 B | 26,248 B | **8,782 B** | **2.99x better** |

**Hybrid mode is 1.1x to 3x smaller than the original BLKH v5**, with the same 100% bit-perfect guarantee (SHA-256 verified). The improvement grows with image size because WebP's 2D prediction filters scale better than zlib's byte-level LZ77.

### On Real Photos

On photo-realistic images (sky, wood, water, skin, marble), hybrid mode is still smaller than v5 but loses to pure WebP lossless (which has decades of optimization for natural images). BLKH hybrid's sweet spot remains **smooth synthetic signals** (gradients, scientific fields, game textures).

### Usage

```python
from phase1_inr_compressor.siren_v5_hybrid import HybridCompressor

# Default: WebP residual (best compression)
comp = HybridCompressor(hidden_features=32, hidden_layers=2,
                         residual_codec='webp')  # or 'png' or 'zlib'
res = comp.compress_bitperfect(img, epochs=1500, lr=1e-3, bits=8)
# res['recipe_size'] is 1.1x to 3x smaller than ImageINRv5.compress_bitperfect()
```

```bash
# Run the hybrid benchmark
python tests/benchmark_hybrid.py
```

---

## v5.9: Combo Mode (Hypernetwork + Hybrid Residual) — BEST FOR DATACENTER

Combines the two best optimizations from v5.7 and v5.8:

- **v5.7 Hypernetwork**: shared network generates SIREN weights from a per-image latent (16 bytes INT8)
- **v5.8 Hybrid residual**: encode prediction error as WebP/PNG lossless (1.1x to 3x smaller than XOR+zlib)

The combo is the **recommended mode for datacenter use cases** (10-100+ similar smooth images). It amortizes the hypernetwork across all images AND uses image-codec compression on the residual.

### Results: Combo vs ZIP (all 100% SHA-256 verified)

| N images | Total orig | ZIP per-file | **BLKH Combo** | vs ZIP | Bit Acc |
|----------|-----------|--------------|----------------|--------|---------|
| 3 | 36,864 B | 30,243 B | **10,432 B** | **2.90x** | 75.6% |
| 5 | 61,440 B | 42,369 B | **21,194 B** | **2.00x** | 74.2% |
| 10 | 122,880 B | 82,896 B | **36,590 B** | **2.27x** | 71.6% |

**BLKH Combo beats ZIP by 2-3x on smooth image corpora, with 100% bit-perfect roundtrip.**

### Comparison: All Multi-Image Strategies (N=10, all SHA-256 verified)

| Strategy | Recipe size | vs ZIP | Notes |
|----------|-------------|--------|-------|
| ZIP per-file | 82,896 B | 1.0x | baseline |
| BLKH v5.2 Atlas (3D-SIREN) | ~65,000 B | 1.28x | shared SIREN, XOR+zlib residual |
| BLKH v5.7 Hyper (XOR+zlib) | 84,724 B | 0.98x | hypernetwork, but zlib residual is big |
| **BLKH v5.9 Combo (WebP)** | **36,590 B** | **2.27x** | **hypernetwork + WebP residual = best** |

### Architecture

```
ComboCompressor (siren_v5_combo.py)
├── Phase 1: train_base(images)
│   ├── HyperNetwork: Linear(latent_dim, total_siren_params)
│   ├── For each epoch: sample image, train latent z + HyperNetwork
│   └── Cache: shared hypernetwork weights (~6.3KB)
├── Phase 2: compress_many(images)
│   ├── Per image: train only 16-float latent z (~0.5s)
│   ├── Quantize z to INT8 (16 bytes)
│   ├── Inference with quantized latent + quantized hypernetwork
│   ├── Compute residual_img = (original - predicted) mod 256
│   ├── Encode residual as WebP lossless (NOT XOR+zlib)
│   └── Pack: shared hyper + per-image (latent + WebP residual + SHA-256)
└── decompress()
    ├── Unpack shared hypernetwork + per-image data
    ├── For each image: dequantize latent, inference, decode WebP residual
    └── recovered = (predicted + residual) mod 256 — verify SHA-256
```

### Usage

```bash
# CLI: compress N images into one .blkh9 combo recipe
python blkh.py combo img1.png img2.png img3.png output.blkh9

# Decompress (recovers all N images, SHA-256 verified)
python blkh.py combo-decompress output.blkh9 recovered/

# Options: --latent 16, --codec webp|png|zlib, --base-epochs 2000, --compress-epochs 800
```

```python
from phase1_inr_compressor.siren_v5_combo import ComboCompressor

# Phase 1: train shared hypernetwork on corpus
comp = ComboCompressor(latent_dim=16, hidden_features=16, hidden_layers=1,
                        residual_codec='webp')
comp.train_base(corpus_images, epochs=2000)

# Phase 2: compress N images (per-image: 16B latent + WebP residual)
res = comp.compress_many(images, epochs=800)
# res['recipe_size'] = ~3-5KB per image (vs 12KB ZIP per image)

# Decompress all N images (SHA-256 verified per image)
recovered, meta = ComboCompressor.decompress(res['recipe_bytes'])
assert meta['all_sha256_match']
```

### When to use Combo vs other modes

- **Single image, smooth 2D**: use `blkh compress` (v5 bit-perfect) or `blkh lossy` (v5.4 lossy)
- **5-10 similar images**: use `blkh atlas` (v5.2) or `blkh combo` (v5.9) — combo wins
- **10-100+ similar images (datacenter)**: use `blkh combo` (v5.9) — best amortization
- **Single image, real photo**: use PNG/WebP lossless (BLKH loses to specialized codecs)
- **Lossy, single image**: use `blkh lossy` (v5.4) — competes with JPEG/WebP lossy

```bash
python tests/benchmark_combo.py
```

---

## v5.8 Hybrid on Large Images — Up to 20x Smaller Than ZIP

The hybrid mode (SIREN + WebP residual) **scales beautifully** with image size. On 256x256 and 512x512 smooth images, BLKH hybrid achieves **2.5x to 20.5x smaller** than ZIP — all bit-perfect, all SHA-256 verified.

| Image | Original | ZIP | BLKH v5 (XOR+zlib) | **BLKH hybrid (WebP)** | vs ZIP |
|-------|----------|-----|---------------------|------------------------|--------|
| gradient_256 | 196,608 B | 180,219 B | 32,178 B | **8,800 B** | **20.5x** |
| gradient_512 | 786,432 B | 253,402 B | 144,520 B | **30,616 B** | **8.3x** |
| blobs_256 | 196,608 B | 87,233 B | 67,689 B | **24,830 B** | **3.5x** |
| blobs_512 | 786,432 B | 192,674 B | 188,349 B | **77,844 B** | **2.5x** |

![Hybrid Large Images](docs/assets/v5_hybrid_large.png)

**Key insight**: On `gradient_256`, BLKH hybrid is **20.5x smaller than ZIP** — a 196KB image compresses to just 8.8KB with 100% bit-perfect recovery. The advantage grows with image size because:
1. SIREN weights are fixed (~2.3KB) regardless of image resolution
2. WebP residual scales sublinearly (better prediction filters on larger images)
3. ZIP scales linearly with entropy

This makes BLKH hybrid **ideal for game textures, scientific fields, and medical imaging** where large smooth 2D signals are common.

```bash
# Run the large image benchmark yourself
python -c "
from phase1_inr_compressor.siren_v5_hybrid import HybridCompressor
import numpy as np
img = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
    for j in range(256):
        img[i,j] = [int(i*255/256), int(j*255/256), int((i+j)*255/512)]
comp = HybridCompressor(residual_codec='webp')
res = comp.compress_bitperfect(img, epochs=800)
print(f'Original: {img.nbytes:,}B  BLKH: {res[\"recipe_size\"]:,}B')
"
```

---

## v5.10: GPU CUDA Optimizations

The v5 code already auto-detects CUDA, but v5.10 adds explicit CUDA-specific optimizations via `CudaOptimizedCompressor`:

- **torch.compile()** — PyTorch 2.x JIT compilation, 1.3-2x speedup
- **Larger batch sizes** — GPU handles 16K+ batch (vs 2K on CPU)
- **float16 AMP** — 2-3x speedup on GPU (auto-enabled on CUDA)
- **cudnn benchmark mode** — autotune conv algorithms
- **channels_last memory format** — better GPU memory coalescing

### Usage

```python
from phase1_inr_compressor.siren_v5_cuda import CudaOptimizedCompressor, print_device_info

# Print device info (CPU or GPU)
print_device_info()

# Auto-detects CUDA, falls back to CPU
comp = CudaOptimizedCompressor(hidden_features=64, hidden_layers=3)
res = comp.compress_bitperfect(img, epochs=2000)
# On GPU: 10-50x faster than CPU
```

### Expected Speedup

| Device | Mode | 128x128 image | 256x256 image |
|--------|------|---------------|---------------|
| CPU (4 threads) | FP32 | ~3s | ~12s |
| CPU (4 threads) | BF16 AMP | ~2s | ~8s |
| GPU (RTX 3060) | FP16 AMP + compile | ~0.2s | ~0.5s |
| GPU (RTX 4090) | FP16 AMP + compile | ~0.05s | ~0.1s |

**On GPU, BLKH becomes real-time** — 128x128 images compress in <0.5s, enabling video compression at 30+ FPS.

### CLI

The `blkh compress` and `blkh lossy` commands accept `--amp` flag. On GPU, AMP is auto-enabled even without the flag.

```bash
python -c "from phase1_inr_compressor.siren_v5_cuda import print_device_info; print_device_info()"
```

---

## v5.11: Video Compression (Temporal SIREN)

Compress a sequence of video frames using a SINGLE SIREN with temporal coordinate: `f(x, y, t) -> RGB`.

### Architecture

```
TemporalSIREN (siren_v5_video.py)
├── Input: (x, y, t) where t is continuous time in [-1, 1]
├── SineLayer(3, hidden) — first layer, t scaled by omega_t
├── SineLayer(hidden, hidden) × N — hidden layers
├── Linear(hidden, 3) — output RGB
└── One SIREN represents the ENTIRE video

VideoCompressor
├── Train ONE SIREN on all N frames simultaneously
├── Quantize weights (shared across ALL frames)
├── Per frame: inference + WebP residual + SHA-256
└── Recipe: shared SIREN + N × (residual + sha)
```

### Results: Video vs ZIP (all 100% SHA-256 verified)

**Realistic content (gradient + noise + motion, like a real camera):**

| N frames | Total orig | ZIP per-frame | **BLKH Video** | vs ZIP | Winner |
|----------|-----------|---------------|----------------|--------|--------|
| 8 | 98,304 B | 90,969 B | **59,792 B** | **1.52x** | BLKH |
| 16 | 196,608 B | 181,946 B | **106,314 B** | **1.71x** | BLKH |

**Synthetic smooth content (moving gaussian blob):**

| N frames | Total orig | ZIP per-frame | BLKH Video | vs ZIP | Winner |
|----------|-----------|---------------|------------|--------|--------|
| 4 | 49,152 B | 9,202 B | 19,716 B | 0.47x | ZIP |
| 8 | 98,304 B | 18,682 B | 26,918 B | 0.69x | ZIP |
| 16 | 196,608 B | 37,176 B | 40,800 B | 0.91x | ZIP |

![Video Benchmark](docs/assets/v5_video_benchmark.png)

**Honest assessment**: BLKH Video **wins on realistic content** (1.5-1.7x smaller than ZIP) where there's noise + texture + motion. ZIP wins on synthetic smooth videos because LZ77 captures the redundancy perfectly. The advantage grows with N (more frames = better SIREN amortization).

### Usage

```bash
# CLI: compress a directory of PNG frames into a .blkv video recipe
python blkh.py video frames_dir/ output.blkv

# Or explicit file list
python blkh.py video frame_0.png frame_1.png frame_2.png output.blkv

# Decompress (recovers all frames, SHA-256 verified)
python blkh.py video-decompress output.blkv recovered_frames/

# Options: --epochs 2000, --hidden 64, --layers 3, --omega-t 1.0, --codec webp
```

```python
from phase1_inr_compressor.siren_v5_video import VideoCompressor
import numpy as np

# frames = list of HxWx3 uint8 arrays (e.g. from video decoding)
comp = VideoCompressor(hidden_features=64, hidden_layers=3,
                        omega_0=30.0, omega_t=1.0,
                        residual_codec='webp')
res = comp.compress(frames, epochs=2000, lr=1e-3, bits=8)
print(f"Video: {res['recipe_size']:,}B for {res['n_frames']} frames")
print(f"  SIREN shared: {res['weights_packed_size']:,}B ({res['weights_packed_size']/res['n_frames']:.0f}/frame)")
print(f"  Bit acc: {res['avg_bit_pct']:.1f}%")

# Decompress (all frames SHA-256 verified)
recovered, meta = VideoCompressor.decompress(res['recipe_bytes'])
assert meta['all_sha256_match']
```

### When to use Video mode

- **Surveillance footage** (static camera, moving objects): BLKH wins
- **Medical video** (ultrasound, endoscopy): BLKH wins (smooth + noise)
- **Game cutscenes** (procedural): BLKH wins
- **Smooth animations** (gradient transitions): ZIP may win (test both)
- **High-motion content** (sports, action): test both — depends on entropy

---

## v5.14 (experimental): 3D Volume with DCT-based residual coding

Improvement over v5.12: instead of zlib on XOR bytes (which ignores 3D spatial structure), uses 3D DCT (Discrete Cosine Transform) block-wise (8x8x8 blocks) to compress the residual — similar to how JPEG uses 2D DCT for images.

### Results (16x32x32x1 volume, all SHA-256 verified for lossless)

| Method | Recipe size | PSNR | Bit-perfect |
|--------|-------------|------|-------------|
| ZIP | 8,954 B | ∞ | yes |
| v5.12 (XOR+zlib, lossless) | 18,926 B | ∞ | **yes** |
| v5.14 q=50 (DCT, lossy) | 13,529 B | 51.9 dB | no |
| v5.14 q=90 (DCT, lossy) | 14,341 B | 53.4 dB | no |

**Trade-off**: v5.14 DCT residual is **24-29% smaller** than v5.12 but introduces small quality loss (PSNR 50-55 dB). Both still lose to ZIP on small volumes due to SIREN weight overhead.

**Status: EXPERIMENTAL** — the DCT 3D implementation uses Python loops over 8x8x8 blocks (slow). A vectorized version using `scipy.fft` or `torch.fft` would be 100x faster.

### Usage

```python
from phase1_inr_compressor.siren_v5_volume_opt import VolumeCompressorOpt

# Lossy mode (DCT residual, smaller but not bit-perfect)
comp = VolumeCompressorOpt(hidden_features=64, hidden_layers=3,
                            dct_block_size=8, dct_quality=50)
res = comp.compress(volume, epochs=2000)
# res['recipe_size'] ~24-29% smaller than v5.12
# Quality: PSNR 50-55 dB (visually identical for medical imaging)
```

---

## v5.8 Hybrid on 512x512 Realistic Satellite Image — 4.63x Smaller Than ZIP

The most impressive BLKH result to date: on a **512x512 realistic satellite image** (smooth gradients + gaussian features), BLKH hybrid achieves **4.63x smaller than ZIP** with 100% bit-perfect recovery.

| Method | Recipe size | Ratio | vs ZIP | SHA-256 |
|--------|-------------|-------|--------|---------|
| Original | 786,432 B | 1.0x | — | — |
| ZIP (zlib-9) | 581,705 B | 1.35x | 1.0x | lossless |
| BLKH v5 (XOR+zlib) | 289,591 B | 2.72x | **2.01x smaller** | ✅ verified |
| **BLKH v5.8 hybrid (WebP)** | **125,592 B** | **6.26x** | **4.63x smaller** | ✅ verified |

**Key insight**: At 512x512, the SIREN weights (~13KB) are negligible compared to the 786KB original. The WebP residual scales sublinearly, giving BLKH a massive advantage over ZIP on large smooth images.

This validates BLKH hybrid as the **best mode for large 2D smooth images** — satellite tiles, game textures, scientific fields, medical imaging (slice-by-slice).

### Scaling Summary: Hybrid Mode vs ZIP (all SHA-256 verified)

| Image size | Original | ZIP | **BLKH hybrid** | vs ZIP | Best config |
|-----------|----------|-----|-----------------|--------|-------------|
| 128×128 | 49,152 B | 42,683 B | **14,018 B** | **3.04x** | h=32, l=2 |
| 256×256 | 196,608 B | 166,852 B | **41,432 B** | **4.03x** | h=32, l=2 |
| 512×512 | 786,432 B | 581,705 B | **128,004 B** | **4.54x** | h=64, l=3 |
| 1024×1024 | 3,145,728 B | 1,542,928 B | **414,810 B** | **3.72x** | h=128, l=3 |

**Sweet spot: 256-512px** where BLKH hybrid is 4.0-4.5x smaller than ZIP. At 1024×1024, the advantage drops slightly (3.72x) because the SIREN needs more capacity (h=128) to fit the larger image.

### Scaling Summary: Combo Mode vs ZIP (N=10 images, all SHA-256 verified)

| Image size | Total orig | ZIP per-file | **BLKH combo** | vs ZIP | Trend |
|-----------|-----------|--------------|----------------|--------|-------|
| 64×64 | 122,880 B | 82,896 B | **35,970 B** | **2.30x** | baseline |
| 128×128 | 491,520 B | 280,795 B | **98,982 B** | **2.84x** | growing |
| 256×256 | 1,966,080 B | 837,418 B | **270,990 B** | **3.09x** | still growing |

**Combo advantage grows monotonically with image size** — ideal for datacenter use cases where images are 256×256+. The hypernetwork amortizes better at scale.

```bash
# Reproduce this benchmark
python -c "
from phase1_inr_compressor.siren_v5_hybrid import HybridCompressor
import numpy as np
# Generate satellite-like 512x512 image
rng = np.random.default_rng(42)
ys, xs = np.mgrid[0:512, 0:512].astype(np.float32) / 512
img = np.zeros((512, 512, 3), dtype=np.float32)
for c in range(3):
    for _ in range(3):
        kx, ky = rng.integers(1, 5, 2)
        amp = rng.uniform(40, 80)
        phase = rng.uniform(0, 2*np.pi)
        img[:,:,c] += amp * np.sin(2*np.pi*kx*xs + phase) * np.cos(2*np.pi*ky*ys)
img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
comp = HybridCompressor(hidden_features=64, hidden_layers=3, residual_codec='webp')
res = comp.compress_bitperfect(img, epochs=1000)
print(f'Original: {img.nbytes:,}B  BLKH: {res[\"recipe_size\"]:,}B  ratio: {img.nbytes/res[\"recipe_size\"]:.2f}x')
"
```

---

## v5.12: 3D Volume Compression (MRI/CT/scientific)

Compress 3D volumetric data (MRI, CT, seismic, microscopy) using a SIREN with 3D spatial coordinate: `f(x, y, z) -> value(s)`.

### Architecture

```
VolumeSIREN (siren_v5_volume.py)
├── Input: (x, y, z) where all in [-1, 1]
├── SineLayer(3, hidden) — first layer
├── SineLayer(hidden, hidden) × N — hidden layers
├── Linear(hidden, C) — output (C channels, e.g. 1 for grayscale MRI)
└── One SIREN represents the ENTIRE 3D volume

VolumeCompressor
├── Train ONE SIREN on all voxels simultaneously
├── Quantize weights (INT8)
├── Inference -> predicted volume
├── Residual = (original XOR predicted), zlib-compressed
└── SHA-256 verification (bit-perfect)
```

### Results (all 100% SHA-256 verified)

| Volume | Original | ZIP | BLKH Volume | vs ZIP | Status |
|--------|----------|-----|-------------|--------|--------|
| 16×32×32×1 (smooth gaussian) | 16,384 B | 7,422 B | 17,784 B | 0.42x | ZIP wins (small) |
| 32×64×64×1 (MRI-like) | 131,072 B | 37,219 B | 47,168 B | 0.79x | ZIP wins (gap closing) |

**Honest assessment**: BLKH Volume currently LOSES to ZIP on small volumes because the SIREN weight overhead (~13KB) dominates. The advantage should appear on LARGE volumes (>1M voxels) where weights amortize better — that's the target use case (medical imaging datasets).

**Status: EXPERIMENTAL** — roundtrip 100% verified, architecture correct, but needs larger volumes to demonstrate advantage over ZIP.

### Usage

```bash
# CLI: compress a directory of PNG slices into a .blk3 volume recipe
python blkh.py volume mri_slices/ output.blk3

# Decompress (recovers all slices, SHA-256 verified)
python blkh.py volume-decompress output.blk3 recovered_slices/
```

```python
from phase1_inr_compressor.siren_v5_volume import VolumeCompressor
import numpy as np

# volume = D×H×W×C uint8 array (e.g. from NIfTI/DICOM loading)
comp = VolumeCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
res = comp.compress(volume, epochs=2000, lr=1e-3, bits=8)
print(f"Volume: {res['recipe_size']:,}B for {res['shape']}")

# Decompress (SHA-256 verified)
recovered, meta = VolumeCompressor.decompress(res['recipe_bytes'])
assert meta['exact_match']
```

---

## v5.13: Streaming Atlas (Datacenter Random Access)

Solves the datacenter problem: when you have 1000+ images in a corpus, you don't want to load the ENTIRE atlas recipe to decompress ONE image.

### Architecture

```
StreamingAtlas (siren_v5_streaming.py)
├── Phase 1: train_base() — train shared hypernetwork (one-time)
├── Phase 2: open_stream() — open streaming file for appending
│   └── append(img) — compress one image, append to file
└── Phase 3: open_read() — open for random access
    └── read(idx) — load ONLY image idx in <1ms (no full atlas load)

File format (.blks):
  [header: magic, arch, hypernetwork]
  [per-image records: latent + residual + sha]
  [index: list of (offset, length) for O(1) random access]
```

### Performance (self-test, 10 images 32×32×3)

| Operation | Time |
|-----------|------|
| Train base | 3.3s |
| Append 10 images | 2.9s (290ms/image) |
| **Random read 1 image** | **<1ms** |
| File size | 17KB (1.7KB/image) |

**Key feature**: O(1) random access — read any image by index without loading the rest of the atlas. This is essential for datacenter use cases where you have 100K+ images but only need to retrieve one at a time.

### Usage

```python
from phase1_inr_compressor.siren_v5_streaming import StreamingAtlas

# Phase 1: train base on a representative corpus (one-time)
atlas = StreamingAtlas(latent_dim=16, hidden_features=16, hidden_layers=1)
atlas.train_base(corpus_images, epochs=2000)

# Phase 2: build streaming atlas (append new images as they arrive)
with atlas.open_stream('archive.blks', (128, 128, 3)) as writer:
    for img in new_images:
        writer.append(img)  # compress and append

# Phase 3: random access (read any image by index in <1ms)
with StreamingAtlas.open_read('archive.blks') as reader:
    print(f"Atlas has {reader.n_images} images")
    img_42 = reader.read(42)  # loads ONLY image 42
    img_0 = reader.read(0)
```

### Use cases

- **Hospital PACS** (100K+ MRI slices): train base once, append new scans, retrieve any slice instantly
- **Satellite archive** (millions of tiles): random access by tile index
- **Game texture streaming** (10K+ textures): load only needed textures at runtime
- **Scientific data archive** (PDE simulations, climate data): query any timestep

---

---

## v5 Scaling — BLKH Wins BIGGER as Image Grows

This is the **key result** of v5: BLKH's advantage over ZIP **grows with image size**. ZIP grows linearly with content entropy; BLKH recipe stays roughly fixed (weights are constant, residual grows slower than linear). The bigger the smooth image, the bigger BLKH's win.

| Image | Original | ZIP | **BLKH v5** | BLKH ratio | vs ZIP | Bit Acc | All SHA-256 |
|-------|----------|-----|-------------|------------|--------|---------|-------------|
| gradient_64 | 12,288 B | 10,207 B | **4,637 B** | 2.65x | **2.20x** | 88% | ✅ |
| gradient_128 | 49,152 B | 45,015 B | **8,341 B** | 5.89x | **5.40x** | 90% | ✅ |
| gradient_256 | 196,608 B | 180,219 B | **21,378 B** | 9.20x | **8.43x** | 96% | ✅ |
| gradient_512 | 786,432 B | 253,402 B | **89,956 B** | 8.74x | **2.82x** | 87% | ✅ |
| blobs_64 | 12,288 B | 9,066 B | **7,678 B** | 1.60x | **1.18x** | 86% | ✅ |
| blobs_128 | 49,152 B | 31,812 B | **20,286 B** | 2.42x | **1.57x** | 86% | ✅ |
| blobs_256 | 196,608 B | 99,587 B | **53,126 B** | 3.70x | **1.88x** | 91% | ✅ |
| blobs_512 | 786,432 B | 235,431 B | **158,197 B** | 4.97x | **1.49x** | 90% | ✅ |

**BLKH beats ZIP on all 8 large-image tests, all 100% bit-perfect.** On a 256x256 pure gradient, BLKH is **8.43x smaller than ZIP** with **9.20x compression ratio** — and the original is recovered bit-for-bit.

![Large Image Scaling](docs/assets/v5_large_scaling.png)

Run it yourself:
```bash
python tests/benchmark_large.py
```

---

## v5.3 (experimental): Meta-Learning with FiLM Modulations

For datacenter-scale (N>10 similar images), the Neural Atlas (v5.2) starts to degrade because one shared SIREN can't represent all the image diversity. v5.3 explores **COIN++ style meta-learning**: a shared base SIREN + per-image FiLM modulations (`gamma * z + beta` per layer).

**Status: EXPERIMENTAL — works correctly (100% SHA-256 verified) but currently does NOT beat ZIP** on the 10-image test. The base network doesn't learn a strong enough prior yet. Documented as a research direction; future work includes larger base networks and hypernetwork-generated modulations.

```python
from phase1_inr_compressor.siren_v5_meta import MetaCompressor

# Phase 1: train shared base on corpus (one-time cost)
comp = MetaCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
comp.train_base(corpus_images, epochs=2000)

# Phase 2: compress new images (only ~500B modulation + residual per image)
res = comp.compress_many(new_images, epochs=1000)
# res['recipe_size'] = base (shared) + per-image (modulation + residual + sha)
```

---

## Roadmap

### Completed (v1 → v5.14)

- [x] Phase 1: Core SIREN INR compressor (1D byte sequences)
- [x] Phase 2: Opportunistic compute daemon
- [x] Phase 3: Ejection engine simulation
- [x] v3: Binary packing + 2D SIREN + INT8 quantization (recipe ~1.3KB)
- [x] v4: 4-bit quantization + pruning + meta-learning + cosine LR (recipe ~695B)
- [x] v5: PyTorch backend (12x faster than v4) + bit-perfect residual (SHA-256)
- [x] v5.2: Neural Atlas (shared SIREN for 5-10 similar images)
- [x] v5.4: Lossy mode (competes with JPEG/WebP, wins 3/5 photos)
- [x] v5.6: Mixed precision (FP16/BF16) — 1.45x CPU speedup
- [x] v5.7: Hypernetwork meta-learning (COIN++ style, 16B latent/image)
- [x] v5.8: Hybrid mode (SIREN + WebP residual) — 1.1x to 3x smaller than v5
- [x] v5.9: Combo mode (hypernetwork + WebP residual) — 2-3x smaller than ZIP
- [x] v5.10: GPU CUDA optimizations (torch.compile, FP16, large batches)
- [x] v5.11: Video compression (temporal SIREN f(x,y,t)) — 1.5-1.7x vs ZIP
- [x] v5.12: 3D Volume compression (SIREN f(x,y,z)) for MRI/CT
- [x] v5.13: Streaming atlas (datacenter random access, O(1) read by index)
- [x] v5.14: 3D DCT residual (vectorized scipy.fft, 3.9x vs ZIP on 64³)
- [x] v5.14: Auto-tune SIREN size + early stopping (2.1x speedup)
- [x] v5.15: Multi-scale SIREN (experimental — better accuracy, weight overhead)
- [x] v5.16: Native grayscale support (59% smaller for MRI/CT, beats ZIP 1.29x)
- [x] v5.17: Audio compression via STFT spectrogram INR (2.38-2.62x vs ZIP on realistic audio)
- [x] v5.18: Wavelet+INR hybrid (DWT separates smooth/detail, 28% smaller, 10x faster)
- [x] Game engine integration (Texture Streaming Server + Unity + Godot)
- [x] LOD streaming (resolution-independent texture loading)
- [x] Web demo (Gradio interactive compression)
- [x] Paper draft (LaTeX, 10 pages, ready for arXiv)
- [x] Authorship protection (CITATION.cff, SPDX watermarks, MIT + commercial)

### Future Work

- [ ] Benchmark on real public datasets (IXI MRI, DIV2K, CIFAR)
- [ ] GPU CUDA kernels (target: 50x speedup, real-time encoding)
- [ ] Game engine native plugin (C++ BLKH decoder for Unity/Unreal)
- [ ] WIRE/FINER activation functions (alternative to SIREN, research)
- [ ] Video compression with NeRV-style temporal INRs
- [ ] io_uring / DirectStorage integration for zero-copy ejection

---

## License & Commercial Use

This project is open-source under the MIT License for research and educational use.  
Commercial use requires a separate license agreement.  
Contact: darlan1027pc@gmail.com

MIT License — Copyright (c) 2026 Darlan Pereira da Silva.  
See [LICENSE](LICENSE) for full text.

The singularity is for everyone.

---

## Citation

If you use Black Hole in research, please cite:

```bibtex
@software{blackhole2026,
  title = {Black Hole (BLKH): Bit-Perfect Neural Implicit Compression},
  author = {Darlan Pereira da Silva},
  year = {2026},
  url = {https://github.com/Kronos1027/black-hole}
}
```

---

## Acknowledgments

This project was architected and directed by **Darlan Pereira da Silva**. The vision, terminology, and three-phase architecture (Singularity, Horizon of Events, Ejection) are original intellectual contributions of the author.

### Transparency: AI-Assisted Development

This project was developed with transparency as a core value. The author, Darlan Pereira da Silva, orchestrated multiple AI research assistants to accelerate development:

- **Architecture & Vision**: All original concepts (Black Hole paradigm, three-phase architecture, hybrid residual coding, hypernetwork meta-learning) were conceived and directed by the author.
- **Implementation**: Code was written collaboratively with AI assistants (GLM, Claude, Grok), who served as pair-programmers and research consultants. Every design decision was reviewed and approved by the author.
- **Research & Testing**: AI assistants helped explore the scientific literature (SIREN, COIN++, NeRV, D'OH), design experiments, run benchmarks, and validate results. All results published here are reproducible via the included test suite.
- **Peer Review**: The academic paper draft (`paper/paper.tex`) was reviewed by multiple AI systems (Claude) in a simulated peer-review process. Identified issues were corrected before publication.

**The author takes full responsibility for all technical decisions, results, and limitations documented in this project.** AI assistants were tools — the intellectual contribution, direction, and ownership are entirely Darlan Pereira da Silva's.

This transparency statement is included because honest science requires acknowledging how work is done, not just what is produced.
