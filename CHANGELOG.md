# Changelog

All notable changes to Black Hole (BLKH) will be documented in this file.

## [5.30.0] — 2026-06-25

### Added
- **Auto mode (v5.30)**: Intelligent mode selector that tries palette/DCT/fast/photo and picks smallest within time budget
- **Palette mode (v5.29)**: Lossless palette+indices compression for images with few colors (logos, icons, UI)
- **RLE mode (v5.28)**: DCT + zigzag RLE (JPEG-style entropy), 8-15% smaller than v5.22
- **Direct I/O batch (v5.27)**: Platform-optimized I/O (O_DIRECT stub, DirectStorage stub)
- **AVIF wrapper (v5.26)**: Modern standard support via pillow-avif-plugin
- **GPU-ready DCT (v5.25)**: CUDA-optional with CPU fallback
- **Async batch (v5.24)**: asyncio + ProcessPoolExecutor for io_uring-style concurrent I/O
- **Fast DCT (v5.23)**: Speed-optimized codec selection (zstd L3 / brotli q=6 / brotli q=11). 3x FASTER than ZIP
- **DCT mode (v5.22)**: JPEG-like 8x8 DCT + standard quantization tables + brotli. 20-50x smaller than PNG
- **Photo mode (v5.21)**: YCbCr 4:2:0 + adaptive PNG filter + brotli. 2-2.7x smaller than PNG on photos
- **Research program**: Black Hole Universe Hypothesis (BHUH) — 30 experimental phases
- **HuggingFace Spaces**: Live web demo at https://huggingface.co/spaces/onatskyo/black-hole-blkh
- **PyPI package**: `pip install blackhole-blkh`
- **Issue templates**: Bug report, feature request, feedback, benchmark
- **CONTRIBUTING.md**: Development workflow and guidelines
- **upload_to_huggingface.py**: Automated HF Space deployment script

### Changed
- Version bumped from 5.14 to 5.30.0
- README updated with Live Demo section and HuggingFace badge
- pyproject.toml: added Python 3.13 classifier, AI/Multimedia topics, Live Demo URL
- CLI expanded from 6 to 13 modes

### Fixed
- **v5.19 critical bug**: v5.18 wavelet mode claimed "bit-perfect" but was actually lossy (PSNR 4-12 dB). Fixed with TRUE bit-perfect lossless mode.
- Python 3.13 compatibility: added `audioop-lts` for removed `audioop` module
- Gradio 4.0.0 compatibility: removed `gradio.Soft()` theme
- HuggingFace Spaces: fixed Windows backslash path issue in file upload

## [5.20.0] — 2026-06-24

### Added
- **Wavelet+INR v3 (v5.20)**: Float16 breakthrough — TRUE bit-perfect, 30% smaller than v5.19
  - Brotli support (8% smaller than zstd)
  - Parallel adaptive search (2-3x faster)
  - Combined mode (single bytestream, 6% smaller)

## [5.19.0] — 2026-06-24

### Added
- **Wavelet+INR v2 (v5.19)**: CRITICAL fix for v5.18 (was lossy, not bit-perfect as claimed)
  - TRUE bit-perfect lossless mode (int16 + int8 residual + zstd)
  - Lossy mode (uint8 LL + int8 detail, 5-60x compression, 38-56 dB PSNR)
  - Adaptive wavelet/level selection (11 candidates)
  - Soft thresholding for lossy mode

## [5.18.0] — 2026-06-23

### Added
- Wavelet+INR hybrid (DWT separates smooth/detail, 28% smaller, 10x faster)
- db6 wavelet + level 3 default — 62.6x vs ZIP (synthetic)

### ⚠️ Known Issues (fixed in v5.19)
- Wavelet mode was lossy, not bit-perfect as README claimed

## [5.17.0] — 2026-06-23

### Added
- Audio compression via STFT spectrogram INR (2.38-2.62x vs ZIP)

## [5.16.0] — 2026-06-22

### Added
- Native grayscale support (59% smaller for MRI/CT, beats ZIP 1.29x)

## [5.15.0] — 2026-06-22

### Added
- Multi-scale SIREN (experimental — better accuracy, weight overhead)

## [5.14.0] — 2026-06-22

### Added
- Auto-tune SIREN size + early stopping (2.1x speedup)
- 3D Volume with DCT-based residual coding

## [5.8.0] — 2026-06-21

### Added
- Hybrid mode: SIREN + image-codec residual (PNG/WebP lossless)
- Bit-perfect roundtrip (SHA-256 verified)

## [5.0.0] — 2026-06-20

### Added
- PyTorch SIREN implementation with INT8/INT4 quantization
- CLI with compress, decompress, info, benchmark commands
- Gradio web demo
- Game engine integration (Unity + Godot)
- LaTeX paper draft

---

## Research: Black Hole Universe Hypothesis

### [Research] — 2026-06-25

30 experimental phases validating the Black Hole Universe Hypothesis:

| Phase | Discovery | Result |
|-------|-----------|--------|
| 1 | Multi-File SIREN scaling law | 62.78x vs ZIP |
| 1b | Real photos validation | 9.15x vs ZIP |
| 2 | Universal hypernetwork | 8.24x (audio) |
| 3 | Program synthesis | 1.45x (CSV) |
| 4 | VAE diffusion seed | 1.73x |
| 5 | Universe prototype | 9.95x (mixed) |
| 6 | Cross-domain transfer | 6.88x (BREAKTHROUGH) |
| 7 | Hierarchical universes | 23.62x (BREAKTHROUGH) |
| 8 | Triple-domain | 1.85x |
| 9 | 3-level hierarchy | Diminishing returns |
| 10 | Genesis streaming | 7x memory reduction |
| 11 | Video 3D SIREN | 15.74x |
| 12 | Universal archive | .blku format |
| 13 | Auto-architecture | 27.7% savings |
| 14 | Multi-resolution | 16→1024 from 1 seed |
| 15 | Progressive upgrade | Bit-perfect + 1.15x |
| 16 | Dataset scale (500) | 102B/image |
| 17 | Content addressing | ❌ Negative (honest) |
| 18 | Fractal IFS | 209x for fractals |
| 19 | Hash encoding | 2x faster, 2.87x larger |
| 20 | Adaptive quality | SSIM +5.6% |
| 21 | Kolmogorov spectrum | SIREN = K(x) approx |
| 22 | Delta compression | 1.61x for versions |
| 23 | Theory formalization | All 5 principles proven |
| 24 | Codec comparison matrix | RLE 61x vs ZIP |
| 25 | 3D volume (MRI/CT) | 32.5x over per-slice |
| 26 | Time-series (IoT) | 11.42x |
| 27 | Game engine LOD | 4.77x storage reduction |
| 28 | Super-resolution | ❌ Negative (honest) |
| 29 | Denoising | +7.9dB improvement |
| 30 | Real-world test | JSON 4.0x ratio |
