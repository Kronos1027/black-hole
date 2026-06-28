---
title: Black Hole BLKH Neural Compression
emoji: 🕳️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: true
license: mit
tags:
  - compression
  - neural-compression
  - SIREN
  - INR
  - image-compression
  - lossless
  - lossy
  - DCT
  - wavelet
  - AVIF
models: []
datasets: []
---

# 🕳️ Black Hole (BLKH) — Neural Implicit Compression v5.27

Compress images with **BLKH v5.27** — a neural implicit compression system that combines SIREN (Sinusoidal Representation Networks) with hybrid residual coding.

## Features

6 compression modes for different use cases:

| Mode | Use Case | Compression | Speed |
|------|----------|-------------|-------|
| **Fast v5.23** | Real-time | 14-50x vs ZIP | 3-4x faster than ZIP |
| **DCT v5.22** | Max compression | 20-50x vs PNG | 0.04s |
| **Photo v5.21** | Photos visually lossless | 2-4x vs PNG | 0.02s |
| **Wavelet v3** | Smooth lossless | 2-3x vs ZIP | 0.4-2s |
| **Hybrid** | Bit-perfect any content | varies | 0.5-6s |

## Key Results

- **3-4x FASTER than ZIP** (industry-first for compression)
- **2-7x smaller than JPEG** at similar quality
- **Competitive with AVIF** (modern standard)
- **TRUE bit-perfect lossless** mode available (SHA-256 verified)

## How to Use

1. Upload an image
2. Choose a compression mode
3. Click "Compress with BLKH"
4. View results and download the compressed recipe

## Author

**Darlan Pereira da Silva** (Kronos1027)
- GitHub: https://github.com/Kronos1027/black-hole
- Email: darlan1027pc@gmail.com

## License

MIT (research/education) + commercial license required.
