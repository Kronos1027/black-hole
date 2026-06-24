# BLKH Web Demo — HuggingFace Spaces Deployment

This directory contains everything needed to deploy the BLKH web demo to HuggingFace Spaces.

## Quick Deploy

1. Create a new Space at https://huggingface.co/spaces → New Space
2. Choose "Gradio" SDK
3. Upload these files:
   - `app.py` (copy of `blkh_web_demo.py`)
   - `requirements.txt`
4. The Space will auto-build and deploy

## Requirements

```
numpy>=1.24.0
Pillow>=10.0.0
torch>=2.0.0
scipy>=1.10.0
PyWavelets>=1.5.0
zstandard>=0.22.0
brotli>=1.0.0
gradio>=4.0.0
pillow-avif-plugin>=1.0.0
```

## Features

The web demo provides 6 compression modes:

1. **Instant (~0.5s)** — Hybrid SIREN + PNG residual (bit-perfect)
2. **Turbo (~1s)** — Hybrid SIREN + WebP residual (bit-perfect)
3. **Quality (~6s)** — Hybrid SIREN + WebP residual, 800 epochs (bit-perfect)
4. **Wavelet v3** — TRUE bit-perfect wavelet+float16+brotli (fast, smooth images)
5. **Photo v5.21** — YCbCr 4:2:0 + brotli (lossy, 2-2.7x smaller than PNG on photos)
6. **DCT v5.22** — JPEG-like DCT + brotli (lossy, 20-50x smaller than PNG, max compression)

## Example Output

```
BLKH DCT v5.22 (Maximum Compression, Lossy) Results
==================================================
  Original:           49,152 bytes
  ZIP (zlib-9):       31,174 bytes  (1.58x)
  PNG (lossless):     27,097 bytes  (1.81x)
  WebP (lossless):    22,134 bytes  (2.22x)
  BLKH DCT:            1,153 bytes  (42.63x)

  BLKH vs ZIP:     27.04x ✓ BLKH wins
  BLKH vs PNG:     23.50x ✓ BLKH wins
  BLKH vs WebP:    19.19x ✓ BLKH wins

  Best format:     BLKH
  PSNR:            29.9 dB (lossy, quality=0.9)
  Encoding time:   0.04s
```

## Performance Comparison

| Mode | Use Case | Compression | Speed |
|------|----------|-------------|-------|
| Wavelet v3 | Smooth lossless | 2-3x vs ZIP | 0.4-2s |
| Photo v5.21 | Natural photos | 2-4x vs PNG | 0.02-0.1s |
| DCT v5.22 | Max compression | 20-50x vs PNG | 0.04-0.4s |
| DCT q=0.5 | Extreme compression | 50-165x vs PNG | 0.04s |

## License

MIT — free for research and education. Commercial use requires separate license.
Contact: darlan1027pc@gmail.com

## Author

Darlan Pereira da Silva (Kronos1027)
GitHub: https://github.com/Kronos1027/black-hole
