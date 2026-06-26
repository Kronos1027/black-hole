#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""BLKH v5.21 — Final comprehensive benchmark across all modes and content types."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))

import numpy as np
import zlib
import io
import time
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from siren_v5_wavelet_v3 import WaveletINRCompressorV3
from siren_v5_photo import PhotoCompressor


def make_smooth(size, seed=42):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(max(3, size//100)):
            kx, ky = rng.integers(1, max(5, size//100), 2)
            amp = rng.uniform(40, 80)
            phase = rng.uniform(0, 2*np.pi)
            img[:, :, c] += amp * np.sin(2*np.pi*kx*xs + phase) * np.cos(2*np.pi*ky*ys)
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


def run_benchmark():
    print("=" * 95)
    print("BLKH v5.21 — Final Comprehensive Benchmark")
    print("=" * 95)
    print()

    # Test cases: (name, image, is_photo)
    test_cases = [
        ('Smooth 128x128', make_smooth(128), False),
        ('Smooth 256x256', make_smooth(256), False),
        ('Smooth 512x512', make_smooth(512), False),
    ]

    # Add real photos if available
    photos_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'sample_photos')
    if os.path.exists(photos_dir):
        for fname in sorted(os.listdir(photos_dir)):
            if fname.endswith('.png'):
                path = os.path.join(photos_dir, fname)
                img = np.array(Image.open(path).convert('RGB'))
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                test_cases.append((f'Photo {fname}', img, True))

    for name, img, is_photo in test_cases:
        print(f"\n--- {name} ({img.shape[0]}x{img.shape[1]}x{img.shape[2]}) ---")
        orig = img.nbytes
        zip_sz = len(zlib.compress(img.tobytes(), 9))
        png_buf = io.BytesIO()
        Image.fromarray(img).save(png_buf, format='PNG', optimize=True)
        png_sz = png_buf.tell()
        print(f"  Original: {orig:,}B  ZIP: {zip_sz:,}B  PNG: {png_sz:,}B")

        # v5.20 wavelet v3 (lossless)
        try:
            t0 = time.time()
            comp = WaveletINRCompressorV3(wavelet='auto', level='auto', lossless=True,
                                            codec='brotli', combined=True, parallel=True)
            res = comp.compress(img, verbose=False)
            dt = time.time() - t0
            rec, meta = WaveletINRCompressorV3.decompress(res['recipe_bytes'])
            sha_ok = '✅' if meta['sha256_match'] else '❌'
            print(f"  v5.20 wavelet v3 (LOSSLESS): {res['recipe_size']:>8,}B  "
                  f"vsZIP={zip_sz/res['recipe_size']:.2f}x  vsPNG={png_sz/res['recipe_size']:.2f}x  "
                  f"SHA={sha_ok}  {dt:.2f}s")
        except Exception as e:
            print(f"  v5.20 wavelet v3: ERROR {e}")

        # v5.21 photo (lossy)
        try:
            t0 = time.time()
            comp = PhotoCompressor(subsampling='420', codec='brotli')
            res = comp.compress(img, verbose=False)
            dt = time.time() - t0
            rec, meta = PhotoCompressor.decompress(res['recipe_bytes'])
            mse = np.mean((img.astype(float) - rec.astype(float))**2)
            psnr = 10*np.log10(255**2 / max(mse, 1e-10))
            print(f"  v5.21 photo (LOSSY):          {res['recipe_size']:>8,}B  "
                  f"vsZIP={zip_sz/res['recipe_size']:.2f}x  vsPNG={png_sz/res['recipe_size']:.2f}x  "
                  f"PSNR={psnr:.1f}dB  {dt:.2f}s")
        except Exception as e:
            print(f"  v5.21 photo: ERROR {e}")

    print()
    print("=" * 95)
    print("Use case guide:")
    print("  • Smooth synthetic (gradients, satellite, medical): 'blkh wavelet3 --parallel --combined'")
    print("  • Natural photos: 'blkh photo' (lossy, 2-4x smaller than PNG)")
    print("  • Bit-perfect on photos: 'blkh compress --auto-tune --amp --patience 5'")
    print("=" * 95)


if __name__ == '__main__':
    run_benchmark()
