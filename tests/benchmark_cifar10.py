#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""BLKH v5.23 — Large-scale CIFAR-10 benchmark (10000 real images)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))

import pickle
import numpy as np
import zlib
import io
import time
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from siren_v5_fast import FastDCTCompressor
from siren_v5_photo import PhotoCompressor
from siren_v5_dct import DCTCompressor


def load_cifar10(path='/tmp/cifar-10-batches-py'):
    """Load CIFAR-10 test batch (10000 images, 32x32x3)."""
    test_path = os.path.join(path, 'test_batch')
    if not os.path.exists(test_path):
        return None
    with open(test_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images


def benchmark_mode(images, mode_name, comp_class, kwargs, sample_size=200):
    """Benchmark a single mode on a sample of images."""
    N = len(images)
    sample = min(sample_size, N)
    indices = np.linspace(0, N-1, sample).astype(int)

    t0 = time.time()
    total_size = 0
    total_psnr = 0
    for i in indices:
        img = images[i]
        comp = comp_class(**kwargs)
        res = comp.compress(img, verbose=False)
        rec, _ = comp_class.decompress(res['recipe_bytes'])
        mse = np.mean((img.astype(float) - rec.astype(float))**2)
        psnr = 10*np.log10(255**2 / max(mse, 1e-10))
        total_size += res['recipe_size']
        total_psnr += psnr
    dt = time.time() - t0

    avg_psnr = total_psnr / sample
    avg_size = total_size / sample
    full_size = avg_size * N
    throughput = sample * images[0].nbytes / dt / 1024 / 1024

    return {
        'mode': mode_name,
        'avg_size': avg_size,
        'avg_psnr': avg_psnr,
        'time_per_img_ms': dt / sample * 1000,
        'full_dataset_mb': full_size / 1024 / 1024,
        'throughput_mbs': throughput,
    }


def main():
    print("=" * 95)
    print("BLKH v5.23 — Large-Scale CIFAR-10 Benchmark (10000 real 32x32 images)")
    print("=" * 95)

    images = load_cifar10()
    if images is None:
        print("CIFAR-10 not found. Download from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        return

    N = len(images)
    print(f"\nDataset: CIFAR-10 test batch")
    print(f"  Images: {N}")
    print(f"  Shape: {images.shape[1:]} per image")
    print(f"  Total size: {images.nbytes/1024/1024:.1f} MB")

    # ZIP baseline
    print(f"\n--- Baseline: ZIP (zlib level 9) ---")
    t0 = time.time()
    zip_total = 0
    for i in range(N):
        zip_total += len(zlib.compress(images[i].tobytes(), 9))
    zip_time = time.time() - t0
    print(f"  Total: {zip_total:,}B ({zip_total/1024/1024:.1f}MB)")
    print(f"  Time: {zip_time:.1f}s ({images.nbytes/zip_time/1024/1024:.1f} MB/s)")
    print(f"  Per image: {zip_total/N:.0f}B")

    # Test all modes
    print(f"\n--- BLKH modes (sampling 200 images for speed) ---")
    print(f"{'Mode':<25} {'Avg Size':>10} {'PSNR':>8} {'Time/img':>10} {'Full Dataset':>14} {'vs ZIP':>8} {'Throughput':>12}")
    print("-" * 100)

    modes = [
        ('v5.23 fast q=0.9', FastDCTCompressor, {'quality': 0.9, 'speed': 'fast'}),
        ('v5.23 balanced q=0.9', FastDCTCompressor, {'quality': 0.9, 'speed': 'balanced'}),
        ('v5.22 DCT q=0.9', DCTCompressor, {'quality': 0.9, 'codec': 'brotli'}),
        ('v5.22 DCT q=0.5', DCTCompressor, {'quality': 0.5, 'codec': 'brotli'}),
        ('v5.21 photo', PhotoCompressor, {'subsampling': '420', 'codec': 'brotli'}),
    ]

    results = []
    for name, cls, kwargs in modes:
        r = benchmark_mode(images, name, cls, kwargs, sample_size=200)
        r['zip_total'] = zip_total
        r['vs_zip'] = zip_total / (r['full_dataset_mb'] * 1024 * 1024)
        results.append(r)
        print(f"{r['mode']:<25} {r['avg_size']:>8.0f}B {r['avg_psnr']:>6.1f}dB "
              f"{r['time_per_img_ms']:>8.1f}ms {r['full_dataset_mb']:>11.1f}MB "
              f"{r['vs_zip']:>7.2f}x {r['throughput_mbs']:>9.1f}MB/s")

    # Find best mode for different criteria
    print(f"\n--- Best modes by criteria ---")
    fastest = min(results, key=lambda x: x['time_per_img_ms'])
    smallest = min(results, key=lambda x: x['full_dataset_mb'])
    best_psnr = max(results, key=lambda x: x['avg_psnr'])
    best_ratio = max(results, key=lambda x: x['vs_zip'])

    print(f"  Fastest: {fastest['mode']} ({fastest['time_per_img_ms']:.1f}ms/img, {fastest['throughput_mbs']:.1f}MB/s)")
    print(f"  Smallest: {smallest['mode']} ({smallest['full_dataset_mb']:.1f}MB)")
    print(f"  Best PSNR: {best_psnr['mode']} ({best_psnr['avg_psnr']:.1f}dB)")
    print(f"  Best ratio vs ZIP: {best_ratio['mode']} ({best_ratio['vs_zip']:.2f}x smaller)")

    print(f"\n--- Summary ---")
    print(f"  BLKH v5.23 fast is {zip_time / (fastest['time_per_img_ms'] * N / 1000):.1f}x faster than ZIP")
    print(f"  BLKH smallest is {zip_total / (smallest['full_dataset_mb'] * 1024 * 1024):.1f}x smaller than ZIP")
    print(f"  ZIP speed: {images.nbytes/zip_time/1024/1024:.1f} MB/s")
    print(f"  BLKH fast speed: {fastest['throughput_mbs']:.1f} MB/s")


if __name__ == '__main__':
    main()
