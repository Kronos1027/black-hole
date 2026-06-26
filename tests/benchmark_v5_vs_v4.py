#!/usr/bin/env python3
"""
benchmark_v5_vs_v4.py — Compare v4 (numpy) vs v5 (PyTorch) vs ZIP
================================================================
Measures:
    - Compression ratio
    - Encode time (training)
    - Bit accuracy
    - SHA-256 roundtrip integrity
    - Speedup v5 vs v4

Goal: prove v5 is faster than v4 while keeping identical bit-perfect quality.
"""
import sys
import os
import time
import zlib
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
import numpy as np
import torch
torch.set_num_threads(4)


def make_smooth_image(size, seed=42):
    """Smooth 2D image — case where SIREN beats ZIP."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        cy, cx = rng.uniform(size * 0.2, size * 0.8, 2)
        sigma = rng.uniform(size * 0.1, size * 0.25)
        amp = rng.uniform(80, 200)
        img[:, :, c] = amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_pure_gradient(size):
    """Pure gradient — best case for SIREN."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            img[i, j] = [int(i * 255 / size), int(j * 255 / size), int((i + j) * 255 / (2 * size))]
    return img


def benchmark_v5(img, epochs=1500, bits=8, batch_size=2048):
    from siren_v5_torch import ImageINRv5
    # Use SAME architecture as v4 for fair comparison (hidden=32, layers=2)
    comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
    t0 = time.time()
    res = comp.compress_bitperfect(img, epochs=epochs, lr=1e-3,
                                    bits=bits, prune_threshold=0.0,
                                    batch_size=batch_size, verbose=False)
    dt = time.time() - t0
    recon, meta = ImageINRv5.decompress(res['recipe_bytes'])
    return {
        'recipe_size': res['recipe_size'],
        'weights_size': res['weights_packed_size'],
        'residual_size': res['residual_compressed_size'],
        'bit_pct': res['model_bit_accuracy'],
        'psnr_db': res['psnr_db'],
        'train_time_s': res['train_time_s'],
        'predict_time_s': res['predict_time_s'],
        'total_time_s': dt,
        'sha256_match': meta['exact_match'],
    }


def benchmark_v4(img, epochs=1500, bits=8, prune=0.0):
    from siren_v4_bitperfect import ImageINRV4BitPerfect
    comp = ImageINRV4BitPerfect(hidden_dim=32, num_layers=2, omega_0=30.0)
    t0 = time.time()
    res = comp.compress(img, epochs=epochs, lr=1e-3,
                        bits=bits, prune_threshold=prune,
                        zlib_level=9, verbose=False)
    dt = time.time() - t0
    recon, meta = ImageINRV4BitPerfect.decompress(res['recipe_bytes'])
    return {
        'recipe_size': res['recipe_size'],
        'weights_size': res['siren_recipe_size'],
        'residual_size': res['residual_compressed_size'],
        'bit_pct': res['model_bit_accuracy'],
        'psnr_db': res['psnr_db'],
        'train_time_s': res['train_time_s'],
        'predict_time_s': res['predict_time_s'],
        'total_time_s': dt,
        'sha256_match': meta['exact_match'],
    }


def main():
    print("=" * 95)
    print("  v5 (PyTorch) vs v4 (numpy) vs ZIP — bit-perfect benchmark")
    print(f"  PyTorch {torch.__version__} | device: {('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 95)

    tests = [
        ('gradient_64',     make_pure_gradient(64)),
        ('gradient_128',    make_pure_gradient(128)),
        ('smooth_blobs_64', make_smooth_image(64, seed=42)),
        ('smooth_blobs_128', make_smooth_image(128, seed=42)),
    ]

    results = []
    for name, img in tests:
        print(f"\n--- {name} ({img.nbytes:,}B) ---")
        zip_size = len(zlib.compress(img.tobytes(), 9))
        print(f"  ZIP: {zip_size:,}B  (ratio {img.nbytes/zip_size:.2f}x)")

        # v4 (numpy)
        try:
            t0 = time.time()
            r4 = benchmark_v4(img, epochs=1500, bits=8, prune=0.0)
            v4_total = time.time() - t0
            print(f"  v4:  {r4['recipe_size']:>6,}B  bit%={r4['bit_pct']:.1f}  "
                  f"PSNR={r4['psnr_db']:.1f}  ok={r4['sha256_match']}  "
                  f"time={r4['total_time_s']:.1f}s")
        except Exception as e:
            print(f"  v4 FAILED: {e}")
            r4 = None

        # v5 (PyTorch)
        try:
            r5 = benchmark_v5(img, epochs=1500, bits=8, batch_size=2048)
            print(f"  v5:  {r5['recipe_size']:>6,}B  bit%={r5['bit_pct']:.1f}  "
                  f"PSNR={r5['psnr_db']:.1f}  ok={r5['sha256_match']}  "
                  f"time={r5['total_time_s']:.1f}s  (train {r5['train_time_s']:.1f}s)")
            if r4:
                speedup = r4['total_time_s'] / r5['total_time_s']
                print(f"  v5 speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"  v5 FAILED: {e}")
            import traceback; traceback.print_exc()
            r5 = None

        winner_v5 = "BLKH" if (r5 and r5['recipe_size'] < zip_size) else "ZIP"
        winner_v4 = "BLKH" if (r4 and r4['recipe_size'] < zip_size) else "ZIP"
        if r4 and r5:
            results.append({
                'name': name,
                'orig_size': img.nbytes,
                'zip_size': zip_size,
                'v4_recipe': r4['recipe_size'], 'v4_time_s': r4['total_time_s'],
                'v4_bit_pct': r4['bit_pct'], 'v4_ok': r4['sha256_match'],
                'v5_recipe': r5['recipe_size'], 'v5_time_s': r5['total_time_s'],
                'v5_bit_pct': r5['bit_pct'], 'v5_ok': r5['sha256_match'],
                'speedup': r4['total_time_s'] / r5['total_time_s'],
                'v5_winner': winner_v5,
            })

    print("\n" + "=" * 95)
    print("  SUMMARY")
    print("=" * 95)
    print(f"{'test':<22}{'orig':>8}{'zip':>8}{'v4':>8}{'v5':>8}"
          f"{'v4_t':>7}{'v5_t':>7}{'speed':>7}{'v5 win':>9}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<22}{r['orig_size']:>8,}{r['zip_size']:>8,}"
              f"{r['v4_recipe']:>8,}{r['v5_recipe']:>8,}"
              f"{r['v4_time_s']:>6.1f}s{r['v5_time_s']:>6.1f}s"
              f"{r['speedup']:>6.2f}x{r['v5_winner']:>9}")

    out = os.path.join(os.path.dirname(__file__), 'benchmark_v5_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out}")


if __name__ == '__main__':
    main()
