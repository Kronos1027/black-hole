#!/usr/bin/env python3
"""
benchmark_bitperfect.py — Benchmark final do modo bit-perfect BLKH vs ZIP
=========================================================================
Testa o caso de uso real onde BLKH+Residual deve vencer ZIP:
imagens 2D suaves em diferentes resoluções.

Mostra que com INT8 (sem pruning agressivo) o BLKH+Residual:
- Mantém 100% bit accuracy (SHA-256 verificado)
- Vence ZIP em sinais suaves
- Perde para ZIP em sinais caóticos (esperado, Shannon)
"""
import sys
import os
import time
import zlib
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
from siren_v4_bitperfect import ImageINRV4BitPerfect
import numpy as np


def make_pure_gradient(size):
    """Gradiente puro — caso onde SIREN tem residual ≈ 0."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            img[i, j] = [int(i * 255 / size), int(j * 255 / size), int((i + j) * 255 / (2 * size))]
    return img


def make_gaussian_blobs(size, seed=42):
    """Blobs gaussianos — smooth 2D, caso favorável."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        cy, cx = rng.uniform(size * 0.2, size * 0.8, 2)
        sigma = rng.uniform(size * 0.1, size * 0.25)
        amp = rng.uniform(80, 200)
        img[:, :, c] = amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_random_image(size, seed=42):
    """Imagem aleatória — caso onde ZIP deve vencer (Shannon)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def benchmark_one(name, img, epochs=1500, bits=8, prune=0.0):
    zip_size = len(zlib.compress(img.tobytes(), 9))
    orig_size = img.nbytes

    comp = ImageINRV4BitPerfect(hidden_dim=32, num_layers=2, omega_0=30.0)
    t0 = time.time()
    try:
        res = comp.compress(img, epochs=epochs, lr=1e-3, bits=bits, prune_threshold=prune)
    except Exception as e:
        return {'name': name, 'error': str(e)}
    dt = time.time() - t0

    recon, meta = ImageINRV4BitPerfect.decompress(res['recipe_bytes'])

    winner = 'BLKH' if res['recipe_size'] < zip_size else 'ZIP'
    return {
        'name': name,
        'orig_size': orig_size,
        'zip_size': zip_size,
        'siren_recipe': res['siren_recipe_size'],
        'residual_compressed': res['residual_compressed_size'],
        'total_recipe': res['recipe_size'],
        'model_bit_pct': round(res['model_bit_accuracy'], 2),
        'psnr_db': round(res['psnr_db'], 2),
        'sha256_match': meta['exact_match'],
        'compress_time_s': round(dt, 2),
        'winner': winner,
        'blkh_vs_zip': round(res['recipe_size'] / zip_size, 3),
    }


def main():
    print("=" * 80)
    print("  BLACK HOLE — BIT-PERFECT BENCHMARK (v4 + Residual XOR)")
    print("  100% SHA-256 verified roundtrip")
    print("=" * 80)

    tests = [
        ('gradient_64',    make_pure_gradient(64)),
        ('gradient_128',   make_pure_gradient(128)),
        ('blobs_64',       make_gaussian_blobs(64, seed=42)),
        ('blobs_128',      make_gaussian_blobs(128, seed=42)),
        ('random_64',      make_random_image(64, seed=42)),
    ]

    results = []
    for name, img in tests:
        print(f"\n--- {name} ({img.nbytes:,}B) ---")
        r = benchmark_one(name, img, epochs=1500, bits=8, prune=0.0)
        if 'error' in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  ZIP:        {r['zip_size']:>7,} B  (ratio {r['orig_size']/r['zip_size']:.2f}x)")
        print(f"  BLKH+Res:   {r['total_recipe']:>7,} B  (ratio {r['orig_size']/r['total_recipe']:.2f}x)")
        print(f"    SIREN:    {r['siren_recipe']:>7,} B")
        print(f"    Residual: {r['residual_compressed']:>7,} B")
        print(f"  Bit acc:    {r['model_bit_pct']:.1f}%   PSNR: {r['psnr_db']:.1f} dB")
        print(f"  SHA-256:    {'OK' if r['sha256_match'] else 'FAIL'}")
        print(f"  Time:       {r['compress_time_s']:.1f}s")
        print(f"  Winner:     {r['winner']}  (BLKH/ZIP = {r['blkh_vs_zip']:.3f})")
        results.append(r)

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"{'test':<18}{'orig':>9}{'zip':>9}{'blkh':>9}{'bit%':>8}{'ok':>5}{'winner':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<18}{r['orig_size']:>9,}{r['zip_size']:>9,}"
              f"{r['total_recipe']:>9,}{r['model_bit_pct']:>7.1f}%"
              f"{'OK' if r['sha256_match'] else 'FAIL':>5}{r['winner']:>10}")

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), 'benchmark_bitperfect_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
