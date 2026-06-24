#!/usr/bin/env python3
"""
benchmark_realistic.py — BLKH v5 on realistic data types
=========================================================
Tests on data that simulates real-world use cases:
  - Medical imaging (MRI-like smooth grayscale volumes, 3 channels)
  - Satellite tiles (smooth multispectral gradients)
  - Scientific grids (PDE solution fields)
  - Game textures (procedural smooth surfaces)

Each is 128x128 or larger — the sweet spot for INRs.
"""
import sys
import os
import time
import zlib
import json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
torch.set_num_threads(4)

from siren_v5_torch import ImageINRv5


# ============================================================
#  Realistic data generators
# ============================================================
def make_mri_like(size=128, seed=42):
    """Simulate an MRI slice — smooth tissue boundaries with gaussian blobs
    representing different tissue types. RGB channels simulate T1/T2/PD."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Background tissue
    img[:, :, 0] = 60 + 20 * np.sin(xs * 0.05) * np.cos(ys * 0.05)
    img[:, :, 1] = 80 + 25 * np.sin(ys * 0.04)
    img[:, :, 2] = 50 + 15 * np.cos(xs * 0.06 + ys * 0.03)
    # Add 4-6 "tissue regions" as gaussian blobs
    for _ in range(rng.integers(4, 7)):
        cy, cx = rng.uniform(15, size - 15, 2)
        sigma = rng.uniform(8, 20)
        amp = rng.uniform(50, 150)
        for c in range(3):
            img[:, :, c] += amp * np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma**2)) * (c + 1) / 3
    return np.clip(img, 0, 255).astype(np.uint8)


def make_satellite_like(size=128, seed=42):
    """Simulate a satellite tile — smooth multispectral gradients (vegetation
    indices, water, urban) with low-frequency variation."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)
    # 3-4 large-scale features
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(1, 4, 2)
            amp = rng.uniform(30, 80)
            phase = rng.uniform(0, 2 * np.pi)
            img[:, :, c] += amp * np.sin(2 * np.pi * kx * xs / size + phase) * \
                            np.cos(2 * np.pi * ky * ys / size)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9) * 255
    return img.astype(np.uint8)


def make_pde_field(size=128, seed=42):
    """Simulate a PDE solution field (e.g. heat equation, fluid vorticity).
    Smooth low-frequency content, 3 channels representing different fields."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(4):
            kx, ky = rng.integers(1, 5, 2)
            amp = rng.uniform(-1, 1)
            phase = rng.uniform(0, 2 * np.pi)
            img[:, :, c] += amp * np.sin(2 * np.pi * kx * xs + phase) * \
                            np.cos(2 * np.pi * ky * ys)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9) * 255
    return img.astype(np.uint8)


def make_game_texture(size=128, seed=42):
    """Simulate a procedural game texture — large smooth regions with one
    high-frequency detail layer (low amplitude)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Base gradient
    for c in range(3):
        img[:, :, c] = 80 + 100 * np.sin((xs + ys) * np.pi * (c + 1))
    # 3 gaussian "highlights"
    for _ in range(3):
        cy, cx = rng.uniform(0.2, 0.8, 2)
        sigma = rng.uniform(0.1, 0.25)
        amp = rng.uniform(40, 80)
        img += amp * np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma**2))[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def make_photo_like(size=128, seed=42):
    """Simulate a photo — mix of smooth gradients + some noise (hardest case)."""
    rng = np.random.default_rng(seed)
    base = make_pde_field(size, seed)
    noise = rng.normal(0, 5, base.shape)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ============================================================
#  Benchmark
# ============================================================
def benchmark(name, img, epochs=1500):
    zip_size = len(zlib.compress(img.tobytes(), 9))
    print(f"\n--- {name} ({img.shape}, {img.nbytes:,}B) ---")
    print(f"  ZIP: {zip_size:,}B  (ratio {img.nbytes/zip_size:.2f}x)")

    comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
    t0 = time.time()
    res = comp.compress_bitperfect(img, epochs=epochs, lr=1e-3,
                                     bits=8, batch_size=2048, verbose=False)
    dt = time.time() - t0
    recon, meta = ImageINRv5.decompress(res['recipe_bytes'])
    winner = "BLKH" if res['recipe_size'] < zip_size else "ZIP"
    print(f"  BLKH: {res['recipe_size']:,}B  (ratio {img.nbytes/res['recipe_size']:.2f}x)")
    print(f"    weights: {res['weights_packed_size']:,}B   residual: {res['residual_compressed_size']:,}B")
    print(f"    bit acc: {res['model_bit_accuracy']:.1f}%   PSNR: {res['psnr_db']:.1f}dB")
    print(f"    SHA-256: {'OK' if meta['exact_match'] else 'FAIL'}")
    print(f"    time: {dt:.1f}s  vs ZIP: {zip_size/res['recipe_size']:.3f}x  -> {winner}")
    return {
        'name': name,
        'orig_size': img.nbytes,
        'zip_size': zip_size,
        'blkh_size': res['recipe_size'],
        'weights_size': res['weights_packed_size'],
        'residual_size': res['residual_compressed_size'],
        'bit_pct': round(res['model_bit_accuracy'], 2),
        'psnr_db': round(res['psnr_db'], 2),
        'sha256_ok': meta['exact_match'],
        'time_s': round(dt, 2),
        'winner': winner,
        'atlas_vs_zip': round(zip_size / res['recipe_size'], 3),
    }


def main():
    print("=" * 95)
    print("  BLKH v5 — Realistic Data Benchmark")
    print("=" * 95)
    results = []

    for name, fn in [
        ('mri_like_128',         lambda: make_mri_like(128, seed=42)),
        ('satellite_like_128',   lambda: make_satellite_like(128, seed=42)),
        ('pde_field_128',        lambda: make_pde_field(128, seed=42)),
        ('game_texture_128',     lambda: make_game_texture(128, seed=42)),
        ('photo_with_noise_128', lambda: make_photo_like(128, seed=42)),
    ]:
        r = benchmark(name, fn())
        results.append(r)

    print("\n" + "=" * 95)
    print("  REALISTIC DATA SUMMARY")
    print("=" * 95)
    print(f"{'data':<25}{'orig':>10}{'zip':>9}{'blkh':>9}{'bit%':>8}{'psnr':>8}{'win':>8}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<25}{r['orig_size']:>10,}{r['zip_size']:>9,}"
              f"{r['blkh_size']:>9,}{r['bit_pct']:>7.1f}%{r['psnr_db']:>7.1f}dB"
              f"{r['winner']:>8}")

    out = Path(__file__).parent / 'benchmark_realistic_results.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")


if __name__ == '__main__':
    main()
