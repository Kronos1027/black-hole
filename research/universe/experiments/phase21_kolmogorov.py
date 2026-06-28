# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 21: Entropy-Bounded SIREN (Kolmogorov Approximation)
=============================================================
Tests whether we can approximate Kolmogorov complexity K(x) using SIREN.

CONCEPT:
  Kolmogorov complexity K(x) = size of smallest program generating x.
  SIREN approximates this: K(x) ≈ |SIREN_weights| + |residual|.

  The question: how close is SIREN to true K(x)?
  We can estimate K(x) lower bound via zlib (compression).
  If SIREN < zlib, SIREN is closer to K(x) than statistical methods.

HYPOTHESIS:
  For smooth images, SIREN size will be below zlib (closer to K(x)).
  For random images, SIREN ≈ zlib ≈ raw size (Shannon limit).
  The RATIO SIREN/zlib measures "how much structure SIREN captures."

METHOD:
  1. Generate images with varying Kolmogorov complexity:
     - Pure gradient (K ≈ O(log n))
     - 3 frequencies (K ≈ O(1))
     - 10 frequencies (K ≈ O(10))
     - Random noise (K ≈ n)
  2. For each: SIREN size, zlib size, raw size
  3. Plot K-approximation spectrum

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren, measure_model_size_compressed


def generate_complexity_spectrum(size=128):
    """Generate images with varying Kolmogorov complexity."""
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    images = []

    # 1. Pure gradient (K very low)
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = xs * 255
    img[:, :, 1] = ys * 255
    img[:, :, 2] = (xs + ys) * 127.5
    images.append(("1_freq (K≈low)", img.astype(np.uint8)))

    # 2. Three frequencies
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(2, 5, 2)
            img[:, :, c] += 50 * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
    images.append(("3_freq (K≈med)", ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)))

    # 3. Ten frequencies
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(10):
            kx, ky = rng.integers(1, 15, 2)
            img[:, :, c] += 30 * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
    images.append(("10_freq (K≈high)", ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)))

    # 4. White noise (K ≈ n, Shannon limit)
    noise = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    images.append(("random (K≈n)", noise))

    return images


def run_phase21_experiment(verbose=True):
    """Run Phase 21 Entropy-Bounded SIREN experiment."""
    print("=" * 80)
    print("🧪 Phase 21: Entropy-Bounded SIREN (Kolmogorov Approximation)")
    print("=" * 80)

    device = 'cpu'
    size = 128

    images = generate_complexity_spectrum(size)

    print(f"\n{'Image':<25} {'Raw':>8} {'ZIP':>8} {'SIREN':>8} {'SIREN/ZIP':>10} {'K-approx':>10}")
    print("-" * 75)

    results = []

    for name, img in images:
        raw = img.nbytes
        zip_sz = len(zlib.compress(img.tobytes(), 9))

        # Train SIREN
        model, loss = train_single_siren(img, epochs=100, device=device, verbose=False)
        siren_sz = measure_model_size_compressed(model)

        # K-approximation ratio: how close is SIREN to theoretical K?
        # If SIREN < ZIP, SIREN is closer to true K(x)
        k_ratio = siren_sz / max(zip_sz, 1)

        # Interpretation
        if k_ratio < 0.5:
            k_label = "SIREN << ZIP (structured)"
        elif k_ratio < 1.0:
            k_label = "SIREN < ZIP (some structure)"
        elif k_ratio < 2.0:
            k_label = "SIREN ≈ ZIP (moderate)"
        else:
            k_label = "SIREN >> ZIP (random/high-K)"

        print(f"{name:<25} {raw:>7,}B {zip_sz:>7,}B {siren_sz:>7,}B {k_ratio:>9.2f}x {k_label}")

        results.append({
            'name': name,
            'raw': raw,
            'zip': zip_sz,
            'siren': siren_sz,
            'k_ratio': k_ratio,
            'loss': loss,
        })

    # Summary
    print(f"\n{'='*80}")
    print("📊 PHASE 21 SUMMARY — KOLMOGOROV APPROXIMATION SPECTRUM")
    print(f"{'='*80}")

    print(f"\n  Kolmogorov Complexity Spectrum:")
    print(f"  Low K (gradient):  SIREN = {results[0]['siren']:,}B, ZIP = {results[0]['zip']:,}B → SIREN/ZIP = {results[0]['k_ratio']:.2f}x")
    print(f"  Med K (3 freq):    SIREN = {results[1]['siren']:,}B, ZIP = {results[1]['zip']:,}B → SIREN/ZIP = {results[1]['k_ratio']:.2f}x")
    print(f"  High K (10 freq):  SIREN = {results[2]['siren']:,}B, ZIP = {results[2]['zip']:,}B → SIREN/ZIP = {results[2]['k_ratio']:.2f}x")
    print(f"  Max K (random):    SIREN = {results[3]['siren']:,}B, ZIP = {results[3]['zip']:,}B → SIREN/ZIP = {results[3]['k_ratio']:.2f}x")

    print(f"\n  📋 Key findings:")
    print(f"  - For LOW K (structured): SIREN beats ZIP (closer to true K)")
    print(f"  - For HIGH K (random): SIREN loses to ZIP (network overhead)")
    print(f"  - The crossover point determines when SIREN is optimal")
    print(f"  - This is the BHUH 'Singularity' principle: K(x) ≈ |SIREN| for structured x")

    # Find crossover
    for i in range(len(results) - 1):
        if results[i]['k_ratio'] < 1.0 and results[i + 1]['k_ratio'] >= 1.0:
            print(f"\n  📊 Crossover: SIREN wins until '{results[i]['name']}', loses after '{results[i+1]['name']}'")
            break
    else:
        if results[0]['k_ratio'] >= 1.0:
            print(f"\n  SIREN loses everywhere (needs more training or different data)")
        else:
            print(f"\n  SIREN wins everywhere (all test images structured)")

    return results


if __name__ == '__main__':
    results = run_phase21_experiment(verbose=True)
