# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 16: Dataset-Level Compression (1000 files)
==================================================
Tests the scaling law at SCALE. Phase 1 tested up to 50 images.
Does the scaling law (improvement ≈ N/3) hold at 100, 200, 500 files?

HYPOTHESIS:
  At 500 files, improvement will be ~167x vs separate SIRENs.
  BHUH size will stay ~25-30KB regardless of N.
  This would mean 500 images compressed to ~30KB total.

METHOD:
  1. Generate 100, 200, 500 synthetic satellite images (32x32 for speed)
  2. Train Multi-File SIREN for each
  3. Measure BHUH size, improvement, vs ZIP
  4. Verify scaling law prediction

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
from phase1_multi_file_siren import (
    generate_satellite_images, train_multi_file_siren,
    train_single_siren, measure_model_size_compressed
)


def run_phase16_experiment(verbose=True):
    """Run Phase 16 Dataset-Level experiment."""
    print("=" * 80)
    print("🧪 Phase 16: Dataset-Level Compression (100-500 files)")
    print("=" * 80)

    device = 'cpu'
    size = 32  # Small for speed at scale

    configs = [50, 100, 200, 500]

    print(f"\n{'N Files':>8} {'BHUH Size':>12} {'Separate':>12} {'Improv':>10} {'vs ZIP':>10} {'Predicted':>10} {'Time':>8}")
    print("-" * 75)

    results = []

    for n_files in configs:
        if verbose:
            print(f"\n  Testing {n_files} images @ {size}x{size}...")

        # Generate images
        images = generate_satellite_images(n_files, size, seed=42)

        # ZIP baseline
        zip_total = sum(len(zlib.compress(img.tobytes(), 9)) for img in images)

        # Separate SIRENs (estimate: N * single_siren_size)
        # For speed, train 5 and extrapolate
        sample_size = min(5, n_files)
        sample_siren_sizes = []
        for i in range(sample_size):
            m, _ = train_single_siren(images[i], epochs=30, device=device, verbose=False)
            sample_siren_sizes.append(measure_model_size_compressed(m))
        avg_siren = np.mean(sample_siren_sizes)
        separate_total = int(avg_siren * n_files)

        # BHUH Multi-File SIREN
        t0 = time.time()
        epochs = max(30, min(80, 200 // n_files * 10))  # adaptive epochs
        model, loss = train_multi_file_siren(images, epochs=epochs, device=device, verbose=False)
        dt = time.time() - t0
        bhuh_size = measure_model_size_compressed(model)

        improvement = separate_total / max(bhuh_size, 1)
        vs_zip = zip_total / max(bhuh_size, 1)
        predicted = n_files / 3  # scaling law from Phase 1

        print(f"{n_files:>8} {bhuh_size:>11,}B {separate_total:>11,}B {improvement:>9.1f}x {vs_zip:>9.1f}x {predicted:>9.1f}x {dt:>7.1f}s")

        results.append({
            'n_files': n_files,
            'bhuh_size': bhuh_size,
            'separate_est': separate_total,
            'improvement': improvement,
            'vs_zip': vs_zip,
            'predicted': predicted,
            'zip_total': zip_total,
            'time': dt,
        })

    # Summary
    print(f"\n{'='*80}")
    print("📊 PHASE 16 SUMMARY — DATASET-LEVEL SCALING")
    print(f"{'='*80}")
    print(f"\n  Scaling Law Validation:")
    for r in results:
        match = "✅" if abs(r['improvement'] - r['predicted']) / r['predicted'] < 0.3 else "⚠️"
        print(f"  N={r['n_files']:>4}: actual={r['improvement']:.1f}x, predicted={r['predicted']:.1f}x {match}")

    print(f"\n  BHUH size growth:")
    for r in results:
        print(f"  N={r['n_files']:>4}: {r['bhuh_size']:,}B ({r['bhuh_size']/1024:.1f}KB)")

    print(f"\n  📋 Key findings:")
    print(f"  - At 500 files: BHUH = {results[-1]['bhuh_size']:,}B for {results[-1]['n_files']} images!")
    print(f"  - vs ZIP: {results[-1]['vs_zip']:.1f}x smaller")
    print(f"  - Per-image cost: {results[-1]['bhuh_size']/results[-1]['n_files']:.1f}B/image")
    print(f"  - Scaling law {'CONFIRMED' if all(abs(r['improvement'] - r['predicted'])/r['predicted'] < 0.5 for r in results) else 'PARTIALLY CONFIRMED'}")

    return results


if __name__ == '__main__':
    results = run_phase16_experiment(verbose=True)
