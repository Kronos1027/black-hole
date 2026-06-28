#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Phase 1 Scaling Benchmark — Tests how shared roots benefit scales with N."""
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import run_experiment


def main():
    print("=" * 80)
    print("📊 Phase 1 Scaling Benchmark — Shared Roots vs N")
    print("=" * 80)
    print()

    configs = [
        (5, 128, 40, 80),
        (10, 128, 40, 80),
        (20, 128, 40, 80),
        (50, 128, 40, 80),
    ]

    all_results = []

    for n_images, size, ep_single, ep_multi in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {n_images} images @ {size}x{size}")
        print(f"{'='*60}")

        result = run_experiment(
            n_images=n_images,
            size=size,
            epochs_single=ep_single,
            epochs_multi=ep_multi,
            verbose=False
        )
        all_results.append(result)

        print(f"\n  Separate SIRENs: {result['baseline_size']:,}B")
        print(f"  BHUH shared:     {result['bhuh_size']:,}B")
        print(f"  Improvement:     {result['improvement']:.2f}x")
        print(f"  ZIP:             {result['zip_size']:,}B")
        print(f"  BHUH vs ZIP:     {result['zip_size']/result['bhuh_size']:.2f}x")

    # Summary
    print("\n" + "=" * 80)
    print("📈 SCALING LAW SUMMARY")
    print("=" * 80)
    print(f"\n{'N':>5} {'Size':>10} {'Separate':>12} {'BHUH':>10} {'Improv':>10} {'ZIP':>12} {'vs ZIP':>10}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['n_images']:>5} {'128x128':>10} {r['baseline_size']:>11,}B {r['bhuh_size']:>9,}B "
              f"{r['improvement']:>9.2f}x {r['zip_size']:>11,}B "
              f"{r['zip_size']/r['bhuh_size']:>9.2f}x")

    # Save results
    output = {
        'experiment': 'phase1_scaling',
        'date': '2026-06-25',
        'results': all_results,
    }
    output_path = os.path.join(os.path.dirname(__file__), 'phase1_scaling_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == '__main__':
    main()
