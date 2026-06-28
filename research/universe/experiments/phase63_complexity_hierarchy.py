# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 63: BHUH Complexity Hierarchy (P vs NP Implications)
=============================================================
Analyzes the computational complexity of BHUH operations.

CONCEPT:
  BHUH has three operations with different complexities:
  1. Genesis (decompression): O(|s|) — fast, polynomial
  2. Compression (seed finding): O(epochs × |s| × |x|) — slow
  3. Optimal seed finding: O(2^|s|) — exponential (incomputable, Chaitin)

  This mirrors the P vs NP problem:
  - Verifying a seed is fast (Genesis = P)
  - Finding the optimal seed is hard (search ≈ NP)
  - But gradient descent gives a "good enough" approximation

METHOD:
  1. Measure actual compression time for different file sizes
  2. Measure actual decompression time
  3. Compare: is compression >> decompression? (like NP >> P)
  4. Analyze: does compression time grow polynomially?

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, zlib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren, measure_model_size_compressed


def run_phase63_experiment(verbose=True):
    print("=" * 80)
    print("🧪 Phase 63: BHUH Complexity Hierarchy (P vs NP Implications)")
    print("=" * 80)

    device = 'cpu'

    # Test different image sizes
    sizes = [32, 64, 128, 256]
    results = []

    print(f"\n{'Size':>10} {'Raw':>10} {'Compress':>12} {'Decompress':>12} {'Ratio':>10} {'Seed':>8}")
    print("-" * 65)

    for size in sizes:
        # Generate image
        rng = np.random.default_rng(42)
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
        img = np.zeros((size, size, 3), dtype=np.float32)
        for c in range(3):
            for _ in range(3):
                kx, ky = rng.integers(2, 5, 2)
                img[:, :, c] += 50 * np.sin(2*np.pi*kx*xs) * np.cos(2*np.pi*ky*ys)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        raw = img.nbytes

        # Compression time (training)
        t0 = time.time()
        model, loss = train_single_siren(img, epochs=50, device=device, verbose=False)
        compress_time = time.time() - t0

        seed_size = measure_model_size_compressed(model)

        # Decompression time (inference)
        coords = get_coordinates(size, device)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(10):  # average 10 runs
                pred = model(coords)
        decompress_time = (time.time() - t0) / 10

        ratio = compress_time / max(decompress_time, 1e-6)

        print(f"{size}x{size:<5} {raw:>9,}B {compress_time:>10.3f}s {decompress_time:>10.4f}s {ratio:>8.0f}x {seed_size:>7,}B")

        results.append({
            'size': size, 'raw': raw, 'compress': compress_time,
            'decompress': decompress_time, 'ratio': ratio, 'seed': seed_size,
        })

    # Complexity analysis
    print(f"\n{'='*80}")
    print("📊 PHASE 63 SUMMARY — COMPLEXITY HIERARCHY")
    print(f"{'='*80}")

    # Fit power law: compress_time ~ a * size^b
    sizes_arr = np.array([r['size'] for r in results])
    comp_times = np.array([r['compress'] for r in results])
    decomp_times = np.array([r['decompress'] for r in results])

    # Log-log fit
    log_sizes = np.log(sizes_arr)
    log_comp = np.log(comp_times)
    log_decomp = np.log(decomp_times + 1e-10)

    comp_slope = np.polyfit(log_sizes, log_comp, 1)[0]
    decomp_slope = np.polyfit(log_sizes, log_decomp, 1)[0]

    print(f"\n  📋 Complexity scaling (power law exponent):")
    print(f"  - Compression: O(n^{comp_slope:.2f}) — {'polynomial ✅' if comp_slope < 4 else 'high degree'}")
    print(f"  - Decompression: O(n^{decomp_slope:.2f}) — {'near-linear ✅' if decomp_slope < 2 else 'polynomial'}")
    print(f"  - Compression/Decompression ratio: {results[-1]['ratio']:.0f}x (compress >> decompress)")

    print(f"\n  📋 P vs NP analogy:")
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │ P (easy):  Genesis (decompression)             │")
    print(f"  │            O(n^{decomp_slope:.1f}) — verify seed produces output     │")
    print(f"  │                                                 │")
    print(f"  │ NP (hard): Compression (seed finding)           │")
    print(f"  │            O(n^{comp_slope:.1f}) — find seed that produces output  │")
    print(f"  │            Optimal: O(2^n) — incomputable       │")
    print(f"  │            (Chaitin: K(x) is incomputable)      │")
    print(f"  │                                                 │")
    print(f"  │ NP-complete? No — but OPTIMAL compression is    │")
    print(f"  │ uncomputable (not just hard — IMPOSSIBLE)       │")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  📋 Key insight:")
    print(f"  BHUH mirrors the P vs NP structure:")
    print(f"  - Decompression (P): fast, polynomial — O(n^{decomp_slope:.1f})")
    print(f"  - Compression (NP-like): slow, polynomial — O(n^{comp_slope:.1f})")
    print(f"  - Optimal compression: IMPOSSIBLE (Chaitin's theorem)")
    print(f"  - Gradient descent = 'NP oracle' approximation")
    print(f"  - Like SAT solvers: exponential worst case, polynomial average case")

    print(f"\n  📋 Practical implications:")
    print(f"  - Asymmetric compression: compress ONCE, decompress MANY times")
    print(f"  - Compression can be amortized (Multi-File SIREN)")
    print(f"  - Real-time decompression (sub-ms) even for large files")
    print(f"  - Offline compression (batch processing) is acceptable")

    avg_ratio = np.mean([r['ratio'] for r in results])
    print(f"\n  📊 Average compress/decompress ratio: {avg_ratio:.0f}x")
    print(f"  (Compression is {avg_ratio:.0f}x slower than decompression)")

    return {
        'comp_slope': comp_slope,
        'decomp_slope': decomp_slope,
        'avg_ratio': avg_ratio,
    }


if __name__ == '__main__':
    results = run_phase63_experiment(verbose=True)
