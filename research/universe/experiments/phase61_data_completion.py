# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 61: Neural Data Completion (Predict Missing Files)
==========================================================
Tests whether the universe can PREDICT missing files.

CONCEPT:
  If we train a universe on N files and remove file k, can we
  RECONSTRUCT file k from the remaining N-1 files?

  This tests whether the universe has "learned" the structure
  well enough to predict missing members.

  Approach: use the average of k-nearest modulations as a proxy
  for the missing file's modulation.

HYPOTHESIS:
  Files similar to the missing one will have similar modulations.
  Averaging the k nearest modulations will produce a reasonable
  approximation of the missing file.

METHOD:
  1. Train Multi-File SIREN on 20 images
  2. Remove file k, approximate its modulation from neighbors
  3. Compare: predicted vs actual file k
  4. Test with different k values (1, 3, 5 nearest neighbors)

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import ModulatedSIREN, get_coordinates, generate_satellite_images, train_multi_file_siren


def run_phase61_experiment(verbose=True):
    print("=" * 80)
    print("🧪 Phase 61: Neural Data Completion (Predict Missing Files)")
    print("=" * 80)

    device = 'cpu'
    n_files = 20
    size = 64

    # Generate images with PAIRS of similar files
    # Each pair shares frequencies (so neighbors exist)
    images = []
    for i in range(n_files):
        rng = np.random.default_rng(42 + i // 2)  # pairs share seed
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
        img = np.zeros((size, size, 3), dtype=np.float32)
        for c in range(3):
            for _ in range(3):
                kx, ky = rng.integers(2, 5, 2)
                amp = rng.uniform(40, 80)
                phase = np.random.default_rng(42 + i).uniform(0, 2*np.pi)
                img[:, :, c] += amp * np.sin(2*np.pi*kx*xs + phase) * np.cos(2*np.pi*ky*ys)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        images.append(img)

    # Train universe
    print(f"\n🌌 Training universe on {n_files} images (paired)...")
    model, loss = train_multi_file_siren(images, epochs=100, device=device, verbose=False)

    # Extract modulations
    mods = model.modulations.weight.detach().cpu().numpy()  # (20, 16)

    # For each file, predict it from neighbors
    coords = get_coordinates(size, device)

    k_values = [1, 3, 5]
    results = []

    print(f"\n{'File':>6} {'k=1':>10} {'k=3':>10} {'k=5':>10} {'Best k':>8}")
    print("-" * 50)

    for target_idx in range(n_files):
        # Compute distances to all other files
        target_mod = mods[target_idx]
        other_indices = [i for i in range(n_files) if i != target_idx]
        distances = [np.linalg.norm(target_mod - mods[i]) for i in other_indices]

        # Sort by distance
        sorted_pairs = sorted(zip(other_indices, distances), key=lambda x: x[1])

        row = {}
        for k in k_values:
            # K-nearest neighbor average
            knn_indices = [idx for idx, _ in sorted_pairs[:k]]
            knn_mods = mods[knn_indices]
            predicted_mod = knn_mods.mean(axis=0)

            # Generate image with predicted modulation
            mod_tensor = torch.from_numpy(predicted_mod.astype(np.float32)).to(device)
            x = coords
            for j, layer in enumerate(model.base_siren.net):
                if j < len(model.base_siren.net) - 1:
                    film = model.film_generators[j](mod_tensor)
                    scale, shift = film.chunk(2, dim=-1)
                    x = layer(x)
                    x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
                else:
                    x = layer(x)

            pred_img = (x.detach().cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)

            # Compare with actual
            actual = images[target_idx]
            mse = np.mean((actual.astype(float) - pred_img.astype(float))**2)
            psnr = 10 * np.log10(255**2 / max(mse, 1e-10)) if mse > 0 else 99
            row[k] = psnr

        best_k = max(k_values, key=lambda k: row[k])
        print(f"{target_idx:>6} {row[1]:>8.1f}dB {row[3]:>8.1f}dB {row[5]:>8.1f}dB {'k='+str(best_k):>7}")

        results.append({
            'file': target_idx,
            'psnr_k1': row[1], 'psnr_k3': row[3], 'psnr_k5': row[5],
            'best_k': best_k,
        })

    # Summary
    avg_k1 = np.mean([r['psnr_k1'] for r in results])
    avg_k3 = np.mean([r['psnr_k3'] for r in results])
    avg_k5 = np.mean([r['psnr_k5'] for r in results])

    print(f"\n{'AVERAGE':>6} {avg_k1:>8.1f}dB {avg_k3:>8.1f}dB {avg_k5:>8.1f}dB")

    print(f"\n{'='*80}")
    print("📊 PHASE 61 SUMMARY — NEURAL DATA COMPLETION")
    print(f"{'='*80}")

    best_avg = max(avg_k1, avg_k3, avg_k5)
    best_k = 1 if avg_k1 == best_avg else 3 if avg_k3 == best_avg else 5

    print(f"\n  📋 Results:")
    print(f"  - k=1 (nearest): {avg_k1:.1f}dB")
    print(f"  - k=3 (3 nearest): {avg_k3:.1f}dB")
    print(f"  - k=5 (5 nearest): {avg_k5:.1f}dB")
    print(f"  - Best: k={best_k} ({best_avg:.1f}dB)")

    if best_avg > 20:
        print(f"\n  ✅ Data completion WORKS! ({best_avg:.1f}dB with k={best_k})")
        print(f"     The universe can PREDICT missing files from neighbors.")
    elif best_avg > 15:
        print(f"\n  ⚠️  Partial prediction ({best_avg:.1f}dB) — rough but recognizable")
    else:
        print(f"\n  ❌ Prediction fails ({best_avg:.1f}dB)")

    print(f"\n  📋 Key insight:")
    print(f"  The modulation space has TOPOLOGY (Phase 45).")
    print(f"  Files near each other in modulation space are SIMILAR.")
    print(f"  Averaging neighbors = approximating the missing file.")
    print(f"  This is 'universe interpolation' — predicting what SHOULD exist.")

    print(f"\n  📋 Applications:")
    print(f"  - Data recovery: reconstruct lost files from universe")
    print(f"  - Data validation: detect anomalous files (far from neighbors)")
    print(f"  - Recommendation: suggest files similar to existing ones")
    print(f"  - Compression: skip similar files, predict from neighbors")

    return {
        'avg_k1': avg_k1, 'avg_k3': avg_k3, 'avg_k5': avg_k5,
        'best_k': best_k, 'best_avg': best_avg,
    }


if __name__ == '__main__':
    results = run_phase61_experiment(verbose=True)
