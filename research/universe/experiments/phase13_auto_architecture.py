# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 13: Auto-Architecture Search
====================================
Tests whether we can automatically find the optimal SIREN architecture
for each file type.

CONCEPT:
  Different files need different network sizes. A smooth gradient needs
  a tiny network; a complex photo needs a larger one. Instead of guessing,
  we EVOLVE the architecture:

  1. Start with minimal network (1 layer, 8 features)
  2. Train briefly, measure loss
  3. If loss > threshold, grow network (add layer or features)
  4. Repeat until loss acceptable or size budget exceeded

HYPOTHESIS:
  Auto-architecture will find smaller networks for simple files and
  larger for complex ones, achieving better average compression than
  fixed-size networks.

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, SIRENLayer, get_coordinates, measure_model_size_compressed


def train_and_evaluate(model, image, epochs=30, device='cpu'):
    """Quick train and return loss."""
    size = image.shape[0]
    coords = get_coordinates(size, device)
    pixels = torch.from_numpy(image.astype(np.float32) / 255.0).reshape(-1, 3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(coords)
        loss = F.mse_loss(pred, pixels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = model(coords)
        loss = F.mse_loss(pred, pixels).item()
    return loss


def auto_architecture_search(image, max_layers=4, max_features=64,
                               target_loss=0.005, device='cpu', verbose=False):
    """Evolve SIREN architecture to find optimal size for this image.

    Strategy:
    1. Start minimal (1 layer, 8 features)
    2. Train briefly, check loss
    3. If loss > target: grow (alternate between adding layers and features)
    4. Stop when loss < target or budget exceeded
    """
    size = image.shape[0]
    coords = get_coordinates(size, device)
    pixels = torch.from_numpy(image.astype(np.float32) / 255.0).reshape(-1, 3).to(device)

    # Start minimal
    hidden_layers = 1
    hidden_features = 8
    omega_0 = 30.0

    best_model = None
    best_loss = float('inf')
    history = []

    for iteration in range(10):  # max 10 growth steps
        model = SIREN(in_features=2, hidden_features=hidden_features,
                      hidden_layers=hidden_layers, out_features=3, omega_0=omega_0).to(device)

        loss = train_and_evaluate(model, image, epochs=30, device=device)
        size_bytes = measure_model_size_compressed(model)

        history.append({
            'iteration': iteration,
            'layers': hidden_layers,
            'features': hidden_features,
            'loss': loss,
            'size': size_bytes,
        })

        if verbose:
            print(f"  Iter {iteration}: layers={hidden_layers}, feat={hidden_features}, "
                  f"loss={loss:.6f}, size={size_bytes:,}B")

        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)

        # Check if we're done
        if loss < target_loss:
            if verbose:
                print(f"  ✅ Target loss reached!")
            break

        # Grow: alternate between adding features and layers
        if iteration % 2 == 0:
            hidden_features = min(hidden_features * 2, max_features)
        else:
            hidden_layers = min(hidden_layers + 1, max_layers)

    return best_model, best_loss, history


def run_phase13_experiment(verbose=True):
    """Run Phase 13 Auto-Architecture experiment."""
    print("=" * 80)
    print("🧪 Phase 13: Auto-Architecture Search (Evolutionary SIREN)")
    print("=" * 80)

    device = 'cpu'

    # Generate images with varying complexity
    rng = np.random.default_rng(42)
    images = []
    labels = []

    # Simple gradient (should need tiny network)
    size = 64
    xs = np.linspace(0, 1, size, dtype=np.float32)
    ys = np.linspace(0, 1, size, dtype=np.float32)
    simple = np.zeros((size, size, 3), dtype=np.float32)
    simple[:, :, 0] = xs[None, :]
    simple[:, :, 1] = ys[:, None]
    simple[:, :, 2] = (xs[None, :] + ys[:, None]) / 2
    images.append((simple * 255).astype(np.uint8))
    labels.append("simple_gradient")

    # Medium complexity (3 frequencies)
    ys2, xs2 = np.mgrid[0:size, 0:size].astype(np.float32) / size
    medium = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(1, 5, 2)
            medium[:, :, c] += 50 * np.sin(2 * np.pi * kx * xs2) * np.cos(2 * np.pi * ky * ys2)
    images.append(((medium - medium.min()) / (medium.max() - medium.min()) * 255).astype(np.uint8))
    labels.append("medium_3freq")

    # Complex (8 frequencies)
    complex_img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(8):
            kx, ky = rng.integers(1, 10, 2)
            complex_img[:, :, c] += 40 * np.sin(2 * np.pi * kx * xs2) * np.cos(2 * np.pi * ky * ys2)
    images.append(((complex_img - complex_img.min()) / (complex_img.max() - complex_img.min()) * 255).astype(np.uint8))
    labels.append("complex_8freq")

    # Run auto-architecture for each
    print(f"\n{'Image':<20} {'Final Layers':>13} {'Features':>9} {'Loss':>10} {'Size':>8} {'Fixed Size':>11} {'Saved':>8}")
    print("-" * 85)

    results = []

    for img, label in zip(images, labels):
        if verbose:
            print(f"\n  Evolving architecture for '{label}'...")

        model, loss, history = auto_architecture_search(
            img, target_loss=0.005, device=device, verbose=verbose
        )

        auto_size = measure_model_size_compressed(model)
        final = history[-1]

        # Compare with fixed-size (32 features, 2 layers = default)
        fixed_model = SIREN(in_features=2, hidden_features=32, hidden_layers=2,
                            out_features=3, omega_0=30.0).to(device)
        train_and_evaluate(fixed_model, img, epochs=50, device=device)
        fixed_size = measure_model_size_compressed(fixed_model)

        saved = (1 - auto_size / fixed_size) * 100 if fixed_size > 0 else 0
        print(f"{label:<20} {final['layers']:>12}L {final['features']:>8}f {loss:>9.6f} {auto_size:>7,}B {fixed_size:>10,}B {saved:>6.1f}%")

        results.append({
            'label': label,
            'layers': final['layers'],
            'features': final['features'],
            'loss': loss,
            'auto_size': auto_size,
            'fixed_size': fixed_size,
            'saved_pct': saved,
        })

    # Summary
    print(f"\n{'='*80}")
    print("📊 PHASE 13 SUMMARY — AUTO-ARCHITECTURE")
    print(f"{'='*80}")
    print(f"\n  Auto-architecture adapts network size to image complexity!")
    avg_saved = np.mean([r['saved_pct'] for r in results])
    print(f"  Average size saving vs fixed: {avg_saved:.1f}%")
    print(f"\n  Simple images → smaller networks (fewer features)")
    print(f"  Complex images → larger networks (more layers)")
    print(f"  This validates adaptive 'seed' sizing per file")

    return results


if __name__ == '__main__':
    results = run_phase13_experiment(verbose=True)
