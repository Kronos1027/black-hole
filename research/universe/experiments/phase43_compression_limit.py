# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 43: Compression Limit Analysis (Theoretical Maximum)
============================================================
Derives the theoretical maximum compression ratio for BHUH.

CONCEPT:
  What's the BEST possible compression BHUH can achieve?
  - Lower bound: K(x) (Kolmogorov complexity) — incomputable
  - Practical: |SIREN_min| + |residual_min|
  - SIREN_min depends on: signal bandwidth, dynamic range, smoothness
  - Residual_min depends on: SIREN approximation quality

HYPOTHESIS:
  For smooth signals, theoretical max compression = |x| / |SIREN_min|
  where |SIREN_min| ≈ 500B (smallest useful network).
  For 1MB image → 500B = 2000x compression (theoretical limit).

METHOD:
  1. Train progressively smaller SIRENs (16→8→4 features, 2→1→0 layers)
  2. Find minimum viable network size
  3. Compute theoretical max ratio
  4. Compare with practical results

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, measure_model_size_compressed


def train_siren_custom(img, hidden_features, hidden_layers, epochs=80, device='cpu'):
    """Train SIREN with custom architecture."""
    size = img.shape[0]
    coords = get_coordinates(size, device)
    pixels = torch.from_numpy(img.astype(np.float32) / 255.0).reshape(-1, 3).to(device)

    model = SIREN(in_features=2, hidden_features=hidden_features,
                  hidden_layers=hidden_layers, out_features=3, omega_0=30.0).to(device)
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
    return model, loss


def run_phase43_experiment(verbose=True):
    print("=" * 80)
    print("🧪 Phase 43: Compression Limit Analysis (Theoretical Maximum)")
    print("=" * 80)

    device = 'cpu'
    size = 128

    # Generate smooth image
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(2, 5, 2)
            img[:, :, c] += 50 * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    raw_size = img.nbytes
    zip_size = len(zlib.compress(img.tobytes(), 9))

    # Test progressively smaller architectures
    configs = [
        (64, 3, 'large'),
        (32, 2, 'medium'),
        (16, 2, 'small'),
        (16, 1, 'tiny'),
        (8, 1, 'minimal'),
        (8, 0, 'bare'),
        (4, 1, 'micro'),
        (4, 0, 'atomic'),
    ]

    print(f"\n  Image: {size}x{size}, Raw: {raw_size:,}B, ZIP: {zip_size:,}B")
    print(f"\n{'Config':<12} {'Feat':>5} {'Lyr':>4} {'Size':>8} {'Loss':>10} {'PSNR':>8} {'Ratio':>8} {'vs ZIP':>8}")
    print("-" * 70)

    results = []
    for features, layers, name in configs:
        model, loss = train_siren_custom(img, features, layers, epochs=60, device=device)
        comp_size = measure_model_size_compressed(model)
        psnr = 10 * np.log10(1.0 / max(loss, 1e-10))
        ratio = raw_size / max(comp_size, 1)
        vs_zip = zip_size / max(comp_size, 1)

        usable = "✅" if psnr > 20 else "⚠️" if psnr > 15 else "❌"
        print(f"{name:<12} {features:>5} {layers:>4} {comp_size:>7,}B {loss:>9.6f} {psnr:>6.1f}dB {ratio:>6.0f}x {vs_zip:>6.2f}x {usable}")

        results.append({
            'name': name, 'features': features, 'layers': layers,
            'size': comp_size, 'loss': loss, 'psnr': psnr,
            'ratio': ratio, 'vs_zip': vs_zip,
        })

    # Find minimum viable (PSNR > 20dB)
    viable = [r for r in results if r['psnr'] > 20]
    if viable:
        min_viable = min(viable, key=lambda x: x['size'])
        max_ratio = raw_size / min_viable['size']
        print(f"\n{'='*80}")
        print("📊 PHASE 43 SUMMARY — COMPRESSION LIMIT")
        print(f"{'='*80}")
        print(f"\n  Minimum viable network: {min_viable['name']} ({min_viable['features']}f, {min_viable['layers']}L)")
        print(f"  Size: {min_viable['size']:,}B")
        print(f"  PSNR: {min_viable['psnr']:.1f}dB")
        print(f"  Compression ratio: {max_ratio:.0f}x")
        print(f"  vs ZIP: {min_viable['vs_zip']:.2f}x")

        # Theoretical limits for different file sizes
        print(f"\n  📋 Theoretical maximum compression (smooth data):")
        print(f"  {'File Size':>12} {'Min Seed':>10} {'Max Ratio':>10}")
        print(f"  {'-'*35}")
        for fsize in ['1KB', '100KB', '1MB', '10MB', '100MB', '1GB']:
            bytes_map = {'1KB': 1024, '100KB': 102400, '1MB': 1048576,
                        '10MB': 10485760, '100MB': 104857600, '1GB': 1073741824}
            n = bytes_map[fsize]
            ratio = n / min_viable['size']
            print(f"  {fsize:>12} {min_viable['size']:>9,}B {ratio:>8.0f}x")

        print(f"\n  📋 Key insight:")
        print(f"  For smooth/structured data, BHUH compression ratio grows")
        print(f"  LINEARLY with file size (seed stays constant!)")
        print(f"  A 1GB smooth file could theoretically compress to {min_viable['size']:,}B = {1073741824/min_viable['size']:.0f}x!")
        print(f"  This is the power of algorithmic compression: O(1) seed, O(n) data")

    return results


if __name__ == '__main__':
    results = run_phase43_experiment(verbose=True)
