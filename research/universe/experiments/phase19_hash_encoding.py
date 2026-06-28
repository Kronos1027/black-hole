# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 19: Hash-Based Neural Encoding (Instant-NGP Style)
==========================================================
Tests whether hash-based positional encoding (from Instant-NGP by
NVIDIA, 2022) can replace SIREN's sinusoidal activation for faster
training and smaller seeds.

CONCEPT:
  Instant-NGP uses a multiresolution hash table instead of sinusoidal
  features. This is:
  - 10-100x faster to train
  - Fixed-size (hash table doesn't grow)
  - Potentially smaller seeds

HYPOTHESIS:
  Hash-based encoding will train 5-10x faster than SIREN while achieving
  similar compression, because hash lookups are O(1) vs sinusoidal O(N).

METHOD:
  1. Implement simplified hash grid encoding
  2. Train on same images as Phase 1
  3. Compare: training time, seed size, reconstruction quality

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, generate_satellite_images, measure_model_size_compressed


class HashGridEncoder(nn.Module):
    """Simplified multiresolution hash grid (Instant-NGP inspired)."""
    def __init__(self, n_levels=4, table_size=512, hash_dim=2, out_dim=32):
        super().__init__()
        self.n_levels = n_levels
        self.table_size = table_size
        self.hash_dim = hash_dim

        # Per-resolution hash tables
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(table_size, hash_dim) * 0.01)
            for _ in range(n_levels)
        ])

        # Resolution per level (geometric progression)
        self.resolutions = [2 ** (2 + i) for i in range(n_levels)]  # 4, 8, 16, 32...

        # Output projection
        self.proj = nn.Linear(n_levels * hash_dim, out_dim)

    def hash_coords(self, coords, resolution, table_size):
        """Hash 2D coordinates to table index."""
        # Scale to resolution grid
        scaled = coords * resolution
        # Floor to get grid cell
        cell = torch.floor(scaled).long()
        # Hash: simple spatial hash
        idx = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663)
        idx = idx % table_size
        idx = torch.clamp(idx, 0, table_size - 1)
        return idx

    def forward(self, coords):
        """coords: (N, 2) in [0, 1]"""
        features = []
        for i, res in enumerate(self.resolutions):
            idx = self.hash_coords(coords, res, self.table_size)
            feat = self.tables[i][idx]  # (N, hash_dim)
            features.append(feat)

        # Concatenate all levels
        x = torch.cat(features, dim=-1)  # (N, n_levels * hash_dim)
        return self.proj(x)


class HashSIREN(nn.Module):
    """SIREN with hash grid input encoding instead of sinusoidal."""
    def __init__(self, hidden_features=32, hidden_layers=2, n_hash_levels=4):
        super().__init__()
        self.encoder = HashGridEncoder(
            n_levels=n_hash_levels,
            table_size=512,
            hash_dim=2,
            out_dim=hidden_features
        )
        # Simple MLP after encoding
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_features, 3))
        self.decoder = nn.Sequential(*layers)

    def forward(self, coords):
        x = self.encoder(coords)
        return self.decoder(x)


def train_hash_siren(image, epochs=100, lr=3e-3, device='cpu', verbose=False):
    """Train hash-based SIREN on image."""
    size = image.shape[0]
    coords = get_coordinates(size, device)
    pixels = torch.from_numpy(image.astype(np.float32) / 255.0).reshape(-1, 3).to(device)

    model = HashSIREN(hidden_features=32, hidden_layers=2, n_hash_levels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(coords)
        loss = F.mse_loss(pred, pixels)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"  HashSIREN Epoch {epoch}: loss={loss.item():.6f}")

    return model, loss.item()


def run_phase19_experiment(verbose=True):
    """Run Phase 19 Hash-Based Encoding experiment."""
    print("=" * 80)
    print("🧪 Phase 19: Hash-Based Neural Encoding (Instant-NGP Style)")
    print("=" * 80)

    device = 'cpu'

    # Generate test images
    print("\n📦 Generating test images...")
    images = generate_satellite_images(n_images=10, size=128, seed=42)
    total_raw = sum(img.nbytes for img in images)
    total_zip = sum(len(zlib.compress(img.tobytes(), 9)) for img in images)
    print(f"  10 images @ 128x128, Raw: {total_raw:,}B, ZIP: {total_zip:,}B")

    # Baseline: SIREN
    print("\n🔵 Baseline: Standard SIREN...")
    from phase1_multi_file_siren import train_single_siren
    siren_total = 0
    siren_total_time = 0
    siren_losses = []
    for img in images:
        t0 = time.time()
        model, loss = train_single_siren(img, epochs=80, device=device, verbose=False)
        dt = time.time() - t0
        siren_total += measure_model_size_compressed(model)
        siren_total_time += dt
        siren_losses.append(loss)
    siren_avg_loss = np.mean(siren_losses)
    print(f"  Total: {siren_total:,}B, Time: {siren_total_time:.1f}s, Avg loss: {siren_avg_loss:.6f}")

    # HashSIREN
    print("\n🌌 HashSIREN (Instant-NGP inspired)...")
    hash_total = 0
    hash_total_time = 0
    hash_losses = []
    for img in images:
        t0 = time.time()
        model, loss = train_hash_siren(img, epochs=80, device=device, verbose=False)
        dt = time.time() - t0
        hash_total += measure_model_size_compressed(model)
        hash_total_time += dt
        hash_losses.append(loss)
    hash_avg_loss = np.mean(hash_losses)
    print(f"  Total: {hash_total:,}B, Time: {hash_total_time:.1f}s, Avg loss: {hash_avg_loss:.6f}")

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 19 RESULTS — HASH vs SINUSOIDAL ENCODING")
    print(f"{'='*80}")
    print(f"\n  {'Method':<25} {'Total Size':>12} {'Train Time':>12} {'Avg Loss':>12} {'vs ZIP':>10}")
    print(f"  {'-'*75}")
    print(f"  {'SIREN (sinusoidal)':<25} {siren_total:>11,}B {siren_total_time:>10.1f}s {siren_avg_loss:>11.6f} {total_zip/siren_total:>9.2f}x")
    print(f"  {'HashSIREN (hash grid)':<25} {hash_total:>11,}B {hash_total_time:>10.1f}s {hash_avg_loss:>11.6f} {total_zip/hash_total:>9.2f}x")

    speedup = siren_total_time / max(hash_total_time, 0.001)
    size_ratio = siren_total / max(hash_total, 1)

    print(f"\n  📋 Comparison:")
    print(f"  - Training speed: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} (HashSIREN vs SIREN)")
    print(f"  - Size: {size_ratio:.2f}x ({'smaller' if size_ratio > 1 else 'larger'})")
    print(f"  - Quality: SIREN loss={siren_avg_loss:.6f}, Hash loss={hash_avg_loss:.6f}")

    if speedup > 1.5:
        print(f"\n  ✅ HashSIREN is {speedup:.1f}x faster to train!")
    elif speedup > 0.8:
        print(f"\n  ⚠️  Similar training speed")
    else:
        print(f"\n  ❌ HashSIREN is slower ({1/speedup:.1f}x)")

    if hash_total < siren_total:
        print(f"  ✅ HashSIREN seeds are {size_ratio:.2f}x smaller!")
    else:
        print(f"  SIREN seeds are {1/size_ratio:.2f}x smaller")

    return {
        'siren_size': siren_total,
        'hash_size': hash_total,
        'siren_time': siren_total_time,
        'hash_time': hash_total_time,
        'speedup': speedup,
        'size_ratio': size_ratio,
    }


if __name__ == '__main__':
    results = run_phase19_experiment(verbose=True)
