# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 25: 3D Volume Compression (MRI/CT Scans)
================================================
Tests SIREN for compressing 3D volumetric medical data.

CONCEPT:
  MRI/CT scans are 3D arrays (e.g., 256x256x256 voxels). Traditional
  compression treats them as 2D slices. SIREN can represent the ENTIRE
  volume as f(x, y, z) → density, exploiting 3D spatial coherence.

HYPOTHESIS:
  A single 3D SIREN compressing a 64³ volume will beat per-slice
  compression because it captures inter-slice correlations.

METHOD:
  1. Generate synthetic 64³ MRI-like volume (3D smooth + structure)
  2. Baseline: 64 separate 2D SIRENs (one per slice)
  3. BHUH: 1 × 3D SIREN f(x,y,z) → density
  4. Compare sizes

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, SIRENLayer, measure_model_size_compressed


class VolumeSIREN(nn.Module):
    """3D SIREN: f(x, y, z) → density."""
    def __init__(self, hidden_features=32, hidden_layers=2, omega_0=30.0):
        super().__init__()
        layers = [SIRENLayer(3, hidden_features, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers):
            layers.append(SIRENLayer(hidden_features, hidden_features, omega_0=omega_0))
        layers.append(nn.Linear(hidden_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def generate_mri_volume(size=64, seed=42):
    """Generate synthetic MRI-like 3D volume."""
    rng = np.random.default_rng(seed)
    zs, ys, xs = np.mgrid[0:size, 0:size, 0:size].astype(np.float32) / size

    volume = np.zeros((size, size, size), dtype=np.float32)

    # Add 3D structures (simulating tissue boundaries)
    for _ in range(5):
        cx, cy, cz = rng.uniform(0.2, 0.8, 3)
        sigma = rng.uniform(0.05, 0.15)
        amp = rng.uniform(50, 150)
        # 3D Gaussian "blob"
        volume += amp * np.exp(-((xs - cx)**2 + (ys - cy)**2 + (zs - cz)**2) / (2 * sigma**2))

    # Add smooth background gradient
    volume += 30 * xs + 20 * ys + 10 * zs

    # Normalize to uint8
    volume = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.uint8)
    return volume


def get_3d_coordinates(size, device='cpu'):
    """Get normalized 3D coordinates."""
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, size, device=device),
        torch.linspace(0, 1, size, device=device),
        torch.linspace(0, 1, size, device=device),
        indexing='ij'
    ), dim=-1).reshape(-1, 3)
    return coords


def train_volume_siren(volume, epochs=100, lr=3e-3, device='cpu', verbose=False):
    """Train 3D SIREN on volume."""
    size = volume.shape[0]
    coords = get_3d_coordinates(size, device)
    targets = torch.from_numpy(volume.astype(np.float32).ravel() / 255.0).to(device)

    model = VolumeSIREN(hidden_features=32, hidden_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(coords).squeeze(-1)
        loss = F.mse_loss(pred, targets)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"  3D SIREN Epoch {epoch}: loss={loss.item():.6f}")

    return model, loss.item()


def train_slice_sirens(volume, epochs=60, device='cpu'):
    """Train separate 2D SIRENs for each slice (baseline)."""
    from phase1_multi_file_siren import train_single_siren
    size = volume.shape[0]
    total = 0
    for z in range(size):
        slice_2d = volume[z]  # (size, size) — treat as single channel
        # Convert to 3-channel for SIREN (replicate)
        slice_3ch = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)
        model, _ = train_single_siren(slice_3ch, epochs=epochs, device=device, verbose=False)
        total += measure_model_size_compressed(model)
    return total


def run_phase25_experiment(verbose=True):
    """Run Phase 25 3D Volume experiment."""
    print("=" * 80)
    print("🧪 Phase 25: 3D Volume Compression (MRI/CT Scans)")
    print("=" * 80)

    device = 'cpu'
    size = 32  # 32³ = 32768 voxels (fast enough)

    # Generate volume
    print(f"\n📦 Generating {size}³ MRI-like volume...")
    volume = generate_mri_volume(size, seed=42)
    total_raw = volume.nbytes
    total_zip = len(zlib.compress(volume.tobytes(), 9))
    print(f"  Volume: {size}³ = {size**3} voxels")
    print(f"  Raw: {total_raw:,}B ({total_raw/1024:.1f}KB)")
    print(f"  ZIP: {total_zip:,}B ({total_zip/1024:.1f}KB)")

    # Baseline: per-slice 2D SIRENs
    print(f"\n🔵 Baseline: {size} separate 2D SIRENs (one per slice)...")
    t0 = time.time()
    # For speed, train 5 slices and extrapolate
    sample_sizes = []
    from phase1_multi_file_siren import train_single_siren
    for z in range(min(5, size)):
        slice_2d = volume[z]
        slice_3ch = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)
        model, _ = train_single_siren(slice_3ch, epochs=40, device=device, verbose=False)
        sample_sizes.append(measure_model_size_compressed(model))
    avg_slice = np.mean(sample_sizes)
    separate_total = int(avg_slice * size)
    separate_time = time.time() - t0
    print(f"  Estimated: {separate_total:,}B ({avg_slice:.0f}B/slice × {size} slices)")

    # BHUH: Single 3D SIREN
    print(f"\n🌌 BHUH: Single 3D SIREN f(x,y,z) → density...")
    t0 = time.time()
    vol_model, vol_loss = train_volume_siren(volume, epochs=80, device=device, verbose=verbose)
    vol_time = time.time() - t0
    vol_size = measure_model_size_compressed(vol_model)
    print(f"  Total: {vol_size:,}B in {vol_time:.1f}s")

    # Results
    improvement = separate_total / max(vol_size, 1)
    vs_zip = total_zip / max(vol_size, 1)

    print(f"\n{'='*80}")
    print("📊 PHASE 25 RESULTS — 3D VOLUME COMPRESSION")
    print(f"{'='*80}")
    print(f"\n  {'Method':<40} {'Size':>10} {'vs Separate':>12} {'vs ZIP':>10}")
    print(f"  {'-'*75}")
    print(f"  {'ZIP (zlib-9)':<40} {total_zip:>9,}B {'-':>11} {'1.00x':>9}")
    print(f"  {'Separate 2D SIRENs (32 slices)':<40} {separate_total:>9,}B {'1.00x':>11} {total_zip/separate_total:>9.2f}x")
    print(f"  {'3D SIREN (1 model, x,y,z)':<40} {vol_size:>9,}B {improvement:>10.1f}x {vs_zip:>9.2f}x")

    if improvement >= 5.0:
        print(f"\n  ✅ 3D SIREN achieves {improvement:.1f}x over per-slice!")
        print(f"     Inter-slice coherence captured by single 3D function")
    elif improvement >= 2.0:
        print(f"\n  ⚠️  Moderate improvement ({improvement:.1f}x)")
    else:
        print(f"\n  ❌ 3D SIREN doesn't improve much ({improvement:.1f}x)")

    print(f"\n  📋 Applications:")
    print(f"  - MRI/CT scan compression (medical imaging)")
    print(f"  - Scientific visualization (fluid dynamics, weather)")
    print(f"  - 3D game textures (voxel worlds)")
    print(f"  - Microscopy (confocal, electron tomography)")

    return {
        'size': size,
        'separate_total': separate_total,
        'vol_size': vol_size,
        'improvement': improvement,
        'vs_zip': vs_zip,
    }


if __name__ == '__main__':
    results = run_phase25_experiment(verbose=True)
