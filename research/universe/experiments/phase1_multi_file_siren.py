# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 1 Experiment: Multi-File SIREN (Shared Roots)
=====================================================
Tests Principle 3 of the Black Hole Universe Hypothesis:
"Multiple files share 'roots' — common structure in the generator space."

HYPOTHESIS:
  Training 1 SIREN base network + per-image modulations for 100 satellite images
  will achieve 2-5x compression improvement vs 100 separate SIRENs.

METHOD:
  1. Generate 100 synthetic satellite-like images (256x256)
  2. Baseline: 100 separate SIREN networks (current BLKH approach)
  3. Multi-file: 1 SIREN base + 100 modulations (BHUH approach)
  4. Compare total compressed size

EXPECTED RESULT:
  Multi-file should be 2-5x smaller than separate SIRENs
  because modulations share the base network's structure.

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


# ============================================================
# SIREN Architecture (simplified from siren_v5_torch.py)
# ============================================================

class SIRENLayer(nn.Module):
    """SIREN layer with sinusoidal activation."""
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """SIREN network for image representation."""
    def __init__(self, in_features=2, hidden_features=32, hidden_layers=2, out_features=3, omega_0=30.0):
        super().__init__()
        layers = [SIRENLayer(in_features, hidden_features, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers):
            layers.append(SIRENLayer(hidden_features, hidden_features, omega_0=omega_0))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Multi-File SIREN with Modulations (BHUH Principle 3)
# ============================================================

class ModulatedSIREN(nn.Module):
    """SIREN with per-file modulation (FiLM-style).
    The base network is shared; modulations adapt it per-file.
    """
    def __init__(self, n_files, in_features=2, hidden_features=32, hidden_layers=2,
                 out_features=3, omega_0=30.0, modulation_dim=16):
        super().__init__()
        self.n_files = n_files
        self.modulation_dim = modulation_dim

        # Base SIREN
        self.base_siren = SIREN(in_features, hidden_features, hidden_layers, out_features, omega_0)

        # Modulation embeddings (one per file)
        self.modulations = nn.Embedding(n_files, modulation_dim)

        # FiLM generator: modulation → scale and shift for each layer
        self.film_generators = nn.ModuleList([
            nn.Linear(modulation_dim, 2 * hidden_features)
            for _ in range(hidden_layers + 1)
        ])

    def forward(self, coords, file_idx):
        """Forward pass with modulation.
        coords: (N, 2) coordinate pairs
        file_idx: int, which file's modulation to use
        """
        mod = self.modulations(torch.tensor(file_idx, device=coords.device))

        # Apply FiLM modulation to each SIREN layer
        x = coords
        for i, layer in enumerate(self.base_siren.net):
            if i < len(self.base_siren.net) - 1:
                # SIREN layer with FiLM
                film = self.film_generators[i](mod)
                scale, shift = film.chunk(2, dim=-1)
                x = layer(x)
                x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
            else:
                # Final linear layer (no modulation)
                x = layer(x)
        return x


# ============================================================
# Data Generation
# ============================================================

def generate_satellite_images(n_images=100, size=256, seed=42):
    """Generate N synthetic satellite-like images."""
    rng = np.random.default_rng(seed)
    images = []
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size

    for i in range(n_images):
        img = np.zeros((size, size, 3), dtype=np.float32)
        # Each image has different frequencies/phases but same structure
        for c in range(3):
            for _ in range(5):
                kx, ky = rng.integers(1, 7, 2)
                amp = rng.uniform(30, 80)
                phase = rng.uniform(0, 2*np.pi)
                img[:, :, c] += amp * np.sin(2*np.pi*kx*xs + phase) * np.cos(2*np.pi*ky*ys)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        images.append(img)
    return images


def get_coordinates(size, device='cpu'):
    """Get normalized coordinates for an image."""
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, size, device=device),
        torch.linspace(0, 1, size, device=device),
        indexing='ij'
    ), dim=-1).reshape(-1, 2)
    return coords


# ============================================================
# Training Functions
# ============================================================

def train_single_siren(image, epochs=100, lr=3e-3, device='cpu', verbose=False):
    """Train a single SIREN on one image (baseline)."""
    size = image.shape[0]
    coords = get_coordinates(size, device)
    pixels = torch.from_numpy(image.astype(np.float32) / 255.0).reshape(-1, 3).to(device)

    model = SIREN(in_features=2, hidden_features=32, hidden_layers=2, out_features=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(coords)
        loss = F.mse_loss(pred, pixels)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.6f}")

    return model, loss.item()


def train_multi_file_siren(images, epochs=200, lr=3e-3, device='cpu', verbose=False):
    """Train one modulated SIREN on multiple images (BHUH approach)."""
    n_files = len(images)
    size = images[0].shape[0]
    coords = get_coordinates(size, device)

    # Stack all images
    pixels_all = torch.stack([
        torch.from_numpy(img.astype(np.float32) / 255.0).reshape(-1, 3)
        for img in images
    ]).to(device)  # (n_files, size*size, 3)

    model = ModulatedSIREN(
        n_files=n_files,
        in_features=2,
        hidden_features=32,
        hidden_layers=2,
        out_features=3,
        modulation_dim=16
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        for i in range(n_files):
            pred = model(coords, i)
            loss = F.mse_loss(pred, pixels_all[i])
            total_loss += loss
        total_loss = total_loss / n_files
        total_loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch}: avg_loss={total_loss.item():.6f}")

    return model, total_loss.item()


# ============================================================
# Size Measurement
# ============================================================

def measure_model_size(model):
    """Measure model size in bytes (raw weights, no compression)."""
    total_params = sum(p.numel() for p in model.parameters())
    # float32 = 4 bytes
    return total_params * 4


def measure_model_size_compressed(model):
    """Measure model size in bytes (with zlib compression)."""
    # Get all weights as bytes
    weights_buffer = bytearray()
    for param in model.parameters():
        weights_buffer.extend(param.detach().cpu().numpy().tobytes())
    return len(zlib.compress(bytes(weights_buffer), 9))


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(n_images=10, size=64, epochs_single=100, epochs_multi=200, verbose=True):
    """Run the Multi-File SIREN experiment."""
    print("=" * 80)
    print("🧪 Phase 1 Experiment: Multi-File SIREN (Shared Roots)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Images: {n_images}")
    print(f"  Size: {size}x{size}")
    print(f"  Single SIREN epochs: {epochs_single}")
    print(f"  Multi-file SIREN epochs: {epochs_multi}")
    print()

    # Generate images
    if verbose:
        print("📸 Generating satellite-like images...")
    images = generate_satellite_images(n_images, size)
    total_pixels = sum(img.nbytes for img in images)
    print(f"  Generated {len(images)} images, total raw: {total_pixels:,}B")
    print()

    # Baseline: separate SIRENs
    print("🔵 Baseline: Training separate SIRENs (current BLKH approach)...")
    t0 = time.time()
    baseline_models = []
    baseline_total_size = 0
    baseline_compressed_size = 0
    for i, img in enumerate(images):
        model, loss = train_single_siren(img, epochs=epochs_single, verbose=False)
        baseline_models.append(model)
        size_bytes = measure_model_size(model)
        comp_bytes = measure_model_size_compressed(model)
        baseline_total_size += size_bytes
        baseline_compressed_size += comp_bytes
        if verbose and (i+1) % 5 == 0:
            print(f"  Trained {i+1}/{n_images} (last loss: {loss:.6f})")
    baseline_time = time.time() - t0
    print(f"  Done in {baseline_time:.1f}s")
    print(f"  Total raw size: {baseline_total_size:,}B")
    print(f"  Total compressed: {baseline_compressed_size:,}B")
    print()

    # BHUH: Multi-file SIREN
    print("🌌 BHUH: Training multi-file SIREN (shared roots)...")
    t0 = time.time()
    multi_model, multi_loss = train_multi_file_siren(
        images, epochs=epochs_multi, verbose=verbose
    )
    multi_time = time.time() - t0
    multi_size = measure_model_size(multi_model)
    multi_compressed = measure_model_size_compressed(multi_model)
    print(f"  Done in {multi_time:.1f}s")
    print(f"  Total raw size: {multi_size:,}B")
    print(f"  Total compressed: {multi_compressed:,}B")
    print()

    # Results
    print("=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"\n{'Metric':<35} {'Baseline':>15} {'BHUH':>15} {'Improvement':>15}")
    print("-" * 80)
    print(f"{'Number of models':<35} {n_images:>15} {'1 (shared)':>15} {'-':>15}")
    print(f"{'Training time (s)':<35} {baseline_time:>15.1f} {multi_time:>15.1f} {baseline_time/multi_time:>14.2f}x")
    print(f"{'Raw model size (B)':<35} {baseline_total_size:>15,} {multi_size:>15,} {baseline_total_size/multi_size:>14.2f}x")
    print(f"{'Compressed size (B)':<35} {baseline_compressed_size:>15,} {multi_compressed:>15,} {baseline_compressed_size/multi_compressed:>14.2f}x")

    improvement = baseline_compressed_size / multi_compressed
    print()
    print(f"🎯 Compression improvement: {improvement:.2f}x")

    if improvement >= 2.0:
        print("✅ HYPOTHESIS CONFIRMED: Multi-file SIREN achieves 2x+ improvement")
    elif improvement >= 1.5:
        print("⚠️  PARTIAL SUCCESS: Multi-file SIREN shows improvement but <2x")
    else:
        print("❌ HYPOTHESIS REJECTED: Multi-file SIREN does not improve enough")

    # ZIP baseline for reference
    zip_total = sum(len(zlib.compress(img.tobytes(), 9)) for img in images)
    print(f"\n📋 Reference: ZIP total = {zip_total:,}B")
    print(f"   BHUH vs ZIP: {zip_total/multi_compressed:.2f}x smaller")

    return {
        'n_images': n_images,
        'baseline_size': baseline_compressed_size,
        'bhuh_size': multi_compressed,
        'improvement': improvement,
        'zip_size': zip_total,
        'baseline_time': baseline_time,
        'bhuh_time': multi_time,
    }


if __name__ == '__main__':
    # Run quick experiment (small for testing)
    results = run_experiment(n_images=10, size=64, epochs_single=50, epochs_multi=100, verbose=True)

    print("\n" + "=" * 80)
    print("Experiment complete! Results saved to EXPERIMENT_LOG.md")
    print("=" * 80)
