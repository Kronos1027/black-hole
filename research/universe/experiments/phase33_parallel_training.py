# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 33: Parallel Universe Training (Multi-Worker)
=====================================================
Tests whether Multi-File SIREN training can be parallelized.

CONCEPT:
  Training on N files sequentially is slow. Can we parallelize?
  - Each file's gradient can be computed independently
  - Gradients are accumulated and applied once per epoch
  - This is "data parallelism" across files

HYPOTHESIS:
  Using torch.multiprocessing, we can achieve 2-4x speedup on multi-core
  machines by parallelizing per-file gradient computation.

METHOD:
  1. Generate 20 images
  2. Baseline: sequential training (current approach)
  3. Parallel: distribute files across workers
  4. Measure speedup and verify same result quality

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import ModulatedSIREN, get_coordinates, generate_satellite_images, measure_model_size_compressed


def train_sequential(images, epochs=80, device='cpu', verbose=False):
    """Sequential training (baseline)."""
    n = len(images)
    size = images[0].shape[0]
    coords = get_coordinates(size, device)
    pixels_all = torch.stack([
        torch.from_numpy(img.astype(np.float32) / 255.0).reshape(-1, 3)
        for img in images
    ]).to(device)

    model = ModulatedSIREN(n_files=n, hidden_features=32, hidden_layers=2, modulation_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    t0 = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        for i in range(n):
            pred = model(coords, i)
            total_loss += F.mse_loss(pred, pixels_all[i])
        total_loss = total_loss / n
        total_loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"  Seq Epoch {epoch}: loss={total_loss.item():.6f}")
    dt = time.time() - t0

    return model, total_loss.item(), dt


def train_batched(images, epochs=80, device='cpu', verbose=False):
    """Batched training (all files in one forward pass via batching).

    Instead of looping over files, we batch all files together by
    tiling coordinates and using a batch dimension for modulations.
    This leverages PyTorch's vectorized operations.
    """
    n = len(images)
    size = images[0].shape[0]
    coords = get_coordinates(size, device)

    # Stack all pixel targets: (n, size*size, 3)
    pixels_all = torch.stack([
        torch.from_numpy(img.astype(np.float32) / 255.0).reshape(-1, 3)
        for img in images
    ]).to(device)

    model = ModulatedSIREN(n_files=n, hidden_features=32, hidden_layers=2, modulation_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Pre-compute all modulations
    t0 = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Vectorized: compute all files at once
        # Tile coords for all files: (n * size*size, 2)
        all_coords = coords.repeat(n, 1)  # (n * size*size, 2)
        # File indices: (n * size*size,)
        file_indices = torch.arange(n, device=device).repeat_interleave(size * size)

        # Get all modulations
        all_mods = model.modulations(file_indices)  # (n * size*size, mod_dim)

        # Apply FiLM and forward (vectorized)
        x = all_coords
        for i, layer in enumerate(model.base_siren.net):
            if i < len(model.base_siren.net) - 1:
                # FiLM per-file
                film = model.film_generators[i](all_mods)
                scale, shift = film.chunk(2, dim=-1)
                x = layer(x)
                x = x * (1 + scale) + shift
            else:
                x = layer(x)

        # Reshape: (n, size*size, 3)
        pred = x.reshape(n, size * size, 3)

        # Compute loss across all files
        total_loss = F.mse_loss(pred, pixels_all)
        total_loss.backward()
        optimizer.step()

        if verbose and epoch % 20 == 0:
            print(f"  Batch Epoch {epoch}: loss={total_loss.item():.6f}")

    dt = time.time() - t0
    return model, total_loss.item(), dt


def run_phase33_experiment(verbose=True):
    """Run Phase 33 Parallel Training experiment."""
    print("=" * 80)
    print("🧪 Phase 33: Parallel Universe Training (Vectorized Batch)")
    print("=" * 80)

    device = 'cpu'
    n_files = 20
    size = 64

    print(f"\n📦 Generating {n_files} images @ {size}x{size}...")
    images = generate_satellite_images(n_files, size, seed=42)

    # Baseline: sequential
    print(f"\n🔵 Baseline: Sequential training (loop over files)...")
    seq_model, seq_loss, seq_time = train_sequential(images, epochs=60, device=device, verbose=verbose)
    seq_size = measure_model_size_compressed(seq_model)
    print(f"  Time: {seq_time:.1f}s, Loss: {seq_loss:.6f}, Size: {seq_size:,}B")

    # Vectorized batch
    print(f"\n🌌 Vectorized batch training (all files in one forward pass)...")
    batch_model, batch_loss, batch_time = train_batched(images, epochs=60, device=device, verbose=verbose)
    batch_size = measure_model_size_compressed(batch_model)
    print(f"  Time: {batch_time:.1f}s, Loss: {batch_loss:.6f}, Size: {batch_size:,}B")

    # Results
    speedup = seq_time / max(batch_time, 0.001)
    quality_ratio = batch_loss / max(seq_loss, 1e-10)

    print(f"\n{'='*80}")
    print("📊 PHASE 33 RESULTS — PARALLEL/VECTORIZED TRAINING")
    print(f"{'='*80}")
    print(f"\n  {'Method':<30} {'Time':>8} {'Loss':>10} {'Size':>8} {'Speedup':>8}")
    print(f"  {'-'*68}")
    print(f"  {'Sequential (loop)':<30} {seq_time:>6.1f}s {seq_loss:>9.6f} {seq_size:>7,}B {'1.00x':>7}")
    print(f"  {'Vectorized batch':<30} {batch_time:>6.1f}s {batch_loss:>9.6f} {batch_size:>7,}B {speedup:>6.2f}x")

    print(f"\n  📋 Analysis:")
    print(f"  - Speedup: {speedup:.2f}x {'✅ faster' if speedup > 1 else '❌ slower'}")
    print(f"  - Quality: {'similar' if abs(quality_ratio - 1) < 0.2 else 'different'} (ratio: {quality_ratio:.2f})")
    print(f"  - Size: {'same' if abs(seq_size - batch_size) < 100 else 'different'} ({seq_size} vs {batch_size})")

    if speedup > 1.5:
        print(f"\n  ✅ Vectorized training is {speedup:.1f}x faster!")
        print(f"     PyTorch's batched operations leverage CPU vectorization (SIMD)")
        print(f"     GPU acceleration would amplify this further (CUDA cores)")
    elif speedup > 0.8:
        print(f"\n  ⚠️  Similar speed (overhead offsets gains at this scale)")
    else:
        print(f"\n  ❌ Vectorized is slower (memory overhead at small scale)")

    print(f"\n  📋 Applications:")
    print(f"  - Large-scale dataset compression (1000+ files)")
    print(f"  - Real-time compression (live streaming)")
    print(f"  - GPU acceleration (CUDA batched SIREN)")

    return {
        'seq_time': seq_time,
        'batch_time': batch_time,
        'speedup': speedup,
        'seq_loss': seq_loss,
        'batch_loss': batch_loss,
    }


if __name__ == '__main__':
    results = run_phase33_experiment(verbose=True)
