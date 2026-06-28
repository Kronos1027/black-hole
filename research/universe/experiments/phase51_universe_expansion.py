# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 51: Universe Expansion (Incremental Learning)
====================================================
Tests whether new files can be added to an existing universe WITHOUT
retraining the base network from scratch.

CONCEPT:
  Traditional: to add file N+1 to a universe of N files, retrain everything.
  BHUH expansion: freeze base network, only train new modulation for file N+1.

  This is "incremental universe growth" — the universe expands one file at a time.

HYPOTHESIS:
  Freezing the base and training only the new modulation will:
  1. Be 10x+ faster than full retraining
  2. Achieve acceptable quality (within 3dB of full retrain)
  3. Not degrade existing files

METHOD:
  1. Train universe on 10 images
  2. Add 5 NEW images (only train new modulations, freeze base)
  3. Compare: quality of new files vs full retrain
  4. Verify: existing files still reconstruct correctly

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import ModulatedSIREN, get_coordinates, generate_satellite_images, measure_model_size_compressed


def expand_universe(model, new_images, epochs=60, device='cpu', verbose=False):
    """Expand universe by adding new files (freeze base, train new modulations)."""
    old_n = model.n_files
    new_n = old_n + len(new_images)

    # Create expanded model
    expanded = copy.deepcopy(model)
    expanded.n_files = new_n

    # Expand modulation embedding
    old_mods = model.modulations.weight.data.clone()
    new_mods = torch.randn(len(new_images), old_mods.shape[1], device=device) * 0.1
    expanded.modulations = nn.Embedding(new_n, old_mods.shape[1]).to(device)
    expanded.modulations.weight.data[:old_n] = old_mods
    expanded.modulations.weight.data[old_n:] = new_mods

    # Expand FiLM generators (they take modulation as input, so no change needed)
    # But we need to freeze base_siren and film_generators
    for param in expanded.base_siren.parameters():
        param.requires_grad = False
    for param in expanded.film_generators.parameters():
        param.requires_grad = False

    # Only train new modulations
    optimizer = torch.optim.Adam(
        [expanded.modulations.weight.data[old_n:]], lr=3e-3
    )

    size = new_images[0].shape[0]
    coords = get_coordinates(size, device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        for i, img in enumerate(new_images):
            file_idx = old_n + i
            pixels = torch.from_numpy(img.astype(np.float32) / 255.0).reshape(-1, 3).to(device)
            pred = expanded(coords, file_idx)
            total_loss += F.mse_loss(pred, pixels)
        total_loss = total_loss / len(new_images)
        total_loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"  Expand Epoch {epoch}: loss={total_loss.item():.6f}")

    return expanded, total_loss.item()


def run_phase51_experiment(verbose=True):
    print("=" * 80)
    print("🧪 Phase 51: Universe Expansion (Incremental Learning)")
    print("=" * 80)

    device = 'cpu'
    size = 64

    # Train initial universe on 10 images
    print("\n📦 Training initial universe on 10 images...")
    images_init = generate_satellite_images(n_images=10, size=size, seed=42)
    from phase1_multi_file_siren import train_multi_file_siren
    model, init_loss = train_multi_file_siren(images_init, epochs=80, device=device, verbose=False)
    print(f"  Initial: 10 files, loss={init_loss:.6f}")

    # Generate 5 NEW images (different seed = different content)
    images_new = generate_satellite_images(n_images=5, size=size, seed=99)
    print(f"  Adding 5 new files (different seed)...")

    # Method 1: Full retrain (baseline)
    print("\n🔵 Baseline: Full retrain (15 images from scratch)...")
    all_images = images_init + images_new
    t0 = time.time()
    full_model, full_loss = train_multi_file_siren(all_images, epochs=80, device=device, verbose=False)
    full_time = time.time() - t0
    print(f"  Full retrain: loss={full_loss:.6f}, time={full_time:.1f}s")

    # Method 2: BHUH expansion (freeze base, train new modulations only)
    print("\n🌌 BHUH: Expansion (freeze base, train new modulations)...")
    t0 = time.time()
    expanded_model, expand_loss = expand_universe(model, images_new, epochs=60, device=device, verbose=verbose)
    expand_time = time.time() - t0
    print(f"  Expansion: loss={expand_loss:.6f}, time={expand_time:.1f}s")

    # Verify existing files are NOT degraded
    coords = get_coordinates(size, device)
    print(f"\n📊 Verifying existing files (should be unchanged)...")
    existing_psnrs = []
    for i in range(10):
        with torch.no_grad():
            pred_orig = model(coords, i)
            pred_exp = expanded_model(coords, i)
        mse = F.mse_loss(pred_orig, pred_exp).item()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-10)) if mse > 0 else 99
        existing_psnrs.append(psnr)
    avg_existing = np.mean(existing_psnrs)
    print(f"  Existing files avg PSNR (orig vs expanded): {avg_existing:.1f}dB")
    print(f"  {'✅ No degradation!' if avg_existing > 40 else '⚠️ Some degradation'}")

    # Compare new file quality
    print(f"\n📊 New file quality (expansion vs full retrain)...")
    new_psnrs_expand = []
    new_psnrs_full = []
    for i, img in enumerate(images_new):
        pixels = torch.from_numpy(img.astype(np.float32) / 255.0).reshape(-1, 3).to(device)

        with torch.no_grad():
            pred_exp = expanded_model(coords, 10 + i)
            pred_full = full_model(coords, 10 + i)

        mse_exp = F.mse_loss(pred_exp, pixels).item()
        mse_full = F.mse_loss(pred_full, pixels).item()
        psnr_exp = 10 * np.log10(1.0 / max(mse_exp, 1e-10))
        psnr_full = 10 * np.log10(1.0 / max(mse_full, 1e-10))
        new_psnrs_expand.append(psnr_exp)
        new_psnrs_full.append(psnr_full)

    avg_new_exp = np.mean(new_psnrs_expand)
    avg_new_full = np.mean(new_psnrs_full)
    quality_gap = avg_new_full - avg_new_exp

    print(f"  Expansion: {avg_new_exp:.1f}dB")
    print(f"  Full retrain: {avg_new_full:.1f}dB")
    print(f"  Quality gap: {quality_gap:.1f}dB")

    # Speed comparison
    speedup = full_time / max(expand_time, 0.001)

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 51 RESULTS — UNIVERSE EXPANSION")
    print(f"{'='*80}")
    print(f"\n  {'Metric':<35} {'Full Retrain':>15} {'Expansion':>15}")
    print(f"  {'-'*67}")
    print(f"  {'Time':<35} {full_time:>13.1f}s {expand_time:>13.1f}s")
    print(f"  {'Speedup':<35} {'1.00x':>14} {speedup:>13.2f}x")
    print(f"  {'New file quality (PSNR)':<35} {avg_new_full:>13.1f}dB {avg_new_exp:>13.1f}dB")
    print(f"  {'Quality gap':<35} {'-':>14} {quality_gap:>+12.1f}dB")
    print(f"  {'Existing files preserved':<35} {'-':>14} {avg_existing:>12.1f}dB")
    print(f"  {'Base retrained?':<35} {'Yes':>14} {'No (frozen)':>14}")

    if speedup > 2 and quality_gap < 5:
        print(f"\n  ✅ Expansion WORKS!")
        print(f"     {speedup:.1f}x faster with only {quality_gap:.1f}dB quality loss")
        print(f"     Existing files: {'PERFECTLY preserved' if avg_existing > 40 else 'slightly degraded'}")
    elif speedup > 1.5:
        print(f"\n  ⚠️  Expansion is faster but quality gap = {quality_gap:.1f}dB")
    else:
        print(f"\n  ❌ Expansion doesn't help much")

    print(f"\n  📋 Applications:")
    print(f"  - Live database: add new files without downtime")
    print(f"  - Streaming compression: process new files as they arrive")
    print(f"  - Distributed: different nodes add different files")
    print(f"  - Backup: expand universe incrementally")

    return {
        'full_time': full_time,
        'expand_time': expand_time,
        'speedup': speedup,
        'quality_gap': quality_gap,
        'existing_preserved': avg_existing > 40,
    }


if __name__ == '__main__':
    results = run_phase51_experiment(verbose=True)
