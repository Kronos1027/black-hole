# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 22: Neural Delta Compression (Versioned Files)
======================================================
Tests whether SIREN can compress FILE VERSIONS efficiently.

CONCEPT:
  When a file changes (e.g., source code edit, image retouch),
  the new version shares most structure with the old. Instead of
  compressing from scratch, we DELTA-compress:
  - Train SIREN on version 1 (seed_1)
  - For version 2, fine-tune seed_1 → seed_2
  - Store delta = seed_2 - seed_1 (should be tiny)

  This is "shared roots" across TIME, not just across files.

HYPOTHESIS:
  Delta between SIREN seeds for similar file versions will be much
  smaller than compressing each version independently (2-5x improvement).

METHOD:
  1. Generate base image
  2. Generate 5 versions (progressive edits)
  3. Baseline: 5 independent SIRENs
  4. Delta: SIREN_1 + 4 fine-tuned deltas
  5. Compare sizes

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
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren, measure_model_size_compressed


def generate_versioned_images(size=128, n_versions=5, seed=42):
    """Generate progressive versions of an image (simulating edits)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size

    versions = []
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Version 0: base pattern
    for c in range(3):
        kx, ky = rng.integers(2, 5, 2)
        img[:, :, c] = 60 * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
    versions.append(((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8))

    # Versions 1-4: progressive modifications
    for v in range(1, n_versions):
        # Small change: add one more frequency or adjust amplitude
        for c in range(3):
            kx, ky = rng.integers(1, 8, 2)
            amp = rng.uniform(10, 25)  # small amplitude (small edit)
            img[:, :, c] += amp * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
        versions.append(np.clip(img, 0, 255).astype(np.uint8).clip(0, 255))

    # Normalize each version
    normalized = []
    for v in versions:
        v_float = v.astype(np.float32)
        v_norm = ((v_float - v_float.min()) / (v_float.max() - v_float.min() + 1e-8) * 255).astype(np.uint8)
        normalized.append(v_norm)

    return normalized


def fine_tune_siren(base_model, new_image, epochs=30, lr=1e-3, device='cpu'):
    """Fine-tune existing SIREN on new image version."""
    import copy
    model = copy.deepcopy(base_model)

    size = new_image.shape[0]
    coords = get_coordinates(size, device)
    pixels = torch.from_numpy(new_image.astype(np.float32) / 255.0).reshape(-1, 3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(coords)
        loss = F.mse_loss(pred, pixels)
        loss.backward()
        optimizer.step()

    return model, loss.item()


def compute_delta_size(base_model, new_model):
    """Compute compressed size of weight delta (new - base)."""
    base_params = {k: v.detach().cpu().numpy() for k, v in base_model.named_parameters()}
    new_params = {k: v.detach().cpu().numpy() for k, v in new_model.named_parameters()}

    delta_buf = bytearray()
    for k in sorted(base_params.keys()):
        delta = new_params[k] - base_params[k]
        # Quantize delta to int16 (smaller range)
        max_abs = max(np.abs(delta).max(), 1e-8)
        scale = max_abs / 32767.0
        delta_q = np.round(delta / scale).astype(np.int16)
        delta_buf.extend(delta_q.tobytes())

    return len(zlib.compress(bytes(delta_buf), 9))


def run_phase22_experiment(verbose=True):
    """Run Phase 22 Neural Delta Compression experiment."""
    print("=" * 80)
    print("🧪 Phase 22: Neural Delta Compression (Versioned Files)")
    print("=" * 80)

    device = 'cpu'
    size = 128
    n_versions = 5

    # Generate versioned images
    print(f"\n📦 Generating {n_versions} progressive image versions...")
    versions = generate_versioned_images(size, n_versions, seed=42)
    total_raw = sum(v.nbytes for v in versions)
    total_zip = sum(len(zlib.compress(v.tobytes(), 9)) for v in versions)
    print(f"  {n_versions} versions @ {size}x{size}, Raw: {total_raw:,}B, ZIP: {total_zip:,}B")

    # Baseline: independent SIRENs
    print(f"\n🔵 Baseline: {n_versions} independent SIRENs...")
    t0 = time.time()
    independent_models = []
    independent_total = 0
    for i, img in enumerate(versions):
        model, loss = train_single_siren(img, epochs=80, device=device, verbose=False)
        independent_models.append(model)
        independent_total += measure_model_size_compressed(model)
        if verbose:
            print(f"  Version {i}: loss={loss:.6f}, size={measure_model_size_compressed(model):,}B")
    baseline_time = time.time() - t0
    print(f"  Total: {independent_total:,}B in {baseline_time:.1f}s")

    # Delta: base SIREN + fine-tuned deltas
    print(f"\n🌌 BHUH Delta: Base SIREN + {n_versions-1} fine-tuned deltas...")
    t0 = time.time()

    # Version 0: full SIREN (same as baseline)
    base_model = independent_models[0]
    base_size = measure_model_size_compressed(base_model)

    # Versions 1-4: fine-tune from previous version, store delta
    delta_total = base_size
    prev_model = base_model

    for i in range(1, n_versions):
        # Fine-tune from previous version
        ft_model, ft_loss = fine_tune_siren(prev_model, versions[i], epochs=40, device=device)

        # Compute delta size
        delta_size = compute_delta_size(prev_model, ft_model)
        delta_total += delta_size

        if verbose:
            print(f"  Version {i}: loss={ft_loss:.6f}, delta={delta_size:,}B")

        prev_model = ft_model

    delta_time = time.time() - t0
    print(f"  Total: {delta_total:,}B in {delta_time:.1f}s")

    # Results
    improvement = independent_total / max(delta_total, 1)
    vs_zip = total_zip / max(delta_total, 1)

    print(f"\n{'='*80}")
    print("📊 PHASE 22 RESULTS — NEURAL DELTA COMPRESSION")
    print(f"{'='*80}")
    print(f"\n  {'Method':<40} {'Size':>10} {'vs Independent':>15} {'vs ZIP':>10}")
    print(f"  {'-'*78}")
    print(f"  {'Independent SIRENs (5 models)':<40} {independent_total:>9,}B {'1.00x':>14} {total_zip/independent_total:>9.2f}x")
    print(f"  {'Delta (base + 4 deltas)':<40} {delta_total:>9,}B {improvement:>13.2f}x {vs_zip:>9.2f}x")

    print(f"\n  📋 Breakdown:")
    print(f"  - Base SIREN (v0): {base_size:,}B")
    print(f"  - 4 deltas:        {delta_total - base_size:,}B total ({(delta_total - base_size)//4:,}B avg)")
    print(f"  - Per-delta cost:  {((delta_total - base_size)/4):.0f}B per version update")

    print(f"\n  ⏱️  Training time:")
    print(f"  - Independent: {baseline_time:.1f}s ({baseline_time/n_versions:.1f}s per version)")
    print(f"  - Delta:       {delta_time:.1f}s ({delta_time/n_versions:.1f}s per version)")

    if improvement >= 2.0:
        print(f"\n  ✅ Delta compression achieves {improvement:.2f}x improvement!")
        print(f"     File versions share 'temporal roots' in SIREN weight space")
    elif improvement >= 1.5:
        print(f"\n  ⚠️  Moderate improvement ({improvement:.2f}x)")
    else:
        print(f"\n  ❌ Delta doesn't help much ({improvement:.2f}x)")

    print(f"\n  📋 Applications:")
    print(f"  - Version control: compress file history efficiently")
    print(f"  - Cloud backup: incremental SIREN updates")
    print(f"  - Collaborative editing: share small deltas between users")

    return {
        'independent_total': independent_total,
        'delta_total': delta_total,
        'improvement': improvement,
        'vs_zip': vs_zip,
        'base_size': base_size,
        'delta_avg': (delta_total - base_size) / (n_versions - 1),
    }


if __name__ == '__main__':
    results = run_phase22_experiment(verbose=True)
