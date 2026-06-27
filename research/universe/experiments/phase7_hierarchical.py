# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 7: Hierarchical Universes
=================================
Tests the deepest implication of BHUH — can universes be nested?

CONCEPT:
  A "universe" is a shared-root model for N files.
  A "meta-universe" is a shared-root model for N universes.

  Level 0: Individual files (images, audio, text)
  Level 1: Universe — shared base + per-file modulations (Phase 1-6)
  Level 2: Meta-universe — shared meta-base + per-universe modulations

  If this works, we have a FRACTAL compression structure:
  - Each level shares structure with siblings
  - Total size grows logarithmically, not linearly

HYPOTHESIS:
  Training a meta-universe on 5 groups of 10 images each (50 total)
  will be smaller than 1 universe on 50 images, because the meta-base
  captures cross-group structure.

METHOD:
  1. Generate 5 groups of 10 images (different frequency ranges per group)
  2. Level 1: Train 5 separate universes (one per group)
  3. Level 2: Train 1 meta-universe sharing structure across all 5
  4. Compare total sizes

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
from phase1_multi_file_siren import (
    SIREN, SIRENLayer, ModulatedSIREN, get_coordinates,
    train_multi_file_siren, train_single_siren, measure_model_size_compressed,
    generate_satellite_images
)


class MetaUniverse(nn.Module):
    """Meta-universe: generates modulations for sub-universes.

    Structure:
    - Meta-base: shared across all sub-universes
    - Sub-universe modulations: per-group adaptation
    - File modulations: per-file within each sub-universe
    """
    def __init__(self, n_groups, files_per_group, hidden_features=32,
                 hidden_layers=2, mod_dim=16, meta_mod_dim=8):
        super().__init__()
        self.n_groups = n_groups
        self.files_per_group = files_per_group

        # Base SIREN (shared across ALL files in ALL groups)
        self.base_siren = SIREN(in_features=2, hidden_features=hidden_features,
                                hidden_layers=hidden_layers, out_features=3, omega_0=30.0)

        # Per-group modulation (meta-level)
        self.group_mods = nn.Embedding(n_groups, meta_mod_dim)
        self.group_film = nn.Linear(meta_mod_dim, 2 * hidden_features)

        # Per-file modulation (within each group)
        total_files = n_groups * files_per_group
        self.file_mods = nn.Embedding(total_files, mod_dim)
        self.file_film = nn.Linear(mod_dim, 2 * hidden_features)

    def forward(self, coords, group_idx, file_idx):
        """Forward pass with hierarchical modulation.
        coords: (N, 2)
        group_idx: int (which group/universe)
        file_idx: int (which file within all files)
        """
        # Group-level modulation
        g_mod = self.group_mods(torch.tensor(group_idx, device=coords.device))
        g_film = self.group_film(g_mod)
        g_scale, g_shift = g_film.chunk(2, dim=-1)

        # File-level modulation
        f_mod = self.file_mods(torch.tensor(file_idx, device=coords.device))
        f_film = self.file_film(f_mod)
        f_scale, f_shift = f_film.chunk(2, dim=-1)

        # Apply hierarchical FiLM: group first, then file
        x = coords
        for i, layer in enumerate(self.base_siren.net):
            if i < len(self.base_siren.net) - 1:
                x = layer(x)
                # Group modulation
                x = x * (1 + g_scale.unsqueeze(0) * 0.5) + g_shift.unsqueeze(0) * 0.5
                # File modulation
                x = x * (1 + f_scale.unsqueeze(0) * 0.5) + f_shift.unsqueeze(0) * 0.5
            else:
                x = layer(x)
        return x


def generate_grouped_images(n_groups=5, files_per_group=10, size=64, seed=42):
    """Generate image groups with different characteristics per group."""
    all_groups = []

    for g in range(n_groups):
        rng = np.random.default_rng(seed + g * 100)
        group = []
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size

        # Each group has different frequency range
        freq_min = 1 + g * 2  # group 0: 1-3, group 1: 3-5, etc.
        freq_max = freq_min + 2

        for i in range(files_per_group):
            img = np.zeros((size, size, 3), dtype=np.float32)
            for c in range(3):
                for _ in range(3):
                    kx = rng.integers(freq_min, freq_max + 1)
                    ky = rng.integers(freq_min, freq_max + 1)
                    amp = rng.uniform(40, 80)
                    phase = rng.uniform(0, 2 * np.pi)
                    img[:, :, c] += amp * np.sin(2 * np.pi * kx * xs + phase) * np.cos(2 * np.pi * ky * ys)
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            group.append(img)
        all_groups.append(group)

    return all_groups


def train_meta_universe(groups, epochs=150, device='cpu', verbose=False):
    """Train hierarchical meta-universe."""
    n_groups = len(groups)
    files_per_group = len(groups[0])
    total_files = n_groups * files_per_group

    model = MetaUniverse(
        n_groups=n_groups,
        files_per_group=files_per_group,
        hidden_features=32,
        hidden_layers=2,
        mod_dim=16,
        meta_mod_dim=8
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Prepare data
    all_data = []
    file_idx = 0
    for g, group in enumerate(groups):
        for img in group:
            size = img.shape[0]
            coords = get_coordinates(size, device)
            pixels = torch.from_numpy(img.astype(np.float32) / 255.0).reshape(-1, 3).to(device)
            all_data.append((coords, pixels, g, file_idx))
            file_idx += 1

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        for coords, pixels, g_idx, f_idx in all_data:
            pred = model(coords, g_idx, f_idx)
            loss = F.mse_loss(pred, pixels)
            total_loss += loss
        total_loss = total_loss / total_files
        total_loss.backward()
        optimizer.step()

        if verbose and epoch % 30 == 0:
            print(f"  Meta-universe Epoch {epoch}: avg_loss={total_loss.item():.6f}")

    return model, total_loss.item()


def run_phase7_experiment(verbose=True):
    """Run Phase 7 Hierarchical Universes experiment."""
    print("=" * 80)
    print("🧪 Phase 7: Hierarchical Universes (Universe of Universes)")
    print("=" * 80)

    device = 'cpu'
    n_groups = 5
    files_per_group = 10
    size = 64

    # Generate grouped images
    print(f"\n📦 Generating {n_groups} groups × {files_per_group} images @ {size}x{size}...")
    groups = generate_grouped_images(n_groups, files_per_group, size)
    all_images = [img for group in groups for img in group]
    total_raw = sum(img.nbytes for img in all_images)
    total_zip = sum(len(zlib.compress(img.tobytes(), 9)) for img in all_images)
    print(f"  Total: {len(all_images)} images, Raw: {total_raw:,}B, ZIP: {total_zip:,}B")

    # Baseline 1: Separate SIRENs (50 models)
    print(f"\n🔵 Baseline 1: {len(all_images)} separate SIRENs...")
    t0 = time.time()
    separate_total = 0
    for img in all_images:
        model, _ = train_single_siren(img, epochs=50, device=device, verbose=False)
        separate_total += measure_model_size_compressed(model)
    separate_time = time.time() - t0
    print(f"  Total: {separate_total:,}B in {separate_time:.1f}s")

    # Baseline 2: Single flat universe (Phase 1 approach, 50 files)
    print(f"\n🔵 Baseline 2: Single flat universe (50 files)...")
    t0 = time.time()
    flat_model, _ = train_multi_file_siren(all_images, epochs=100, device=device, verbose=False)
    flat_size = measure_model_size_compressed(flat_model)
    flat_time = time.time() - t0
    print(f"  Total: {flat_size:,}B in {flat_time:.1f}s")

    # Baseline 3: 5 separate universes (one per group)
    print(f"\n🔵 Baseline 3: {n_groups} separate universes (one per group)...")
    t0 = time.time()
    multi_universe_total = 0
    for g, group in enumerate(groups):
        model, _ = train_multi_file_siren(group, epochs=100, device=device, verbose=False)
        sz = measure_model_size_compressed(model)
        multi_universe_total += sz
        if verbose:
            print(f"  Group {g}: {sz:,}B")
    multi_time = time.time() - t0
    print(f"  Total: {multi_universe_total:,}B in {multi_time:.1f}s")

    # BHUH: Meta-universe (hierarchical)
    print(f"\n🌌 BHUH: Meta-universe (hierarchical, {n_groups} groups × {files_per_group} files)...")
    t0 = time.time()
    meta_model, meta_loss = train_meta_universe(groups, epochs=150, device=device, verbose=verbose)
    meta_size = measure_model_size_compressed(meta_model)
    meta_time = time.time() - t0
    print(f"  Total: {meta_size:,}B in {meta_time:.1f}s")

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 7 RESULTS — HIERARCHICAL UNIVERSES")
    print(f"{'='*80}")
    print(f"\n  {'Method':<45} {'Size':>10} {'vs Separate':>12}")
    print(f"  {'-'*70}")
    print(f"  {'Separate SIRENs (50 models)':<45} {separate_total:>9,}B {'1.00x':>11}")
    print(f"  {'Flat universe (1 model, 50 files)':<45} {flat_size:>9,}B {separate_total/flat_size:>10.2f}x")
    print(f"  {'5 separate universes (5 models)':<45} {multi_universe_total:>9,}B {separate_total/multi_universe_total:>10.2f}x")
    print(f"  {'Meta-universe (hierarchical)':<45} {meta_size:>9,}B {separate_total/meta_size:>10.2f}x")

    print(f"\n📋 Comparison:")
    print(f"  Flat universe:      {flat_size:,}B ({separate_total/flat_size:.2f}x vs separate)")
    print(f"  5 separate universes: {multi_universe_total:,}B ({separate_total/multi_universe_total:.2f}x vs separate)")
    print(f"  Meta-universe:      {meta_size:,}B ({separate_total/meta_size:.2f}x vs separate)")

    if meta_size < flat_size:
        print(f"\n✅ Meta-universe BEATS flat universe! ({flat_size/meta_size:.2f}x smaller)")
        print(f"   Hierarchical structure captures cross-group patterns!")
    else:
        print(f"\n⚠️  Meta-universe is larger than flat ({meta_size/flat_size:.2f}x)")
        print(f"   Hierarchical overhead may not pay off at this scale")

    vs_zip = total_zip / meta_size
    print(f"\n  Meta-universe vs ZIP: {vs_zip:.2f}x smaller")

    return {
        'separate': separate_total,
        'flat': flat_size,
        'multi_universe': multi_universe_total,
        'meta': meta_size,
        'vs_separate': separate_total / meta_size,
        'vs_zip': vs_zip,
    }


if __name__ == '__main__':
    results = run_phase7_experiment(verbose=True)
