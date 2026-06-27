# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 9: Three-Level Hierarchical Universe
===========================================
Extends Phase 7's hierarchical discovery to THREE levels.

Phase 7 showed meta-universe (2 levels) beats flat universe by 1.32x.
Does a meta-meta-universe (3 levels) continue to improve?

HIERARCHY:
  Level 0: Individual files (images)
  Level 1: Universe (shared base + per-file modulations)
  Level 2: Meta-universe (shared meta-base + per-group + per-file mods)
  Level 3: Meta-meta-universe (shared meta-meta-base + per-supergroup + per-group + per-file mods)

HYPOTHESIS:
  Each additional hierarchical level adds ~1.2-1.3x improvement.
  3 levels should be ~1.5x better than flat (1.32 * 1.15).

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
    SIREN, SIRENLayer, get_coordinates,
    train_single_siren, train_multi_file_siren,
    measure_model_size_compressed, generate_satellite_images
)


class ThreeLevelUniverse(nn.Module):
    """3-level hierarchical universe: meta-meta → meta → file."""
    def __init__(self, n_supergroups, n_groups_per_sg, files_per_group,
                 hidden_features=32, hidden_layers=2):
        super().__init__()
        self.n_sg = n_supergroups
        self.n_g = n_groups_per_sg
        self.n_f = files_per_group
        total_files = n_supergroups * n_groups_per_sg * files_per_group

        # Base SIREN (shared across ALL files)
        self.base = SIREN(in_features=2, hidden_features=hidden_features,
                          hidden_layers=hidden_layers, out_features=3, omega_0=30.0)

        # Level 3: supergroup modulation
        self.sg_mods = nn.Embedding(n_supergroups, 8)
        self.sg_film = nn.Linear(8, 2 * hidden_features)

        # Level 2: group modulation
        total_groups = n_supergroups * n_groups_per_sg
        self.g_mods = nn.Embedding(total_groups, 8)
        self.g_film = nn.Linear(8, 2 * hidden_features)

        # Level 1: file modulation
        self.f_mods = nn.Embedding(total_files, 16)
        self.f_film = nn.Linear(16, 2 * hidden_features)

    def forward(self, coords, sg_idx, g_idx, f_idx):
        # Level 3 modulation
        sg_mod = self.sg_mods(torch.tensor(sg_idx, device=coords.device))
        sg_film = self.sg_film(sg_mod)
        sg_s, sg_sh = sg_film.chunk(2, dim=-1)

        # Level 2 modulation
        g_mod = self.g_mods(torch.tensor(g_idx, device=coords.device))
        g_film = self.g_film(g_mod)
        g_s, g_sh = g_film.chunk(2, dim=-1)

        # Level 1 modulation
        f_mod = self.f_mods(torch.tensor(f_idx, device=coords.device))
        f_film = self.f_film(f_mod)
        f_s, f_sh = f_film.chunk(2, dim=-1)

        x = coords
        for i, layer in enumerate(self.base.net):
            if i < len(self.base.net) - 1:
                x = layer(x)
                # Hierarchical FiLM (weighted, deepest first)
                w = 0.33
                x = x * (1 + sg_s.unsqueeze(0) * w) + sg_sh.unsqueeze(0) * w
                x = x * (1 + g_s.unsqueeze(0) * w) + g_sh.unsqueeze(0) * w
                x = x * (1 + f_s.unsqueeze(0) * w) + f_sh.unsqueeze(0) * w
            else:
                x = layer(x)
        return x


def generate_3level_data(n_sg=2, n_g=5, n_f=5, size=64, seed=42):
    """Generate 3-level grouped images."""
    all_data = []
    for sg in range(n_sg):
        for g in range(n_g):
            rng = np.random.default_rng(seed + sg * 1000 + g * 100)
            ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
            freq_min = 1 + sg * 3 + g  # different freq per supergroup+group
            freq_max = freq_min + 2

            for f in range(n_f):
                img = np.zeros((size, size, 3), dtype=np.float32)
                for c in range(3):
                    for _ in range(3):
                        kx = rng.integers(freq_min, freq_max + 1)
                        ky = rng.integers(freq_min, freq_max + 1)
                        amp = rng.uniform(40, 80)
                        phase = rng.uniform(0, 2 * np.pi)
                        img[:, :, c] += amp * np.sin(2 * np.pi * kx * xs + phase) * np.cos(2 * np.pi * ky * ys)
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                all_data.append((sg, g, f, img))
    return all_data


def train_3level_universe(data, epochs=120, device='cpu', verbose=False):
    """Train 3-level hierarchical universe."""
    n_sg = max(d[0] for d in data) + 1
    n_g_per_sg = max(d[1] for d in data) + 1
    n_f_per_g = max(d[2] for d in data) + 1
    total_files = len(data)

    model = ThreeLevelUniverse(
        n_supergroups=n_sg,
        n_groups_per_sg=n_g_per_sg,
        files_per_group=n_f_per_g,
        hidden_features=32, hidden_layers=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Prepare data
    train_data = []
    for sg, g, f, img in data:
        size = img.shape[0]
        coords = get_coordinates(size, device)
        pixels = torch.from_numpy(img.astype(np.float32) / 255.0).reshape(-1, 3).to(device)
        # Compute global indices
        g_global = sg * n_g_per_sg + g
        f_global = sg * n_g_per_sg * n_f_per_g + g * n_f_per_g + f
        train_data.append((coords, pixels, sg, g_global, f_global))

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        for coords, pixels, sg, g, f in train_data:
            pred = model(coords, sg, g, f)
            total_loss += F.mse_loss(pred, pixels)
        total_loss = total_loss / total_files
        total_loss.backward()
        optimizer.step()

        if verbose and epoch % 30 == 0:
            print(f"  3-Level Epoch {epoch}: loss={total_loss.item():.6f}")

    return model, total_loss.item()


def run_phase9_experiment(verbose=True):
    """Run Phase 9 Three-Level Hierarchical experiment."""
    print("=" * 80)
    print("🧪 Phase 9: Three-Level Hierarchical Universe")
    print("=" * 80)

    device = 'cpu'
    n_sg, n_g, n_f = 2, 5, 5  # 2 supergroups × 5 groups × 5 files = 50 files

    # Generate data
    print(f"\n📦 Generating {n_sg}×{n_g}×{n_f} = {n_sg*n_g*n_f} images...")
    data = generate_3level_data(n_sg, n_g, n_f, size=64, seed=42)
    all_images = [d[3] for d in data]
    total_zip = sum(len(zlib.compress(img.tobytes(), 9)) for img in all_images)
    print(f"  Total: {len(all_images)} images, ZIP: {total_zip:,}B")

    # Baseline 1: Separate SIRENs
    print(f"\n🔵 Baseline 1: {len(all_images)} separate SIRENs...")
    separate_total = 0
    for img in all_images:
        m, _ = train_single_siren(img, epochs=40, device=device, verbose=False)
        separate_total += measure_model_size_compressed(m)
    print(f"  Total: {separate_total:,}B")

    # Baseline 2: Flat universe (Phase 1)
    print(f"\n🔵 Baseline 2: Flat universe (1 level)...")
    flat_model, _ = train_multi_file_siren(all_images, epochs=80, device=device, verbose=False)
    flat_size = measure_model_size_compressed(flat_model)
    print(f"  Total: {flat_size:,}B")

    # BHUH Level 2: Meta-universe (Phase 7)
    print(f"\n🔵 Baseline 3: Meta-universe (2 levels)...")
    from phase7_hierarchical import MetaUniverse, train_meta_universe, generate_grouped_images

    # Regroup data for 2-level (treat supergroups as groups)
    groups_2level = []
    for sg in range(n_sg):
        sg_imgs = [d[3] for d in data if d[0] == sg]
        # Split into n_g groups of n_f
        for g in range(n_g):
            start = g * n_f
            groups_2level.append(sg_imgs[start:start+n_f])

    meta_model, _ = train_meta_universe(groups_2level, epochs=100, device=device, verbose=False)
    meta_size = measure_model_size_compressed(meta_model)
    print(f"  Total: {meta_size:,}B")

    # BHUH Level 3: Three-level universe
    print(f"\n🌌 BHUH: Three-level universe (meta-meta-universe)...")
    t0 = time.time()
    three_model, three_loss = train_3level_universe(data, epochs=120, device=device, verbose=verbose)
    three_time = time.time() - t0
    three_size = measure_model_size_compressed(three_model)
    print(f"  Total: {three_size:,}B in {three_time:.1f}s")

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 9 RESULTS — THREE-LEVEL HIERARCHY")
    print(f"{'='*80}")
    print(f"\n  {'Method':<45} {'Size':>10} {'vs Separate':>12}")
    print(f"  {'-'*70}")
    print(f"  {'Separate SIRENs':<45} {separate_total:>9,}B {'1.00x':>11}")
    print(f"  {'Level 1: Flat universe':<45} {flat_size:>9,}B {separate_total/flat_size:>10.2f}x")
    print(f"  {'Level 2: Meta-universe':<45} {meta_size:>9,}B {separate_total/meta_size:>10.2f}x")
    print(f"  {'Level 3: Three-level universe':<45} {three_size:>9,}B {separate_total/three_size:>10.2f}x")

    print(f"\n📋 Per-level improvement:")
    if flat_size > 0 and meta_size > 0 and three_size > 0:
        print(f"  Level 1 → 2: {flat_size/meta_size:.2f}x smaller")
        print(f"  Level 2 → 3: {meta_size/three_size:.2f}x smaller")
        print(f"  Level 1 → 3: {flat_size/three_size:.2f}x smaller (total)")

    if three_size < meta_size:
        print(f"\n✅ Three-level BEATS two-level! ({meta_size/three_size:.2f}x smaller)")
        print(f"   Hierarchical scaling continues to level 3!")
    else:
        print(f"\n⚠️  Three-level is larger than two-level ({three_size/meta_size:.2f}x)")
        print(f"   Diminishing returns at level 3 (overhead > benefit)")

    print(f"\n  vs ZIP: {total_zip/three_size:.2f}x smaller")

    return {
        'separate': separate_total,
        'flat': flat_size,
        'meta': meta_size,
        'three_level': three_size,
        'vs_separate': separate_total / three_size,
        'vs_zip': total_zip / three_size,
    }


if __name__ == '__main__':
    results = run_phase9_experiment(verbose=True)
