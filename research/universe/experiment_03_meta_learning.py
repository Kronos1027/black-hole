#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Rigorous Experiment 3: Meta-Learning Compression (COIN++ style)
=====================================================================
Based on: Dupont et al., "COIN++: Neural Compression Across Modalities" (2022)
          ArXiv: 2201.12904

PROBLEM:
  BHUH shared SIREN takes 36s for 5 images (128×128) on CPU.
  COIN takes 20s for same. Both too slow for scaling experiments.

SOLUTION (from COIN++ literature):
  1. META-TRAIN a base SIREN ONCE on a dataset of images
  2. For each NEW image, fit only a small modulation vector (64 floats)
  3. Modulation vector is what gets stored (64 bytes INT8)

EXPECTED SPEEDUP:
  - COIN: 500 epochs × full SIREN (1185 params) = ~4s/image
  - COIN++: 200 epochs × modulation only (64 params) = ~0.5s/image
  - 8-10× faster

EXPECTED COMPRESSION:
  - Per-image: 64 bytes (modulation) vs 1185 bytes (COIN)
  - 18× smaller per image (no shared backbone per-image cost)
  - But: need to store meta-trained base (~5KB once)

BREAK-EVEN:
  - N=1: meta overhead dominates (5000B + 64B vs 1185B)
  - N=10: meta amortized (5000 + 640 vs 11850) = 1.7× smaller
  - N=50: strong advantage (5000 + 3200 vs 59250) = 7.2× smaller

METHOD (rigorous):
  1. Meta-train base SIREN on 5 images (astronaut, camera, cell, coins, moon)
  2. Test on 5 NEW images (page, text, clock, coffee, chelsea)
  3. Compare: COIN (separate) vs BHUH-shared vs BHUH-meta (COIN++ style)
  4. Measure: time, size, PSNR

Author: Darlan Pereira da Silva (Kronos1027)
"""
import os
import sys
import time
import json
import io
import zlib
import struct
import numpy as np
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_real_images(n, size=64):
    """Load n REAL photographs from scikit-image."""
    from skimage.data import (astronaut, camera, cell, coins, moon,
                               page, text, clock, coffee, chelsea)
    sources = [astronaut, camera, cell, coins, moon, page, text, clock, coffee, chelsea]
    images = []
    for i, src_fn in enumerate(sources[:n]):
        arr = src_fn()
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:,:,:3]
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
        pil = Image.fromarray(gray.astype(np.uint8))
        pil = pil.resize((size, size), Image.LANCZOS)
        images.append(np.array(pil))
    return images


def compute_psnr(orig, recon):
    err = orig.astype(np.float64) - recon.astype(np.float64)
    mse = float(np.mean(err ** 2))
    if mse == 0:
        return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def make_modulated_siren(hidden=32, mod_dim=64, omega=15.0):
    """Create SIREN with FiLM modulation (COIN++ style)."""
    import torch
    import torch.nn as nn

    class ModulatedSiren(nn.Module):
        def __init__(self):
            super().__init__()
            # Base layers (meta-trained, shared)
            self.layers = nn.ModuleList()
            d = 2
            for k in range(2):  # 3 layers
                self.layers.append(nn.Linear(d, hidden))
                d = hidden
            self.head = nn.Linear(hidden, 1)
            # FiLM modulation: mod_dim -> scale+bias per layer
            self.film = nn.ModuleList([
                nn.Linear(mod_dim, 2 * hidden) for _ in range(2)
            ])
            self.omega = omega
            # SIREN init
            for i, layer in enumerate(self.layers):
                bound = 1.0 / layer.in_features if i == 0 else np.sqrt(6.0 / hidden) / omega
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.uniform_(-bound, bound)
            bound = np.sqrt(6.0 / hidden) / omega
            self.head.weight.data.uniform_(-bound, bound)
            self.head.bias.data.uniform_(-bound, bound)

        def forward(self, x, z):
            # z: modulation vector (mod_dim,)
            h = x
            for i, layer in enumerate(self.layers):
                film = self.film[i](z)  # (batch, 2*hidden)
                scale = film[:, :hidden]
                bias = film[:, hidden:]
                h = torch.sin(self.omega * layer(h)) * (1 + 0.1 * scale) + 0.1 * bias
            return self.head(h)

    return ModulatedSiren()


def meta_train_base(images, hidden=32, mod_dim=64, epochs=500, lr=1e-3):
    """Meta-train base SIREN on multiple images (COIN++ meta-learning)."""
    import torch
    torch.manual_seed(0)

    n_images = len(images)
    n_pix = images[0].shape[0]
    coords = np.stack(np.meshgrid(
        np.linspace(-1, 1, n_pix), np.linspace(-1, 1, n_pix)
    ), axis=-1).reshape(-1, 2)

    model = make_modulated_siren(hidden, mod_dim)
    # Per-image modulation vectors (to be optimized during meta-training)
    mod_vectors = [torch.zeros(mod_dim, requires_grad=True) for _ in range(n_images)]

    opt = torch.optim.Adam(
        list(model.parameters()) + mod_vectors, lr=lr
    )
    xt = torch.tensor(coords, dtype=torch.float32)
    targets = [torch.tensor((img.astype(np.float32)/255.0).flatten(), dtype=torch.float32)
               for img in images]

    print(f"  Meta-training base on {n_images} images, {epochs} epochs...")
    for ep in range(epochs):
        opt.zero_grad()
        loss = 0
        for i, yt in enumerate(targets):
            z = mod_vectors[i].unsqueeze(0).expand(xt.shape[0], -1)
            pred = model(xt, z).squeeze(-1)
            loss = loss + ((pred - yt) ** 2).mean()
        loss = loss / n_images
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            print(f"    Epoch {ep+1}/{epochs}, loss={float(loss):.6f}")

    return model


def compress_with_meta(model, image, mod_dim=64, epochs=200, lr=5e-3):
    """Compress new image using meta-trained base.
    Only optimize modulation vector (fast).
    """
    import torch

    n_pix = image.shape[0]
    coords = np.stack(np.meshgrid(
        np.linspace(-1, 1, n_pix), np.linspace(-1, 1, n_pix)
    ), axis=-1).reshape(-1, 2)
    target = (image.astype(np.float32) / 255.0).flatten()

    # Freeze base, optimize only modulation
    for p in model.parameters():
        p.requires_grad = False

    z = torch.zeros(mod_dim, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    yt = torch.tensor(target, dtype=torch.float32)

    for ep in range(epochs):
        opt.zero_grad()
        z_batch = z.unsqueeze(0).expand(xt.shape[0], -1)
        pred = model(xt, z_batch).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        z_batch = z.unsqueeze(0).expand(xt.shape[0], -1)
        recon = model(xt, z_batch).squeeze(-1).numpy()
    recon_img = (recon * 255).clip(0, 255).astype(np.uint8).reshape(n_pix, n_pix)
    psnr = compute_psnr(image, recon_img)

    # Restore gradients
    for p in model.parameters():
        p.requires_grad = True

    # Modulation vector is the "recipe" (64 floats)
    z_np = z.detach().numpy()
    # INT8 quantize
    levels = 255
    max_val = np.abs(z_np).max()
    scale = max_val / (levels / 2)
    quantized = np.round(z_np / scale).astype(np.int8)
    compressed = zlib.compress(quantized.tobytes(), 9)
    # Recipe: scale + compressed
    recipe = struct.pack('<f', float(scale)) + compressed

    return {
        'psnr_db': psnr,
        'modulation_bytes': len(recipe),
        'recipe_bytes': len(recipe),  # + base model size (amortized)
    }


def train_coin_single(image, hidden=32, omega=15.0, epochs=200, lr=1e-3):
    """COIN baseline: train separate SIREN per image."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    n_pix = image.shape[0]
    coords = np.stack(np.meshgrid(
        np.linspace(-1, 1, n_pix), np.linspace(-1, 1, n_pix)
    ), axis=-1).reshape(-1, 2)
    target = (image.astype(np.float32) / 255.0).flatten()

    class Siren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            d = 2
            for k in range(2):
                lin = nn.Linear(d, hidden)
                bound = 1.0/d if k == 0 else np.sqrt(6.0/hidden)/omega
                lin.weight.data.uniform_(-bound, bound)
                lin.bias.data.uniform_(-bound, bound)
                self.layers.append(lin)
                d = hidden
            self.head = nn.Linear(hidden, 1)
            bound = np.sqrt(6.0/hidden)/omega
            self.head.weight.data.uniform_(-bound, bound)
            self.head.bias.data.uniform_(-bound, bound)
            self.omega = omega

        def forward(self, x):
            h = x
            for layer in self.layers:
                h = torch.sin(self.omega * layer(h))
            return self.head(h)

    model = Siren()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    yt = torch.tensor(target, dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(xt).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        recon = model(xt).squeeze(-1).numpy()
    recon_img = (recon * 255).clip(0, 255).astype(np.uint8).reshape(n_pix, n_pix)
    psnr = compute_psnr(image, recon_img)

    params = []
    for p in model.parameters():
        params.append(p.detach().numpy().flatten())
    params_flat = np.concatenate(params)
    levels = 255
    max_val = np.abs(params_flat).max()
    scale = max_val / (levels/2)
    quantized = np.round(params_flat / scale).astype(np.int8)
    compressed = zlib.compress(quantized.tobytes(), 9)
    recipe_size = 10 + len(compressed)

    return {'psnr_db': psnr, 'recipe_bytes': recipe_size}


def run_experiment_3():
    print("=" * 72)
    print("BHUH EXPERIMENT 3: Meta-Learning Compression (COIN++ style)")
    print("=" * 72)
    print()
    print("Based on: Dupont et al., COIN++ (2022), ArXiv:2201.12904")
    print()
    print("HYPOTHESIS: Meta-trained base + per-image modulation is 10× faster")
    print("             than training separate SIRENs (COIN).")
    print()

    import torch

    # Use 64×64 for speed
    print("--- Loading 10 REAL photographs (64×64) ---")
    all_images = load_real_images(n=10, size=64)
    train_images = all_images[:5]  # meta-train on these
    test_images = all_images[5:]   # test on NEW images
    print(f"  Meta-train: {len(train_images)} images")
    print(f"  Test: {len(test_images)} NEW images")
    print()

    # ============================================================
    # Step 1: Meta-train base SIREN (ONE TIME COST)
    # ============================================================
    print("--- Step 1: Meta-training base SIREN (one-time cost) ---")
    t0 = time.time()
    base_model = meta_train_base(train_images, hidden=32, mod_dim=64, epochs=300)
    t_meta = time.time() - t0
    print(f"  Meta-training: {t_meta:.1f}s (one-time)")
    print()

    # Estimate base model size (amortized across N test images)
    base_params = sum(int(np.prod(p.shape)) for p in base_model.parameters())
    base_flat = np.concatenate([p.detach().numpy().flatten()
                                 for p in base_model.parameters()])
    levels = 255
    max_val = np.abs(base_flat).max()
    scale = max_val / (levels/2)
    base_quantized = np.round(base_flat / scale).astype(np.int8)
    base_compressed = zlib.compress(base_quantized.tobytes(), 9)
    base_size = len(base_compressed) + 10  # header
    print(f"  Base model: {base_params} params, {base_size}B compressed")
    print(f"  Amortized per image (N=5 test): {base_size/5:.0f}B")
    print()

    # ============================================================
    # Step 2: Compare COIN vs Meta on TEST images
    # ============================================================
    print("--- Step 2: Compressing 5 NEW test images ---")
    print()

    coin_results = []
    meta_results = []

    for i, img in enumerate(test_images):
        print(f"  Image {i+1}/5:")

        # COIN baseline
        t0 = time.time()
        coin_r = train_coin_single(img, hidden=32, epochs=200)
        t_coin = time.time() - t0
        coin_r['time_s'] = t_coin
        coin_results.append(coin_r)
        print(f"    COIN: {coin_r['recipe_bytes']}B, {coin_r['psnr_db']:.2f}dB, {t_coin:.2f}s")

        # Meta (COIN++ style)
        t0 = time.time()
        meta_r = compress_with_meta(base_model, img, mod_dim=64, epochs=200)
        t_meta_img = time.time() - t0
        meta_r['time_s'] = t_meta_img
        meta_r['total_bytes'] = meta_r['recipe_bytes'] + base_size // 5  # amortized
        meta_results.append(meta_r)
        print(f"    Meta: {meta_r['recipe_bytes']}B (+{base_size//5}B amortized), "
              f"{meta_r['psnr_db']:.2f}dB, {t_meta_img:.2f}s")

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("=" * 72)
    print("RESULTS (5 NEW test images, 64×64 grayscale)")
    print("=" * 72)

    coin_total = sum(r['recipe_bytes'] for r in coin_results)
    meta_total = sum(r['total_bytes'] for r in meta_results)
    coin_avg_psnr = np.mean([r['psnr_db'] for r in coin_results])
    meta_avg_psnr = np.mean([r['psnr_db'] for r in meta_results])
    coin_avg_time = np.mean([r['time_s'] for r in coin_results])
    meta_avg_time = np.mean([r['time_s'] for r in meta_results])

    print(f"  {'Method':<20} {'Total Bytes':>12} {'Avg PSNR':>10} {'Avg Time':>10}")
    print(f"  {'COIN (separate)':<20} {coin_total:>12} {coin_avg_psnr:>9.2f}dB {coin_avg_time:>9.2f}s")
    print(f"  {'Meta (COIN++)':<20} {meta_total:>12} {meta_avg_psnr:>9.2f}dB {meta_avg_time:>9.2f}s")
    print()

    size_ratio = coin_total / max(meta_total, 1)
    time_ratio = coin_avg_time / max(meta_avg_time, 1)
    psnr_diff = meta_avg_psnr - coin_avg_psnr

    print(f"  Size ratio (COIN/Meta): {size_ratio:.2f}×")
    print(f"  Time ratio (COIN/Meta): {time_ratio:.2f}×")
    print(f"  PSNR diff (Meta-COIN): {psnr_diff:+.2f}dB")
    print()

    # ============================================================
    # Analysis
    # ============================================================
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()

    if time_ratio > 2 and size_ratio > 1:
        print(f"  ✅ META-LEARNING WORKS:")
        print(f"     {time_ratio:.1f}× faster than COIN")
        print(f"     {size_ratio:.2f}× smaller than COIN (with amortized base)")
        print(f"     PSNR diff: {psnr_diff:+.2f}dB")
        verdict = (f"VALIDATED — Meta-learning (COIN++ style) achieves "
                   f"{time_ratio:.1f}× speedup and {size_ratio:.2f}× compression "
                   f"vs COIN on real photographs. This solves the compute problem "
                   "and enables scaling experiments.")
    elif time_ratio > 2:
        print(f"  ⚠️ FASTER but not smaller:")
        print(f"     {time_ratio:.1f}× faster, but {size_ratio:.2f}× size (base overhead)")
        verdict = f"PARTIAL — {time_ratio:.1f}× faster but size not better at N=5"
    else:
        print(f"  ❌ Meta-learning does not help here")
        verdict = "INVALID — Meta-learning not effective"

    print(f"\n  VERDICT: {verdict}")
    print()
    print("  SIGNIFICANCE:")
    print("    If meta-learning works, we can now run N=10, 20, 50 experiments")
    print("    in reasonable time, enabling full scaling law validation.")

    return {
        'experiment': 3,
        'name': 'Meta-Learning Compression',
        'based_on': 'COIN++ (Dupont et al., 2022)',
        'meta_train_time_s': float(t_meta),
        'base_size_bytes': int(base_size),
        'coin': {'total_bytes': int(coin_total), 'avg_psnr_db': float(coin_avg_psnr),
                 'avg_time_s': float(coin_avg_time)},
        'meta': {'total_bytes': int(meta_total), 'avg_psnr_db': float(meta_avg_psnr),
                 'avg_time_s': float(meta_avg_time)},
        'size_ratio': float(size_ratio),
        'time_ratio': float(time_ratio),
        'psnr_diff': float(psnr_diff),
        'verdict': verdict,
    }


if __name__ == '__main__':
    result = run_experiment_3()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
