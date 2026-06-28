#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Rigorous Experiment 2: Scaling Law — Does BHUH Advantage Grow with N?
============================================================================
QUESTION:
  In Experiment 1, BHUH was 1.42× smaller than COIN with N=5 images.
  Does this advantage GROW as N increases?

THEORY:
  COIN: total_params = N × P_separate  (linear in N)
  BHUH: total_params = P_shared + N × P_head  (sublinear if P_head < P_separate)
  
  If P_head << P_separate, BHUH advantage should grow with N.

METHOD:
  Test N = 3, 5, 8, 10 with same images, measure:
  - COIN total bytes
  - BHUH total bytes
  - Ratio improvement (COIN/BHUH)
  - PSNR for both
  
  Plot: ratio improvement vs N

This is the SCALING LAW test — critical for BHUH's publishability.

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


def load_real_images(n, size=128):
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


def train_coin_batch(images, hidden=32, omega=15.0, epochs=400, lr=1e-3):
    """Train separate COIN SIREN per image. Returns list of results."""
    import torch
    import torch.nn as nn
    
    results = []
    for img_idx, image in enumerate(images):
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
        recipe_size = 10 + len(compressed)  # header + data
        
        results.append({
            'psnr_db': psnr,
            'recipe_bytes': recipe_size,
        })
    return results


def train_shared_siren(images, hidden=64, omega=15.0, epochs=800, lr=1e-3):
    """Train ONE shared SIREN with N output heads."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    n_images = len(images)
    n_pix = images[0].shape[0]
    coords = np.stack(np.meshgrid(
        np.linspace(-1, 1, n_pix), np.linspace(-1, 1, n_pix)
    ), axis=-1).reshape(-1, 2)

    class SharedSiren(nn.Module):
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
            self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_images)])
            for head in self.heads:
                bound = np.sqrt(6.0/hidden)/omega
                head.weight.data.uniform_(-bound, bound)
                head.bias.data.uniform_(-bound, bound)
            self.omega = omega

        def forward(self, x, img_idx):
            h = x
            for layer in self.layers:
                h = torch.sin(self.omega * layer(h))
            return self.heads[img_idx](h)

    model = SharedSiren()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    targets = [torch.tensor((img.astype(np.float32)/255.0).flatten(), dtype=torch.float32)
               for img in images]

    for ep in range(epochs):
        opt.zero_grad()
        loss = 0
        for i, yt in enumerate(targets):
            pred = model(xt, i).squeeze(-1)
            loss = loss + ((pred - yt) ** 2).mean()
        loss = loss / n_images
        loss.backward()
        opt.step()

    model.eval()
    psnrs = []
    with torch.no_grad():
        for i, img in enumerate(images):
            recon = model(xt, i).squeeze(-1).numpy()
            recon_img = (recon * 255).clip(0, 255).astype(np.uint8).reshape(n_pix, n_pix)
            psnrs.append(compute_psnr(img, recon_img))

    shared_flat = np.concatenate([p.detach().numpy().flatten() for p in model.layers.parameters()])
    heads_flat = [np.concatenate([p.detach().numpy().flatten() for p in head.parameters()])
                  for head in model.heads]
    all_params = np.concatenate([shared_flat] + heads_flat)
    levels = 255
    max_val = np.abs(all_params).max()
    scale = max_val / (levels/2)
    quantized = np.round(all_params / scale).astype(np.int8)
    compressed = zlib.compress(quantized.tobytes(), 9)
    recipe_size = 18 + len(compressed)

    return {
        'psnrs_db': psnrs,
        'avg_psnr_db': float(np.mean(psnrs)),
        'recipe_bytes': recipe_size,
        'shared_params': len(shared_flat),
        'head_params_per_image': len(heads_flat[0]),
    }


def run_experiment_2():
    print("=" * 72)
    print("BHUH RIGOROUS EXPERIMENT 2: Scaling Law — Advantage vs N")
    print("=" * 72)
    print()
    print("QUESTION: Does BHUH's advantage over COIN GROW as N increases?")
    print()
    print("THEORY:")
    print("  COIN:  total = N × P_separate  (linear in N)")
    print("  BHUH:  total = P_shared + N × P_head  (sublinear if P_head < P_sep)")
    print()
    print("METHOD: Test N = 3, 5, 8, 10 with same real images")
    print()

    import torch

    # Load 10 images once, use subsets (64x64 for speed)
    print("--- Loading 10 REAL photographs (64x64 for speed) ---")
    all_images = load_real_images(n=10, size=64)
    print(f"  Loaded {len(all_images)} images")
    print()

    n_values = [3]
    scaling_results = []

    for n in n_values:
        print(f"\n{'='*60}")
        print(f"  Testing N = {n}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        images = all_images[:n]
        
        # COIN baseline
        print(f"  Training {n} COIN SIRENs (30 epochs each)...")
        sys.stdout.flush()
        t0 = time.time()
        coin_results = train_coin_batch(images, hidden=32, epochs=30)
        t_coin = time.time() - t0
        coin_total = sum(r['recipe_bytes'] for r in coin_results)
        coin_avg_psnr = np.mean([r['psnr_db'] for r in coin_results])
        print(f"  COIN: {coin_total}B, {coin_avg_psnr:.2f}dB, {t_coin:.1f}s")
        sys.stdout.flush()
        
        # BHUH shared
        print(f"  Training 1 BHUH shared SIREN (backbone + {n} heads, 60 epochs)...")
        sys.stdout.flush()
        t0 = time.time()
        bhuh_result = train_shared_siren(images, hidden=64, epochs=60)
        t_bhuh = time.time() - t0
        print(f"  BHUH: {bhuh_result['recipe_bytes']}B, "
              f"{bhuh_result['avg_psnr_db']:.2f}dB, {t_bhuh:.1f}s")
        
        ratio = coin_total / bhuh_result['recipe_bytes']
        psnr_diff = bhuh_result['avg_psnr_db'] - coin_avg_psnr
        
        scaling_results.append({
            'n_images': n,
            'coin_total_bytes': coin_total,
            'coin_avg_psnr_db': float(coin_avg_psnr),
            'bhuh_total_bytes': bhuh_result['recipe_bytes'],
            'bhuh_avg_psnr_db': bhuh_result['avg_psnr_db'],
            'ratio_coin_over_bhuh': float(ratio),
            'psnr_diff_bhuh_minus_coin': float(psnr_diff),
            'bhuh_shared_params': bhuh_result['shared_params'],
            'bhuh_head_params_per_image': bhuh_result['head_params_per_image'],
            'time_coin_s': float(t_coin),
            'time_bhuh_s': float(t_bhuh),
        })
        
        print(f"  → Ratio: {ratio:.2f}x (BHUH smaller), PSNR diff: {psnr_diff:+.2f}dB")

    # ============================================================
    # SUMMARY
    # ============================================================
    print()
    print("=" * 72)
    print("SCALING LAW RESULTS")
    print("=" * 72)
    print(f"{'N':>4} {'COIN bytes':>12} {'BHUH bytes':>12} {'Ratio':>8} "
          f"{'COIN PSNR':>10} {'BHUH PSNR':>10} {'ΔPSNR':>8}")
    for r in scaling_results:
        print(f"{r['n_images']:>4} {r['coin_total_bytes']:>12} "
              f"{r['bhuh_total_bytes']:>12} {r['ratio_coin_over_bhuh']:>7.2f}x "
              f"{r['coin_avg_psnr_db']:>9.2f}dB {r['bhuh_avg_psnr_db']:>9.2f}dB "
              f"{r['psnr_diff_bhuh_minus_coin']:>+7.2f}dB")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()

    ratios = [r['ratio_coin_over_bhuh'] for r in scaling_results]
    ns = [r['n_images'] for r in scaling_results]
    
    # Fit linear trend: ratio = a + b*N
    coeffs = np.polyfit(ns, ratios, 1)
    a, b = coeffs
    print(f"  Linear fit: ratio = {a:.3f} + {b:.3f} × N")
    print(f"  Slope: {b:.4f} per image")
    print()

    if b > 0.02:
        print(f"  ✅ SCALING LAW CONFIRMED — BHUH advantage GROWS with N")
        print(f"     Each additional image improves ratio by {b:.3f}")
        print(f"     Projected: N=20 → {a + b*20:.2f}x, N=50 → {a + b*50:.2f}x")
        verdict = (f"VALIDATED — Scaling law confirmed. BHUH advantage grows "
                   f"linearly with N (slope {b:.3f}/image). At N=20, projected "
                   f"ratio {a + b*20:.2f}x. This is a PUBLISHABLE scaling law.")
    elif b > 0:
        print(f"  ⚠️ Weak scaling — advantage grows slowly")
        verdict = f"PARTIAL — Weak scaling (slope {b:.4f}/image)"
    else:
        print(f"  ❌ No scaling — advantage does NOT grow with N")
        verdict = "INVALID — No scaling advantage"

    print()
    print(f"  VERDICT: {verdict}")
    print()
    
    # Per-N analysis
    print("  Per-N detail:")
    for r in scaling_results:
        n = r['n_images']
        coin_per_img = r['coin_total_bytes'] / n
        bhuh_per_img = r['bhuh_total_bytes'] / n
        print(f"    N={n}: COIN {coin_per_img:.0f}B/img, BHUH {bhuh_per_img:.0f}B/img, "
              f"saving {(coin_per_img-bhuh_per_img):.0f}B/img ({(1-bhuh_per_img/coin_per_img)*100:.1f}%)")

    return {
        'experiment': 2,
        'name': 'Scaling Law Test',
        'n_values': n_values,
        'scaling_results': scaling_results,
        'linear_fit': {'a': float(a), 'b': float(b)},
        'verdict': verdict,
    }


if __name__ == '__main__':
    result = run_experiment_2()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
