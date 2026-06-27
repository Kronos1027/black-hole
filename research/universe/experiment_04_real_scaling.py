#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Rigorous Experiment 4: REAL Scaling Law Validation
=========================================================
Now that we know COIN at 64×64 is fast (0.24s/image), we can run
the FULL scaling experiment.

Test: N = 3, 5, 8, 10 with REAL photographs
Compare: COIN (separate) vs BHUH (shared SIREN)
Measure: total bytes, avg PSNR, time

This is the DEFINITIVE scaling law test.

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


def train_coin_single(image, hidden=32, omega=15.0, epochs=150, lr=1e-3):
    """COIN: separate SIREN per image."""
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

    return {'psnr_db': psnr, 'recipe_bytes': recipe_size, 'time_s': 0}


def train_shared_siren(images, hidden=64, omega=15.0, epochs=150, lr=1e-3):
    """BHUH: ONE shared SIREN with N output heads."""
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


def run_experiment_4():
    print("=" * 72)
    print("BHUH EXPERIMENT 4: REAL Scaling Law Validation")
    print("=" * 72)
    print()
    print("Test N = 3, 5, 8, 10 with REAL photographs (64×64)")
    print("Compare: COIN (separate) vs BHUH (shared)")
    print()

    import torch

    print("--- Loading 10 REAL photographs (64×64) ---")
    all_images = load_real_images(n=10, size=64)
    print(f"  Loaded {len(all_images)} images")
    print()

    n_values = [3, 5, 8, 10]
    results = []

    for n in n_values:
        print(f"\n{'='*60}")
        print(f"  N = {n}")
        print(f"{'='*60}")
        sys.stdout.flush()

        images = all_images[:n]

        # COIN
        print(f"  COIN: training {n} separate SIRENs (150 epochs each)...")
        sys.stdout.flush()
        t0 = time.time()
        coin_results = []
        for img in images:
            r = train_coin_single(img, hidden=32, epochs=150)
            coin_results.append(r)
        t_coin = time.time() - t0
        coin_total = sum(r['recipe_bytes'] for r in coin_results)
        coin_psnr = np.mean([r['psnr_db'] for r in coin_results])
        print(f"  COIN: {coin_total}B, {coin_psnr:.2f}dB, {t_coin:.1f}s")
        sys.stdout.flush()

        # BHUH shared
        print(f"  BHUH: training 1 shared SIREN (150 epochs)...")
        sys.stdout.flush()
        t0 = time.time()
        bhuh = train_shared_siren(images, hidden=64, epochs=150)
        t_bhuh = time.time() - t0
        print(f"  BHUH: {bhuh['recipe_bytes']}B, {bhuh['avg_psnr_db']:.2f}dB, {t_bhuh:.1f}s")
        sys.stdout.flush()

        ratio = coin_total / bhuh['recipe_bytes']
        psnr_diff = bhuh['avg_psnr_db'] - coin_psnr

        results.append({
            'n': n,
            'coin_bytes': coin_total,
            'coin_psnr': float(coin_psnr),
            'coin_time_s': float(t_coin),
            'bhuh_bytes': bhuh['recipe_bytes'],
            'bhuh_psnr': bhuh['avg_psnr_db'],
            'bhuh_time_s': float(t_bhuh),
            'ratio': float(ratio),
            'psnr_diff': float(psnr_diff),
        })
        print(f"  → Ratio: {ratio:.2f}× (BHUH smaller), PSNR diff: {psnr_diff:+.2f}dB")

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("=" * 72)
    print("SCALING LAW RESULTS — REAL DATA")
    print("=" * 72)
    print(f"{'N':>4} {'COIN B':>8} {'BHUH B':>8} {'Ratio':>7} "
          f"{'COIN dB':>8} {'BHUH dB':>8} {'ΔPSNR':>7} {'COIN t':>7} {'BHUH t':>7}")
    for r in results:
        print(f"{r['n']:>4} {r['coin_bytes']:>8} {r['bhuh_bytes']:>8} "
              f"{r['ratio']:>6.2f}× {r['coin_psnr']:>7.2f} {r['bhuh_psnr']:>7.2f} "
              f"{r['psnr_diff']:>+6.2f} {r['coin_time_s']:>6.1f}s {r['bhuh_time_s']:>6.1f}s")

    # ============================================================
    # Analysis
    # ============================================================
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()

    ns = [r['n'] for r in results]
    ratios = [r['ratio'] for r in results]
    coeffs = np.polyfit(ns, ratios, 1)
    a, b = coeffs
    print(f"  Linear fit: ratio = {a:.3f} + {b:.3f} × N")
    print(f"  Slope: {b:.4f} per image")
    print()

    # Check if ratio increases with N
    increasing = all(ratios[i] < ratios[i+1] for i in range(len(ratios)-1))
    if increasing and b > 0:
        print(f"  ✅ SCALING LAW CONFIRMED — BHUH advantage GROWS with N")
        print(f"     N=3: {ratios[0]:.2f}× → N=10: {ratios[-1]:.2f}×")
        print(f"     Projected N=20: {a + b*20:.2f}×, N=50: {a + b*50:.2f}×")
        verdict = (f"VALIDATED — Scaling law confirmed on real data. "
                   f"BHUH advantage grows from {ratios[0]:.2f}× (N=3) to "
                   f"{ratios[-1]:.2f}× (N=10). Slope: {b:.3f}/image. "
                   f"Projected N=50: {a + b*50:.1f}×. This is PUBLISHABLE.")
    elif b > 0:
        print(f"  ⚠️ Partial scaling — advantage increases but not monotonically")
        verdict = f"PARTIAL — advantage increases (slope {b:.3f}) but not monotonic"
    else:
        print(f"  ❌ No scaling")
        verdict = "INVALID"

    print(f"\n  VERDICT: {verdict}")

    # Check quality
    avg_psnr_diff = np.mean([r['psnr_diff'] for r in results])
    print(f"\n  Average PSNR difference (BHUH - COIN): {avg_psnr_diff:+.2f} dB")
    if abs(avg_psnr_diff) < 3:
        print(f"  ✅ Quality comparable (within 3 dB)")
    else:
        print(f"  ⚠️ Quality difference > 3 dB")

    return {
        'experiment': 4,
        'name': 'REAL Scaling Law',
        'results': results,
        'linear_fit': {'a': float(a), 'b': float(b)},
        'increasing': bool(increasing),
        'verdict': verdict,
    }


if __name__ == '__main__':
    result = run_experiment_4()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
