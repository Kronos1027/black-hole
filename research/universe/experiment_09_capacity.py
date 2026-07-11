#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Experiment 9: Backbone Capacity vs Quality
=================================================
PROBLEM: At N=100, PSNR dropped to 17 dB. Hypothesis: backbone too small.
TEST: Does larger backbone (h=128, h=256) improve quality at N=100?

If YES: capacity was the bottleneck → use bigger backbone
If NO: architecture is fundamentally limited → need different approach

Also test: at what N does each backbone size start failing?
"""
import os, sys, time, json, zlib, struct
import numpy as np
from PIL import Image

def load_crops(n, size=64):
    from skimage.data import (astronaut, camera, cell, coins, moon,
                               page, text, clock, coffee, chelsea)
    sources = [astronaut, camera, cell, coins, moon, page, text, clock, coffee, chelsea]
    images = []
    for src_fn in sources:
        arr = src_fn()
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[2] == 4: arr = arr[:,:,:3]
        H, W = arr.shape[:2]
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
        count = 0
        for cy in range(0, H-size+1, size):
            for cx in range(0, W-size+1, size):
                if count >= 10: break
                images.append(gray[cy:cy+size, cx:cx+size].astype(np.uint8))
                count += 1
                if len(images) >= n: return images
            if count >= 10: break
    return images[:n]

def compute_psnr(orig, recon):
    err = orig.astype(np.float64) - recon.astype(np.float64)
    mse = float(np.mean(err**2))
    return 100.0 if mse == 0 else float(10*np.log10(255.0**2/mse))

def train_shared(images, hidden=64, omega=15.0, epochs=50, lr=1e-3):
    import torch, torch.nn as nn
    torch.manual_seed(0)
    n_images = len(images); n_pix = images[0].shape[0]
    coords = np.stack(np.meshgrid(np.linspace(-1,1,n_pix),np.linspace(-1,1,n_pix)),axis=-1).reshape(-1,2)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(); d = 2
            for k in range(2):
                lin = nn.Linear(d, hidden)
                bound = 1.0/d if k == 0 else np.sqrt(6.0/hidden)/omega
                lin.weight.data.uniform_(-bound,bound); lin.bias.data.uniform_(-bound,bound)
                self.layers.append(lin); d = hidden
            # One-hot conditioning
            self.head = nn.Linear(hidden + n_images, 1)
            bound = np.sqrt(6.0/(hidden+n_images))/omega
            self.head.weight.data.uniform_(-bound,bound); self.head.bias.data.uniform_(-bound,bound)
            self.omega = omega; self.n_images = n_images
        def forward(self, x, idx):
            h = x
            for layer in self.layers: h = torch.sin(self.omega*layer(h))
            onehot = torch.zeros(x.shape[0], self.n_images, device=x.device)
            onehot[:, idx] = 1.0
            return self.head(torch.cat([h, onehot], dim=-1))
    model = M(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    targets = [torch.tensor((img.astype(np.float32)/255.0).flatten(), dtype=torch.float32) for img in images]
    for ep in range(epochs):
        opt.zero_grad(); loss = 0
        for i,yt in enumerate(targets):
            pred = model(xt,i).squeeze(-1); loss = loss + ((pred-yt)**2).mean()
        loss = loss/n_images; loss.backward(); opt.step()
    model.eval(); psnrs = []
    with torch.no_grad():
        for i,img in enumerate(images):
            recon = model(xt,i).squeeze(-1).numpy()
            recon_img = (recon*255).clip(0,255).astype(np.uint8).reshape(n_pix,n_pix)
            psnrs.append(compute_psnr(img, recon_img))
    all_params = np.concatenate([p.detach().numpy().flatten() for p in model.parameters()])
    max_val = np.abs(all_params).max(); scale = max_val/127.5
    quantized = np.round(all_params/scale).astype(np.int8)
    compressed = zlib.compress(quantized.tobytes(), 9)
    return {'avg_psnr_db': float(np.mean(psnrs)), 'recipe_bytes': 18+len(compressed),
            'n_params': len(all_params)}

def run():
    print("="*72)
    print("BHUH EXPERIMENT 9: Backbone Capacity vs Quality")
    print("="*72)
    print()
    print("HYPOTHESIS: Larger backbone improves quality at large N")
    print()
    import torch

    # Test at N=100 with different backbone sizes
    N = 100
    print(f"--- Loading {N} crops ---")
    sys.stdout.flush()
    images = load_crops(N, 64)
    print(f"  Loaded {len(images)} images")
    print()

    backbones = [
        ('h=32',  32),
        ('h=64',  64),
        ('h=128', 128),
    ]

    results = []
    for name, hidden in backbones:
        print(f"\n--- Testing backbone {name} at N={N} ---")
        sys.stdout.flush()
        t0 = time.time()
        r = train_shared(images, hidden=hidden, epochs=50)
        t = time.time()-t0
        r['backbone'] = name; r['hidden'] = hidden; r['time_s'] = t
        results.append(r)
        print(f"  {name}: {r['recipe_bytes']}B, {r['avg_psnr_db']:.2f}dB, "
              f"{r['n_params']} params, {t:.1f}s")
        sys.stdout.flush()

    # Also test at N=50 for comparison
    print(f"\n--- N=50 comparison ---")
    images50 = images[:50]
    for name, hidden in backbones:
        print(f"  Testing {name} at N=50...", end=' ')
        sys.stdout.flush()
        t0 = time.time()
        r = train_shared(images50, hidden=hidden, epochs=50)
        t = time.time()-t0
        print(f"{r['recipe_bytes']}B, {r['avg_psnr_db']:.2f}dB, {t:.1f}s")
        results.append({'backbone': name, 'hidden': hidden, 'n': 50,
                        'recipe_bytes': r['recipe_bytes'], 'avg_psnr_db': r['avg_psnr_db'],
                        'n_params': r['n_params'], 'time_s': t})

    # Summary
    print()
    print("="*72)
    print("CAPACITY RESULTS")
    print("="*72)
    print(f"{'Backbone':<10} {'N':>5} {'Bytes':>8} {'PSNR':>8} {'Params':>8} {'Time':>6}")
    for r in results:
        n = r.get('n', 100)
        print(f"{r['backbone']:<10} {n:>5} {r['recipe_bytes']:>8} "
              f"{r['avg_psnr_db']:>7.2f}dB {r['n_params']:>8} {r['time_s']:>5.1f}s")

    # Analysis
    print()
    print("="*72)
    print("ANALYSIS")
    print("="*72)
    print()

    # Compare h=32 vs h=128 at N=100
    h32 = next(r for r in results if r['backbone']=='h=32' and r.get('n',100)==100)
    h128 = next(r for r in results if r['backbone']=='h=128' and r.get('n',100)==100)

    psnr_gain = h128['avg_psnr_db'] - h32['avg_psnr_db']
    size_ratio = h128['recipe_bytes'] / h32['recipe_bytes']

    print(f"  h=32 → h=128 at N=100:")
    print(f"    PSNR gain: {psnr_gain:+.2f} dB")
    print(f"    Size increase: {size_ratio:.2f}×")
    print(f"    Quality/cost: {psnr_gain/size_ratio:.2f} dB per byte×")
    print()

    if psnr_gain > 3:
        print(f"  ✅ LARGER BACKBOARD HELPS — {psnr_gain:.1f} dB gain")
        print(f"     Capacity WAS the bottleneck")
        verdict = (f"VALIDATED — Larger backbone improves quality by {psnr_gain:.1f} dB. "
                   "Capacity was the bottleneck. Use h=128 for N≥50.")
    elif psnr_gain > 1:
        print(f"  ⚠️ MODEST IMPROVEMENT — {psnr_gain:.1f} dB gain (marginal)")
        verdict = f"PARTIAL — {psnr_gain:.1f} dB gain, marginal improvement"
    else:
        print(f"  ❌ LARGER BACKBONE DOESN'T HELP — architecture is limited")
        verdict = "INVALID — capacity not the bottleneck"

    print(f"\n  VERDICT: {verdict}")

    return {'experiment': 9, 'results': results, 'verdict': verdict,
            'psnr_gain_h32_to_h128': float(psnr_gain)}

if __name__ == '__main__':
    result = run()
    print("\n--- JSON ---")
    print(json.dumps(result, indent=2, default=str))
