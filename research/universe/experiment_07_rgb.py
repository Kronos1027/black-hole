#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Experiment 7: RGB Color Images
=====================================
Question: Does BHUH advantage hold/grow for color (3-channel) images?

Theory:
  Grayscale: 1 channel, SIREN output dim=1
  RGB: 3 channels, SIREN output dim=3 (or 3 separate SIRENs)

  COIN RGB: 3× params per image (or 3× output head)
  BHUH RGB: shared backbone + 3× head per image

  If backbone is shared across ALL images AND all channels,
  BHUH advantage should GROW with RGB (more sharing = more amortization).

Test: N=5, 64×64, RGB vs grayscale
"""
import os, sys, time, json, zlib, struct
import numpy as np
from PIL import Image

def load_real_rgb(n, size=64):
    from skimage.data import (astronaut, camera, cell, coins, moon,
                               page, text, clock, coffee, chelsea)
    sources = [astronaut, camera, cell, coins, moon, page, text, clock, coffee, chelsea]
    images = []
    for src_fn in sources[:n]:
        arr = src_fn()
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[2] == 4: arr = arr[:,:,:3]
        pil = Image.fromarray(arr.astype(np.uint8))
        pil = pil.resize((size, size), Image.LANCZOS)
        images.append(np.array(pil))
    return images

def compute_psnr_rgb(orig, recon):
    """PSNR for RGB image (average over 3 channels)."""
    err = orig.astype(np.float64) - recon.astype(np.float64)
    mse = float(np.mean(err**2))
    return 100.0 if mse == 0 else float(10*np.log10(255.0**2/mse))

def train_coin_rgb(image, hidden=32, omega=15.0, epochs=150, lr=1e-3):
    """COIN: separate SIREN per image, 3-channel output."""
    import torch, torch.nn as nn
    torch.manual_seed(0)
    H, W, C = image.shape
    coords = np.stack(np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H)),axis=-1).reshape(-1,2)
    target = (image.astype(np.float32)/255.0).reshape(-1, 3)  # (N, 3)
    class Siren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(); d = 2
            for k in range(2):
                lin = nn.Linear(d, hidden)
                bound = 1.0/d if k == 0 else np.sqrt(6.0/hidden)/omega
                lin.weight.data.uniform_(-bound,bound); lin.bias.data.uniform_(-bound,bound)
                self.layers.append(lin); d = hidden
            self.head = nn.Linear(hidden, 3)  # 3-channel output
            bound = np.sqrt(6.0/hidden)/omega
            self.head.weight.data.uniform_(-bound,bound); self.head.bias.data.uniform_(-bound,bound)
            self.omega = omega
        def forward(self, x):
            h = x
            for layer in self.layers: h = torch.sin(self.omega*layer(h))
            return self.head(h)
    model = Siren(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    yt = torch.tensor(target, dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad(); pred = model(xt)
        loss = ((pred-yt)**2).mean(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): recon = model(xt).numpy()
    recon_img = (recon*255).clip(0,255).astype(np.uint8).reshape(H,W,3)
    psnr = compute_psnr_rgb(image, recon_img)
    params = [p.detach().numpy().flatten() for p in model.parameters()]
    params_flat = np.concatenate(params)
    max_val = np.abs(params_flat).max(); scale = max_val/127.5
    quantized = np.round(params_flat/scale).astype(np.int8)
    compressed = zlib.compress(quantized.tobytes(), 9)
    return {'psnr_db': psnr, 'recipe_bytes': 10+len(compressed)}

def train_shared_rgb(images, hidden=64, omega=15.0, epochs=150, lr=1e-3):
    """BHUH: shared SIREN backbone, per-image 3-channel heads."""
    import torch, torch.nn as nn
    torch.manual_seed(0)
    n_images = len(images)
    H, W, C = images[0].shape
    coords = np.stack(np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H)),axis=-1).reshape(-1,2)
    class SharedSiren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(); d = 2
            for k in range(2):
                lin = nn.Linear(d, hidden)
                bound = 1.0/d if k == 0 else np.sqrt(6.0/hidden)/omega
                lin.weight.data.uniform_(-bound,bound); lin.bias.data.uniform_(-bound,bound)
                self.layers.append(lin); d = hidden
            self.heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in range(n_images)])
            for head in self.heads:
                bound = np.sqrt(6.0/hidden)/omega
                head.weight.data.uniform_(-bound,bound); head.bias.data.uniform_(-bound,bound)
            self.omega = omega
        def forward(self, x, idx):
            h = x
            for layer in self.layers: h = torch.sin(self.omega*layer(h))
            return self.heads[idx](h)
    model = SharedSiren(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    targets = [torch.tensor((img.astype(np.float32)/255.0).reshape(-1,3), dtype=torch.float32) for img in images]
    for ep in range(epochs):
        opt.zero_grad(); loss = 0
        for i,yt in enumerate(targets):
            pred = model(xt,i); loss = loss + ((pred-yt)**2).mean()
        loss = loss/n_images; loss.backward(); opt.step()
    model.eval(); psnrs = []
    with torch.no_grad():
        for i,img in enumerate(images):
            recon = model(xt,i).numpy()
            recon_img = (recon*255).clip(0,255).astype(np.uint8).reshape(H,W,3)
            psnrs.append(compute_psnr_rgb(img, recon_img))
    shared_flat = np.concatenate([p.detach().numpy().flatten() for p in model.layers.parameters()])
    heads_flat = [np.concatenate([p.detach().numpy().flatten() for p in h.parameters()]) for h in model.heads]
    all_params = np.concatenate([shared_flat]+heads_flat)
    max_val = np.abs(all_params).max(); scale = max_val/127.5
    quantized = np.round(all_params/scale).astype(np.int8)
    compressed = zlib.compress(quantized.tobytes(), 9)
    return {'avg_psnr_db': float(np.mean(psnrs)), 'recipe_bytes': 18+len(compressed)}

def run():
    print("="*72)
    print("BHUH EXPERIMENT 7: RGB Color Images")
    print("="*72)
    print()
    print("Question: Does BHUH advantage hold for RGB (3-channel) images?")
    print("Theory: RGB has 3× data per image, more sharing opportunity")
    print()
    import torch

    N = 5; size = 64
    print(f"--- Loading {N} REAL RGB photographs ({size}×{size}) ---")
    images = load_real_rgb(N, size)
    print(f"  Loaded {len(images)} images, shape {images[0].shape}")
    print()

    # COIN RGB
    print(f"--- COIN RGB: training {N} separate SIRENs (3-channel output) ---")
    sys.stdout.flush()
    t0 = time.time()
    coin_results = [train_coin_rgb(img, epochs=150) for img in images]
    t_coin = time.time()-t0
    coin_total = sum(r['recipe_bytes'] for r in coin_results)
    coin_psnr = np.mean([r['psnr_db'] for r in coin_results])
    print(f"  COIN RGB: {coin_total}B, {coin_psnr:.2f}dB, {t_coin:.1f}s")
    print()

    # BHUH RGB
    print(f"--- BHUH RGB: shared SIREN + {N} 3-channel heads ---")
    sys.stdout.flush()
    t0 = time.time()
    bhuh = train_shared_rgb(images, epochs=150)
    t_bhuh = time.time()-t0
    print(f"  BHUH RGB: {bhuh['recipe_bytes']}B, {bhuh['avg_psnr_db']:.2f}dB, {t_bhuh:.1f}s")
    print()

    ratio = coin_total / bhuh['recipe_bytes']
    psnr_diff = bhuh['avg_psnr_db'] - coin_psnr

    # Compare to grayscale (from Experiment 4, N=5)
    print("="*72)
    print("COMPARISON: Grayscale vs RGB (N=5, 64×64)")
    print("="*72)
    print(f"{'Mode':<12} {'COIN B':>8} {'BHUH B':>8} {'Ratio':>7} {'COIN dB':>8} {'BHUH dB':>8} {'ΔPSNR':>7}")
    # Grayscale from Exp 4
    print(f"{'Grayscale':<12} {'4382':>8} {'3045':>8} {'1.44':>6}× {'25.15':>7} {'23.86':>7} {'-1.30':>6}")
    print(f"{'RGB':<12} {coin_total:>8} {bhuh['recipe_bytes']:>8} {ratio:>6.2f}× "
          f"{coin_psnr:>7.2f} {bhuh['avg_psnr_db']:>7.2f} {psnr_diff:>+6.2f}")
    print()

    # Analysis
    print("="*72)
    print("ANALYSIS")
    print("="*72)
    print()
    print(f"  RGB ratio: {ratio:.2f}×")
    print(f"  Grayscale ratio (Exp 4): 1.44×")
    print(f"  RGB vs grayscale ratio: {ratio/1.44:.2f}×")
    print()

    if ratio > 1.44 * 1.1:
        print(f"  ✅ RGB ADVANTAGE GROWS — BHUH benefits from channel sharing")
        print(f"     RGB ratio {ratio:.2f}× > grayscale 1.44× ({ratio/1.44:.2f}× better)")
        verdict = (f"VALIDATED — RGB scaling confirmed. BHUH advantage grows from "
                   f"1.44× (grayscale) to {ratio:.2f}× (RGB) — {ratio/1.44:.2f}× improvement. "
                   "Shared backbone amortizes across channels too.")
    elif ratio > 1.44 * 0.9:
        print(f"  ⚠️ RGB advantage similar to grayscale (no extra benefit)")
        verdict = f"PARTIAL — RGB ratio {ratio:.2f}× similar to grayscale 1.44×"
    else:
        print(f"  ❌ RGB advantage DROPS — 3-channel harder to fit")
        verdict = f"NEGATIVE — RGB ratio {ratio:.2f}× < grayscale 1.44×"

    print(f"\n  VERDICT: {verdict}")
    print()

    # Per-pixel cost comparison
    n_pixels = N * size * size
    print("Per-pixel cost:")
    print(f"  COIN RGB: {coin_total/n_pixels:.4f} bpp")
    print(f"  BHUH RGB: {bhuh['recipe_bytes']/n_pixels:.4f} bpp")
    print(f"  COIN gray (Exp 4): {4382/n_pixels:.4f} bpp")
    print(f"  BHUH gray (Exp 4): {3045/n_pixels:.4f} bpp")

    return {'experiment': 7, 'name': 'RGB Color Images',
            'rgb_coin_bytes': int(coin_total), 'rgb_bhuh_bytes': int(bhuh['recipe_bytes']),
            'rgb_ratio': float(ratio), 'rgb_psnr_diff': float(psnr_diff),
            'grayscale_ratio': 1.44, 'verdict': verdict}

if __name__ == '__main__':
    result = run()
    print("\n--- JSON ---")
    print(json.dumps(result, indent=2, default=str))
