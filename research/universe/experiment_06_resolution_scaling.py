#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Experiment 6: Resolution Scaling — Does advantage grow with image size?

Theory: BHUH backbone is FIXED size. Larger images amortize backbone better.
  At 64x64:  backbone/total = 4352/4677 = 93% (backbone dominates)
  At 128x128: backbone/total = 4352/4677 = 93% (same params, 4x more pixels)
  At 256x256: backbone/total = 4352/4677 = 93% (same params, 16x more pixels)

So BHUH per-pixel cost DROPS with resolution. COIN per-pixel cost is constant.
BHUH advantage should GROW with image resolution.

Test: N=5 at 64x64, 128x128, 256x256
"""
import os, sys, time, json, zlib, struct
import numpy as np
from PIL import Image

def load_real_images(n, size):
    from skimage.data import (astronaut, camera, cell, coins, moon,
                               page, text, clock, coffee, chelsea)
    sources = [astronaut, camera, cell, coins, moon, page, text, clock, coffee, chelsea]
    images = []
    for src_fn in sources[:n]:
        arr = src_fn()
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[2] == 4: arr = arr[:,:,:3]
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
        pil = Image.fromarray(gray.astype(np.uint8))
        pil = pil.resize((size, size), Image.LANCZOS)
        images.append(np.array(pil))
    return images

def compute_psnr(orig, recon):
    err = orig.astype(np.float64) - recon.astype(np.float64)
    mse = float(np.mean(err**2))
    return 100.0 if mse == 0 else float(10*np.log10(255.0**2/mse))

def train_coin_single(image, hidden=32, omega=15.0, epochs=150, lr=1e-3):
    import torch, torch.nn as nn
    torch.manual_seed(0)
    n_pix = image.shape[0]
    coords = np.stack(np.meshgrid(np.linspace(-1,1,n_pix),np.linspace(-1,1,n_pix)),axis=-1).reshape(-1,2)
    target = (image.astype(np.float32)/255.0).flatten()
    class Siren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(); d = 2
            for k in range(2):
                lin = nn.Linear(d, hidden)
                bound = 1.0/d if k == 0 else np.sqrt(6.0/hidden)/omega
                lin.weight.data.uniform_(-bound,bound); lin.bias.data.uniform_(-bound,bound)
                self.layers.append(lin); d = hidden
            self.head = nn.Linear(hidden,1)
            bound = np.sqrt(6.0/hidden)/omega
            self.head.weight.data.uniform_(-bound,bound); self.head.bias.data.uniform_(-bound,bound)
            self.omega = omega
        def forward(self, x):
            h = x
            for layer in self.layers: h = torch.sin(self.omega*layer(h))
            return self.head(h)
    model = Siren(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32); yt = torch.tensor(target, dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad(); pred = model(xt).squeeze(-1)
        loss = ((pred-yt)**2).mean(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): recon = model(xt).squeeze(-1).numpy()
    recon_img = (recon*255).clip(0,255).astype(np.uint8).reshape(n_pix,n_pix)
    psnr = compute_psnr(image, recon_img)
    params = [p.detach().numpy().flatten() for p in model.parameters()]
    params_flat = np.concatenate(params)
    max_val = np.abs(params_flat).max(); scale = max_val/127.5
    quantized = np.round(params_flat/scale).astype(np.int8)
    compressed = zlib.compress(quantized.tobytes(), 9)
    return {'psnr_db': psnr, 'recipe_bytes': 10+len(compressed)}

def train_shared_siren(images, hidden=64, omega=15.0, epochs=150, lr=1e-3):
    import torch, torch.nn as nn
    torch.manual_seed(0)
    n_images = len(images); n_pix = images[0].shape[0]
    coords = np.stack(np.meshgrid(np.linspace(-1,1,n_pix),np.linspace(-1,1,n_pix)),axis=-1).reshape(-1,2)
    class SharedSiren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(); d = 2
            for k in range(2):
                lin = nn.Linear(d, hidden)
                bound = 1.0/d if k == 0 else np.sqrt(6.0/hidden)/omega
                lin.weight.data.uniform_(-bound,bound); lin.bias.data.uniform_(-bound,bound)
                self.layers.append(lin); d = hidden
            self.heads = nn.ModuleList([nn.Linear(hidden,1) for _ in range(n_images)])
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
    shared_flat = np.concatenate([p.detach().numpy().flatten() for p in model.layers.parameters()])
    heads_flat = [np.concatenate([p.detach().numpy().flatten() for p in h.parameters()]) for h in model.heads]
    all_params = np.concatenate([shared_flat]+heads_flat)
    max_val = np.abs(all_params).max(); scale = max_val/127.5
    quantized = np.round(all_params/scale).astype(np.int8)
    compressed = zlib.compress(quantized.tobytes(), 9)
    return {'avg_psnr_db': float(np.mean(psnrs)), 'recipe_bytes': 18+len(compressed)}

def run():
    print("="*72)
    print("BHUH EXPERIMENT 6: Resolution Scaling")
    print("="*72)
    print()
    print("Theory: BHUH advantage should GROW with image resolution")
    print("  (backbone is fixed, more pixels = better amortization)")
    print()
    import torch

    sizes = [64, 128, 256]
    N = 5  # Fixed N, vary resolution
    results = []

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"  Resolution: {size}×{size}, N={N}")
        print(f"{'='*60}")
        sys.stdout.flush()

        images = load_real_images(N, size)
        print(f"  Loaded {len(images)} images")
        sys.stdout.flush()

        # COIN
        print(f"  COIN: training {N} SIRENs (150 epochs)...")
        sys.stdout.flush()
        t0 = time.time()
        coin_results = [train_coin_single(img, epochs=150) for img in images]
        t_coin = time.time()-t0
        coin_total = sum(r['recipe_bytes'] for r in coin_results)
        coin_psnr = np.mean([r['psnr_db'] for r in coin_results])
        print(f"  COIN: {coin_total}B, {coin_psnr:.2f}dB, {t_coin:.1f}s")
        sys.stdout.flush()

        # BHUH
        print(f"  BHUH: training shared SIREN (150 epochs)...")
        sys.stdout.flush()
        t0 = time.time()
        bhuh = train_shared_siren(images, epochs=150)
        t_bhuh = time.time()-t0
        print(f"  BHUH: {bhuh['recipe_bytes']}B, {bhuh['avg_psnr_db']:.2f}dB, {t_bhuh:.1f}s")
        sys.stdout.flush()

        ratio = coin_total / bhuh['recipe_bytes']
        orig_total = N * size * size
        results.append({
            'resolution': size,
            'n_images': N,
            'orig_total_bytes': orig_total,
            'coin_bytes': coin_total,
            'coin_psnr': float(coin_psnr),
            'bhuh_bytes': bhuh['recipe_bytes'],
            'bhuh_psnr': bhuh['avg_psnr_db'],
            'ratio': float(ratio),
            'psnr_diff': float(bhuh['avg_psnr_db'] - coin_psnr),
            'coin_time_s': float(t_coin),
            'bhuh_time_s': float(t_bhuh),
        })
        print(f"  → Ratio: {ratio:.2f}×, PSNR diff: {bhuh['avg_psnr_db']-coin_psnr:+.2f}dB")

    # Summary
    print()
    print("="*72)
    print("RESOLUTION SCALING RESULTS")
    print("="*72)
    print(f"{'Size':>6} {'Orig B':>8} {'COIN B':>8} {'BHUH B':>8} {'Ratio':>7} "
          f"{'COIN dB':>8} {'BHUH dB':>8} {'ΔPSNR':>7}")
    for r in results:
        print(f"{r['resolution']:>6} {r['orig_total_bytes']:>8} {r['coin_bytes']:>8} "
              f"{r['bhuh_bytes']:>8} {r['ratio']:>6.2f}× "
              f"{r['coin_psnr']:>7.2f} {r['bhuh_psnr']:>7.2f} {r['psnr_diff']:>+6.2f}")

    # Per-pixel analysis
    print()
    print("Per-pixel cost (bytes per pixel):")
    print(f"{'Size':>6} {'COIN bpp':>10} {'BHUH bpp':>10} {'BHUH/COIN':>10}")
    for r in results:
        n_pixels = r['n_images'] * r['resolution']**2
        coin_bpp = r['coin_bytes'] / n_pixels
        bhuh_bpp = r['bhuh_bytes'] / n_pixels
        print(f"{r['resolution']:>6} {coin_bpp:>10.4f} {bhuh_bpp:>10.4f} "
              f"{bhuh_bpp/coin_bpp:>9.2f}×")

    # Analysis
    print()
    print("="*72)
    print("ANALYSIS")
    print("="*72)
    ratios = [r['ratio'] for r in results]
    sizes = [r['resolution'] for r in results]
    increasing = all(ratios[i] < ratios[i+1] for i in range(len(ratios)-1))

    if increasing:
        print(f"  ✅ RESOLUTION SCALING CONFIRMED")
        print(f"     Ratio grows: {ratios[0]:.2f}× ({sizes[0]}) → {ratios[-1]:.2f}× ({sizes[-1]})")
        # Fit log-linear
        log_ratios = np.log(ratios)
        log_sizes = np.log(sizes)
        slope = np.polyfit(log_sizes, log_ratios, 1)[0]
        print(f"     Power law: ratio ∝ size^{slope:.2f}")
        print(f"     Doubling resolution: ratio × {2**slope:.2f}")
        verdict = (f"VALIDATED — BHUH advantage grows with resolution. "
                   f"Power law exponent {slope:.2f}. "
                   f"Doubling resolution multiplies advantage by {2**slope:.2f}×.")
    else:
        print(f"  ⚠️ No clear resolution scaling")
        verdict = "PARTIAL — no clear scaling"

    print(f"\n  VERDICT: {verdict}")

    return {'experiment': 6, 'name': 'Resolution Scaling',
            'results': results, 'verdict': verdict}

if __name__ == '__main__':
    result = run()
    print("\n--- JSON ---")
    print(json.dumps(result, indent=2, default=str))
