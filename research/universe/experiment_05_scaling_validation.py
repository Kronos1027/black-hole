#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""BHUH Experiment 5: Scaling Law at N=20 and N=50"""
import os, sys, time, json, zlib, struct
import numpy as np
from PIL import Image

def load_real_crops(n_total, size=64):
    from skimage.data import (astronaut, camera, cell, coins, moon,
                               page, text, clock, coffee, chelsea)
    sources = [astronaut, camera, cell, coins, moon, page, text, clock, coffee, chelsea]
    crops_per_image = (n_total + len(sources) - 1) // len(sources)
    images = []
    for src_fn in sources:
        arr = src_fn()
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[2] == 4: arr = arr[:,:,:3]
        H, W = arr.shape[:2]
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
        n_crops = 0
        for cy in range(0, H - size + 1, size):
            for cx in range(0, W - size + 1, size):
                if n_crops >= crops_per_image: break
                images.append(gray[cy:cy+size, cx:cx+size].astype(np.uint8))
                n_crops += 1
                if len(images) >= n_total: return images[:n_total]
            if n_crops >= crops_per_image: break
    return images[:n_total]

def compute_psnr(orig, recon):
    err = orig.astype(np.float64) - recon.astype(np.float64)
    mse = float(np.mean(err ** 2))
    return 100.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))

def train_coin_single(image, hidden=32, omega=15.0, epochs=100, lr=1e-3):
    import torch, torch.nn as nn
    torch.manual_seed(0)
    n_pix = image.shape[0]
    coords = np.stack(np.meshgrid(np.linspace(-1,1,n_pix),np.linspace(-1,1,n_pix)),axis=-1).reshape(-1,2)
    target = (image.astype(np.float32)/255.0).flatten()
    class Siren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            d = 2
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
            for layer in self.layers: h = torch.sin(self.omega * layer(h))
            return self.head(h)
    model = Siren()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    yt = torch.tensor(target, dtype=torch.float32)
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

def train_shared_siren(images, hidden=64, omega=15.0, epochs=100, lr=1e-3):
    import torch, torch.nn as nn
    torch.manual_seed(0)
    n_images = len(images)
    n_pix = images[0].shape[0]
    coords = np.stack(np.meshgrid(np.linspace(-1,1,n_pix),np.linspace(-1,1,n_pix)),axis=-1).reshape(-1,2)
    class SharedSiren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            d = 2
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
            for layer in self.layers: h = torch.sin(self.omega * layer(h))
            return self.heads[idx](h)
    model = SharedSiren()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
    return {'psnrs_db': psnrs, 'avg_psnr_db': float(np.mean(psnrs)), 'recipe_bytes': 18+len(compressed)}

def run():
    print("="*72)
    print("BHUH EXPERIMENT 5: Scaling at N=20, 50")
    print("="*72)
    import torch
    n_values = [20, 50]
    results = []
    for n in n_values:
        print(f"\n--- N={n} ---"); sys.stdout.flush()
        images = load_real_crops(n, size=64)
        print(f"  Loaded {len(images)} crops"); sys.stdout.flush()
        print(f"  COIN: training {n} SIRENs..."); sys.stdout.flush()
        t0 = time.time()
        coin_results = [train_coin_single(img, epochs=100) for img in images]
        t_coin = time.time()-t0
        coin_total = sum(r['recipe_bytes'] for r in coin_results)
        coin_psnr = np.mean([r['psnr_db'] for r in coin_results])
        print(f"  COIN: {coin_total}B, {coin_psnr:.2f}dB, {t_coin:.1f}s"); sys.stdout.flush()
        print(f"  BHUH: training shared SIREN..."); sys.stdout.flush()
        t0 = time.time()
        bhuh = train_shared_siren(images, epochs=100)
        t_bhuh = time.time()-t0
        print(f"  BHUH: {bhuh['recipe_bytes']}B, {bhuh['avg_psnr_db']:.2f}dB, {t_bhuh:.1f}s")
        ratio = coin_total/bhuh['recipe_bytes']
        psnr_diff = bhuh['avg_psnr_db']-coin_psnr
        results.append({'n':n,'coin_bytes':coin_total,'coin_psnr':float(coin_psnr),
                        'bhuh_bytes':bhuh['recipe_bytes'],'bhuh_psnr':bhuh['avg_psnr_db'],
                        'ratio':float(ratio),'psnr_diff':float(psnr_diff),
                        'coin_time_s':float(t_coin),'bhuh_time_s':float(t_bhuh)})
        print(f"  → Ratio: {ratio:.2f}x, PSNR diff: {psnr_diff:+.2f}dB")
    print("\n"+"="*72)
    print("RESULTS")
    print("="*72)
    print(f"{'N':>5} {'COIN B':>8} {'BHUH B':>8} {'Ratio':>7} {'Predicted':>10} {'ΔPSNR':>7}")
    for r in results:
        pred = 0.258 + 0.122*r['n']
        print(f"{r['n']:>5} {r['coin_bytes']:>8} {r['bhuh_bytes']:>8} {r['ratio']:>6.2f}x {pred:>9.2f}x {r['psnr_diff']:>+6.2f}")
    # Combined with Exp 4
    exp4 = [(3,0.87),(5,1.44),(8,2.20),(10,2.69)]
    exp5 = [(r['n'],r['ratio']) for r in results]
    all_data = exp4+exp5
    ns = [d[0] for d in all_data]; ratios = [d[1] for d in all_data]
    a,b = np.polyfit(ns, ratios, 1)
    r_sq = np.corrcoef(ns,ratios)[0,1]**2
    print(f"\nCombined fit: ratio = {a:.3f} + {b:.3f}×N, R²={r_sq:.4f}")
    n50 = next(r['ratio'] for r in results if r['n']==50)
    if n50 > 4:
        print(f"\n✅ VALIDATED — N=50 achieves {n50:.2f}× advantage")
    else:
        print(f"\n⚠️ N=50 only {n50:.2f}× (predicted 6.35×)")
    return {'experiment':5,'results':results,'fit':{'a':float(a),'b':float(b),'r2':float(r_sq)}}

if __name__ == '__main__':
    result = run()
    print("\n--- JSON ---")
    print(json.dumps(result, indent=2, default=str))
