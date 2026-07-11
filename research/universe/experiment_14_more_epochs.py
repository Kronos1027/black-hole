#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Experiment 14: Close the 1 dB gap — more epochs at K=50

Exp 13: K=50 with 60 epochs → 27.13 dB (gap: 0.97 dB)
Test: K=50 with 150 epochs → can we reach 28 dB?

Also test: K=50 with larger backbone (h=128) + more epochs
"""
import os, sys, time, json, zlib, struct
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

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

def cluster_images(images, k):
    features = np.array([img.flatten().astype(np.float32)/255.0 for img in images])
    kmeans = KMeans(n_clusters=min(k, len(images)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    groups = [[] for _ in range(min(k, len(images)))]
    for i, label in enumerate(labels):
        groups[label].append(i)
    return groups

def train_shared_group(images, hidden=64, omega=15.0, epochs=60, lr=1e-3):
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
    return {'avg_psnr_db': float(np.mean(psnrs)), 'recipe_bytes': 18+len(compressed)}

def train_coin_single(image, hidden=32, omega=15.0, epochs=50, lr=1e-3):
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

def run_hierarchical(images, K, hidden=64, epochs=60):
    groups = cluster_images(images, K)
    total_bytes = 0; all_psnrs = []
    for group_indices in groups:
        group_images = [images[i] for i in group_indices]
        if len(group_images) <= 1:
            r = train_coin_single(group_images[0], epochs=50)
            total_bytes += r['recipe_bytes']; all_psnrs.append(r['psnr_db'])
        else:
            r = train_shared_group(group_images, hidden=hidden, epochs=epochs)
            total_bytes += r['recipe_bytes']; all_psnrs.extend(r.get('psnrs', [r.get('avg_psnr_db', r.get('psnr_db', 0))]))
    return {'total_bytes': total_bytes, 'avg_psnr': float(np.mean(all_psnrs))}

def run():
    print("="*72)
    print("BHUH EXPERIMENT 14: Close the 1 dB gap")
    print("="*72)
    print()
    print("Exp 13: K=50, 60 epochs → 27.13 dB (gap: 0.97 dB)")
    print("Test: K=50 with more epochs → can we reach 28 dB?")
    print()
    import torch

    N = 100
    images = load_crops(N, 64)
    print(f"Loaded {len(images)} images")
    print()

    coin_bytes = 85601; coin_psnr = 28.10

    # Test different epoch counts at K=50
    configs = [
        ('K=50, 60ep, h=64',  50, 64, 60),
        ('K=50, 100ep, h=64', 50, 64, 100),
        ('K=50, 150ep, h=64', 50, 64, 150),
    ]

    results = []
    for name, K, hidden, epochs in configs:
        print(f"--- {name} ---")
        sys.stdout.flush()
        t0 = time.time()
        r = run_hierarchical(images, K, hidden=hidden, epochs=epochs)
        t = time.time()-t0
        r['name'] = name; r['time_s'] = t
        results.append(r)
        gap = r['avg_psnr'] - coin_psnr
        ratio = coin_bytes / r['total_bytes']
        print(f"  {r['total_bytes']}B, {r['avg_psnr']:.2f}dB, {t:.1f}s")
        print(f"  vs COIN: {ratio:.2f}x smaller, {gap:+.2f} dB")
        sys.stdout.flush()

    # Summary
    print()
    print("="*72)
    print("RESULTS")
    print("="*72)
    print(f"{'Config':<22} {'Bytes':>8} {'PSNR':>8} {'vs COIN':>8} {'dB gap':>8} {'Time':>6}")
    print(f"{'COIN':<22} {coin_bytes:>8} {coin_psnr:>7.2f}dB {'1.0x':>7} {'0.0':>7} {'-':>5}")
    for r in results:
        gap = r['avg_psnr'] - coin_psnr
        ratio = coin_bytes / r['total_bytes']
        print(f"{r['name']:<22} {r['total_bytes']:>8} {r['avg_psnr']:>7.2f}dB "
              f"{ratio:>6.1f}x {gap:>+7.2f} {r['time_s']:>5.1f}s")

    # Analysis
    print()
    best = max(results, key=lambda r: r['avg_psnr'])
    best_gap = best['avg_psnr'] - coin_psnr
    best_ratio = coin_bytes / best['total_bytes']
    
    if best_gap > -0.5:
        print(f"  ✅ MATCHES COIN — gap only {abs(best_gap):.2f} dB!")
        verdict = f"VALIDATED — gap closed to {best_gap:+.2f} dB at {best_ratio:.1f}x smaller"
    elif best_gap > -1.0:
        print(f"  ⚠️ NEARLY MATCHES — gap {abs(best_gap):.2f} dB")
        verdict = f"PARTIAL — gap {best_gap:+.2f} dB, {best_ratio:.1f}x smaller"
    else:
        print(f"  ❌ Gap persists: {abs(best_gap):.2f} dB")
        verdict = f"PARTIAL — gap {best_gap:+.2f} dB"

    print(f"\n  VERDICT: {verdict}")
    
    # Check if more epochs helped
    if len(results) >= 2:
        improvement = results[-1]['avg_psnr'] - results[0]['avg_psnr']
        print(f"\n  Epoch improvement (60→{configs[-1][3]}): {improvement:+.2f} dB")
        if improvement > 0.3:
            print(f"  ✅ More epochs help — quality improves with training")
        else:
            print(f"  ⚠️ More epochs don't help much — convergence reached")

    return {'experiment': 14, 'results': results, 'verdict': verdict}

if __name__ == '__main__':
    result = run()
    print("\n--- JSON ---")
    print(json.dumps(result, indent=2, default=str))
