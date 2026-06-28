#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Experiment 13: K=50 — Can we reach COIN quality?

From Exp 12 trend: +0.5-0.8 dB per +5K
K=25: 24.95 dB → K=50 projected: ~27.5 dB (near COIN's 28.10)

Also test HYBRID approach:
  - Groups with 1-2 images: use COIN (separate SIREN, no backbone overhead)
  - Groups with 3+ images: use shared backbone
  This adapts to actual image similarity structure.
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
    return {'avg_psnr_db': float(np.mean(psnrs)), 'recipe_bytes': 18+len(compressed),
            'psnrs': psnrs}

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

def run_hybrid(images, K, hidden=64, epochs=60):
    """HYBRID: shared backbone for groups≥3, COIN for groups≤2."""
    groups = cluster_images(images, K)
    total_bytes = 0
    all_psnrs = []
    n_shared = 0; n_coin = 0
    
    for group_indices in groups:
        group_images = [images[i] for i in group_indices]
        
        if len(group_images) <= 2:
            # Use COIN for small groups
            for img in group_images:
                r = train_coin_single(img, epochs=50)
                total_bytes += r['recipe_bytes']
                all_psnrs.append(r['psnr_db'])
                n_coin += 1
        else:
            # Use shared backbone for larger groups
            r = train_shared_group(group_images, hidden=hidden, epochs=epochs)
            total_bytes += r['recipe_bytes']
            all_psnrs.extend(r['psnrs'])
            n_shared += len(group_images)
    
    return {
        'total_bytes': total_bytes,
        'avg_psnr': float(np.mean(all_psnrs)) if all_psnrs else 0,
        'n_shared': n_shared, 'n_coin': n_coin,
        'n_groups': len(groups),
    }

def run():
    print("="*72)
    print("BHUH EXPERIMENT 13: K=50 + HYBRID Approach")
    print("="*72)
    print()
    import torch

    N = 100
    print(f"--- Loading {N} crops ---")
    sys.stdout.flush()
    images = load_crops(N, 64)
    print(f"  Loaded {len(images)} images")
    print()

    # COIN baseline
    coin_bytes = 85601; coin_psnr = 28.10
    print(f"COIN baseline: {coin_bytes}B, {coin_psnr}dB")
    print()

    # Test 1: K=50 pure hierarchical
    print("--- Test 1: Hierarchical K=50 ---")
    sys.stdout.flush()
    groups50 = cluster_images(images, 50)
    print(f"  Clusters: {len(groups50)}, sizes: {sorted([len(g) for g in groups50], reverse=True)[:10]}...")
    sys.stdout.flush()
    
    t0 = time.time()
    total_bytes_50 = 0; all_psnrs_50 = []
    for g_idx, group_indices in enumerate(groups50):
        if len(group_indices) < 1: continue
        group_images = [images[i] for i in group_indices]
        if len(group_images) == 1:
            r = train_coin_single(group_images[0], epochs=50)
            total_bytes_50 += r['recipe_bytes']
            all_psnrs_50.append(r['psnr_db'])
        else:
            r = train_shared_group(group_images, epochs=60)
            total_bytes_50 += r['recipe_bytes']
            all_psnrs_50.extend(r['psnrs'])
    t_50 = time.time()-t0
    psnr_50 = np.mean(all_psnrs_50)
    ratio_50 = coin_bytes / total_bytes_50
    print(f"  K=50: {total_bytes_50}B, {psnr_50:.2f}dB, {t_50:.1f}s")
    print(f"  vs COIN: {ratio_50:.2f}x smaller, {psnr_50-coin_psnr:+.2f} dB")
    print()

    # Test 2: HYBRID (K=25, COIN for small groups)
    print("--- Test 2: HYBRID K=25 (shared for ≥3, COIN for ≤2) ---")
    sys.stdout.flush()
    t0 = time.time()
    hybrid = run_hybrid(images, K=25, hidden=64, epochs=60)
    t_hybrid = time.time()-t0
    ratio_hybrid = coin_bytes / hybrid['total_bytes']
    print(f"  Hybrid: {hybrid['total_bytes']}B, {hybrid['avg_psnr']:.2f}dB, {t_hybrid:.1f}s")
    print(f"  Shared: {hybrid['n_shared']} images, COIN: {hybrid['n_coin']} images")
    print(f"  vs COIN: {ratio_hybrid:.2f}x smaller, {hybrid['avg_psnr']-coin_psnr:+.2f} dB")
    print()

    # Test 3: HYBRID with K=50
    print("--- Test 3: HYBRID K=50 ---")
    sys.stdout.flush()
    t0 = time.time()
    hybrid50 = run_hybrid(images, K=50, hidden=64, epochs=60)
    t_hybrid50 = time.time()-t0
    ratio_hybrid50 = coin_bytes / hybrid50['total_bytes']
    print(f"  Hybrid50: {hybrid50['total_bytes']}B, {hybrid50['avg_psnr']:.2f}dB, {t_hybrid50:.1f}s")
    print(f"  Shared: {hybrid50['n_shared']} images, COIN: {hybrid50['n_coin']} images")
    print(f"  vs COIN: {ratio_hybrid50:.2f}x smaller, {hybrid50['avg_psnr']-coin_psnr:+.2f} dB")
    print()

    # Summary
    print("="*72)
    print("COMPLETE RESULTS")
    print("="*72)
    print(f"{'Method':<25} {'Bytes':>8} {'PSNR':>8} {'vs COIN':>8} {'dB gap':>8}")
    print(f"{'COIN (separate)':<25} {coin_bytes:>8} {coin_psnr:>7.2f}dB {'1.0x':>7} {'0.0':>7}")
    print(f"{'Hier K=25 (Exp12)':<25} {25146:>8} {24.95:>7.2f}dB {'3.4x':>7} {'-3.15':>7}")
    print(f"{'Hier K=50':<25} {total_bytes_50:>8} {psnr_50:>7.2f}dB {ratio_50:>6.1f}x {psnr_50-coin_psnr:>+7.2f}")
    print(f"{'Hybrid K=25':<25} {hybrid['total_bytes']:>8} {hybrid['avg_psnr']:>7.2f}dB {ratio_hybrid:>6.1f}x {hybrid['avg_psnr']-coin_psnr:>+7.2f}")
    print(f"{'Hybrid K=50':<25} {hybrid50['total_bytes']:>8} {hybrid50['avg_psnr']:>7.2f}dB {ratio_hybrid50:>6.1f}x {hybrid50['avg_psnr']-coin_psnr:>+7.2f}")

    # Analysis
    print()
    print("="*72)
    print("ANALYSIS")
    print("="*72)
    print()

    best_psnr = max(psnr_50, hybrid['avg_psnr'], hybrid50['avg_psnr'])
    best_name = ''
    best_bytes = 0
    if psnr_50 == best_psnr:
        best_name = f'Hier K=50'; best_bytes = total_bytes_50
    elif hybrid['avg_psnr'] == best_psnr:
        best_name = f'Hybrid K=25'; best_bytes = hybrid['total_bytes']
    else:
        best_name = f'Hybrid K=50'; best_bytes = hybrid50['total_bytes']
    
    best_ratio = coin_bytes / best_bytes
    best_gap = best_psnr - coin_psnr
    
    print(f"  Best: {best_name}")
    print(f"  PSNR: {best_psnr:.2f} dB (COIN: {coin_psnr} dB, gap: {best_gap:+.2f})")
    print(f"  Size: {best_bytes}B ({best_ratio:.2f}x smaller than COIN)")
    print()

    if best_gap > -2:
        print(f"  ✅ NEARLY MATCHES COIN — gap only {abs(best_gap):.1f} dB")
        verdict = (f"VALIDATED — {best_name} nearly matches COIN: "
                   f"{best_psnr:.1f} dB vs {coin_psnr} dB ({best_gap:+.1f}), "
                   f"{best_ratio:.1f}x smaller. Publishable.")
    elif best_gap > -3:
        print(f"  ⚠️ CLOSING THE GAP — {abs(best_gap):.1f} dB difference")
        verdict = f"PARTIAL — {best_ratio:.1f}x smaller, {best_gap:+.1f} dB gap (closing)"
    else:
        print(f"  ⚠️ Still {abs(best_gap):.1f} dB gap")
        verdict = f"PARTIAL — {best_ratio:.1f}x smaller, {best_gap:+.1f} dB gap"

    print(f"\n  VERDICT: {verdict}")

    return {'experiment': 13, 'verdict': verdict,
            'k50': {'bytes': total_bytes_50, 'psnr': float(psnr_50)},
            'hybrid25': {'bytes': hybrid['total_bytes'], 'psnr': hybrid['avg_psnr']},
            'hybrid50': {'bytes': hybrid50['total_bytes'], 'psnr': hybrid50['avg_psnr']}}

if __name__ == '__main__':
    result = run()
    print("\n--- JSON ---")
    print(json.dumps(result, indent=2, default=str))
