#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Experiment 15: Quality-Aware Clustering

INSIGHT from Exp 14: The ~1 dB gap is STRUCTURAL, not training-related.
Hypothesis: KMeans (pixel similarity) groups wrong images together.

PROBLEM with KMeans:
  - Groups by pixel similarity (brightness, texture)
  - But SIREN cares about FREQUENCY CONTENT, not pixel values
  - Two images with similar pixels but different frequency = bad sharing
  - Two images with different pixels but similar frequency = good sharing

SOLUTION: Quality-aware clustering
  1. Fit individual SIREN to each image (fast, 30 epochs)
  2. Measure HOW WELL each SIREN fits (PSNR = difficulty score)
  3. Group by DIFFICULTY, not pixel similarity
  4. Easy images share backbone (they're smooth, SIREN handles well)
  5. Hard images get separate SIRENs (they need full capacity)

This is a NEW idea — not in COIN/COIN++ literature.
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

def compute_difficulty(image, hidden=32, omega=15.0, epochs=30, lr=1e-3):
    """Fit quick SIREN, return PSNR as difficulty score.
    High PSNR = easy (smooth, low frequency)
    Low PSNR = hard (textured, high frequency)
    """
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
    return compute_psnr(image, recon_img)

def cluster_by_difficulty(images, K):
    """Cluster images by SIREN difficulty (PSNR), not pixel similarity."""
    print(f"  Computing difficulty scores for {len(images)} images...")
    sys.stdout.flush()
    difficulties = []
    for i, img in enumerate(images):
        d = compute_difficulty(img, epochs=30)
        difficulties.append(d)
        if (i+1) % 20 == 0:
            print(f"    {i+1}/{len(images)}...", end='\r')
            sys.stdout.flush()
    print()
    
    difficulties = np.array(difficulties)
    print(f"  Difficulty range: {difficulties.min():.2f} - {difficulties.max():.2f} dB")
    print(f"  Mean: {difficulties.mean():.2f} dB, Std: {difficulties.std():.2f} dB")
    
    # Sort by difficulty (easy → hard)
    sorted_indices = np.argsort(difficulties)
    
    # Split into K groups: easiest K/N together, next K/N together, etc.
    # This groups SIMILAR DIFFICULTY images together
    groups = [[] for _ in range(K)]
    for i, idx in enumerate(sorted_indices):
        group_idx = min(i * K // len(sorted_indices), K - 1)
        groups[group_idx].append(int(idx))
    
    # Print group info
    for i, g in enumerate(groups):
        if len(g) > 0:
            group_diffs = [difficulties[j] for j in g]
            print(f"    Group {i}: {len(g)} images, difficulty {np.mean(group_diffs):.1f} dB")
    
    return [g for g in groups if len(g) >= 1], difficulties

def train_shared_group(images, hidden=64, omega=15.0, epochs=80, lr=1e-3):
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
    return {'avg_psnr_db': float(np.mean(psnrs)), 'recipe_bytes': 18+len(compressed), 'psnrs': psnrs}

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

def run_quality_aware(images, K, hidden=64, epochs=80):
    """Run quality-aware hierarchical: group by difficulty."""
    groups, difficulties = cluster_by_difficulty(images, K)
    total_bytes = 0; all_psnrs = []
    for group_indices in groups:
        group_images = [images[i] for i in group_indices]
        if len(group_images) <= 1:
            r = train_coin_single(group_images[0], epochs=50)
            total_bytes += r['recipe_bytes']; all_psnrs.append(r['psnr_db'])
        else:
            r = train_shared_group(group_images, hidden=hidden, epochs=epochs)
            total_bytes += r['recipe_bytes']
            if 'psnrs' in r:
                all_psnrs.extend(r['psnrs'])
            else:
                all_psnrs.append(r['avg_psnr_db'])
    return {'total_bytes': total_bytes, 'avg_psnr': float(np.mean(all_psnrs)),
            'difficulties': difficulties.tolist()}

def run():
    print("="*72)
    print("BHUH EXPERIMENT 15: Quality-Aware Clustering")
    print("="*72)
    print()
    print("HYPOTHESIS: Group by SIREN difficulty, not pixel similarity.")
    print("  Easy images (high PSNR) share backbone well.")
    print("  Hard images (low PSNR) need separate SIRENs.")
    print()
    import torch

    N = 100
    print(f"--- Loading {N} crops ---")
    sys.stdout.flush()
    images = load_crops(N, 64)
    print(f"  Loaded {len(images)} images")
    print()

    coin_bytes = 85601; coin_psnr = 28.10

    # Test quality-aware at K=25 and K=50
    for K in [25, 50]:
        print(f"\n{'='*60}")
        print(f"  Quality-Aware K={K}")
        print(f"{'='*60}")
        sys.stdout.flush()
        t0 = time.time()
        r = run_quality_aware(images, K, hidden=64, epochs=80)
        t = time.time()-t0
        gap = r['avg_psnr'] - coin_psnr
        ratio = coin_bytes / r['total_bytes']
        print(f"  Quality-Aware K={K}: {r['total_bytes']}B, {r['avg_psnr']:.2f}dB, {t:.1f}s")
        print(f"  vs COIN: {ratio:.2f}x smaller, {gap:+.2f} dB")
        
        # Compare to KMeans (Exp 12/13)
        if K == 25:
            kmeans_psnr = 24.95; kmeans_bytes = 25146
        else:
            kmeans_psnr = 27.13; kmeans_bytes = 56450
        
        qa_diff = r['avg_psnr'] - kmeans_psnr
        print(f"  vs KMeans K={K}: {qa_diff:+.2f} dB ({'BETTER' if qa_diff > 0 else 'WORSE'})")
        sys.stdout.flush()

    # Summary
    print()
    print("="*72)
    print("SUMMARY")
    print("="*72)
    print(f"  COIN:          {coin_bytes}B, {coin_psnr}dB")
    print(f"  KMeans K=25:   25146B, 24.95dB (3.4x smaller, -3.15 dB)")
    print(f"  KMeans K=50:   56450B, 27.13dB (1.5x smaller, -0.97 dB)")
    print(f"  Quality K=25:  see above")
    print(f"  Quality K=50:  see above")

if __name__ == '__main__':
    run()
    print("\nDone.")
