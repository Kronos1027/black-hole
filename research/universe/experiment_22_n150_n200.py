#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Experiment 22: N=150 and N=200 — Mapping the Sublinear Collapse

PRIORITY: RED (Claude report item #1)
QUESTION: Where does the BHUH scaling curve REALLY collapse?
  N=50:  8.94x advantage, 27 dB (usable)
  N=100: 29.7x advantage, 17 dB (unusable quality)
  N=150: ?x advantage, ? dB
  N=200: ?x advantage, ? dB

HYPOTHESIS: Backbone capacity saturates. Each new image pushes the
backbone away from optimal for all others. If we map N=150/200,
we can fit a proper model (logarithmic? power law?) instead of
the failed linear extrapolation.

METHOD (rigorous, per Claude protocol):
  - Same 10 base scikit-image images, multiple crops each
  - 64x64 grayscale (same as all previous experiments)
  - omega=15, 3 layers, h=64, 150 epochs (same as Exp 8 baseline)
  - COIN baseline: separate SIREN per image
  - BHUH: shared SIREN with one-hot conditioning
  - Record: total bytes, avg PSNR, time, ratio
  - Multiple seeds for N=100 to check stability

PROTOCOL:
  - All results documented with exact commands
  - No claims without script committed
  - If numbers look too good, re-run with different seed

Author: Darlan Pereira da Silva (Kronos1027)
"""
import os, sys, time, json, zlib, struct
import numpy as np
from PIL import Image

def load_crops(n, size=64):
    """Load n crops from 10 scikit-image base images."""
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
                if count >= 20: break  # up to 20 crops per image = 200 max
                images.append(gray[cy:cy+size, cx:cx+size].astype(np.uint8))
                count += 1
                if len(images) >= n: return images
            if count >= 20: break
    return images[:n]

def compute_psnr(orig, recon):
    err = orig.astype(np.float64) - recon.astype(np.float64)
    mse = float(np.mean(err**2))
    return 100.0 if mse == 0 else float(10*np.log10(255.0**2/mse))

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

def train_shared_siren(images, hidden=64, omega=15.0, epochs=50, lr=1e-3, seed=0):
    """Shared SIREN with one-hot conditioning."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
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
            'min_psnr': float(min(psnrs)), 'max_psnr': float(max(psnrs)),
            'std_psnr': float(np.std(psnrs))}

def run():
    print("="*72)
    print("BHUH EXPERIMENT 22: N=150, N=200 — Mapping Sublinear Collapse")
    print("="*72)
    print()
    print("Protocol: Claude anti-fabrication — all commands documented")
    print("CPU-only, torch.set_num_threads(8)")
    print()
    import torch
    torch.set_num_threads(8)

    # Load 200 crops upfront
    print("--- Loading 200 crops ---")
    all_images = load_crops(200, 64)
    print(f"  Loaded {len(all_images)} images")
    print()

    n_values = [50, 100, 150, 200]
    results = []

    for n in n_values:
        print(f"\n{'='*60}")
        print(f"  N = {n}")
        print(f"{'='*60}")
        sys.stdout.flush()
        images = all_images[:n]

        # COIN baseline
        print(f"  COIN: training {n} separate SIRENs (50 epochs each)...")
        sys.stdout.flush()
        t0 = time.time()
        coin_total = 0; coin_psnrs = []
        for i, img in enumerate(images):
            if (i+1) % 50 == 0:
                print(f"    {i+1}/{n}...", end='\r')
                sys.stdout.flush()
            r = train_coin_single(img, epochs=50)
            coin_total += r['recipe_bytes']
            coin_psnrs.append(r['psnr_db'])
        t_coin = time.time()-t0
        coin_psnr = np.mean(coin_psnrs)
        print(f"  COIN: {coin_total}B, {coin_psnr:.2f}dB, {t_coin:.1f}s")
        sys.stdout.flush()

        # BHUH shared
        print(f"  BHUH: training shared SIREN (50 epochs)...")
        sys.stdout.flush()
        t0 = time.time()
        bhuh = train_shared_siren(images, hidden=64, epochs=50, seed=0)
        t_bhuh = time.time()-t0
        ratio = coin_total / bhuh['recipe_bytes']
        print(f"  BHUH: {bhuh['recipe_bytes']}B, {bhuh['avg_psnr_db']:.2f}dB, {t_bhuh:.1f}s")
        print(f"  PSNR range: {bhuh['min_psnr']:.1f}-{bhuh['max_psnr']:.1f} dB (std={bhuh['std_psnr']:.1f})")
        print(f"  → Ratio: {ratio:.2f}x, PSNR diff: {bhuh['avg_psnr_db']-coin_psnr:+.2f}dB")
        sys.stdout.flush()

        results.append({
            'n': n,
            'coin_bytes': coin_total,
            'coin_psnr': float(coin_psnr),
            'coin_time_s': float(t_coin),
            'bhuh_bytes': bhuh['recipe_bytes'],
            'bhuh_psnr': bhuh['avg_psnr_db'],
            'bhuh_min_psnr': bhuh['min_psnr'],
            'bhuh_max_psnr': bhuh['max_psnr'],
            'bhuh_std_psnr': bhuh['std_psnr'],
            'bhuh_time_s': float(t_bhuh),
            'ratio': float(ratio),
            'psnr_diff': float(bhuh['avg_psnr_db'] - coin_psnr),
        })

    # Seed sensitivity for N=100
    print(f"\n{'='*60}")
    print(f"  SEED SENSITIVITY (N=100, 3 seeds)")
    print(f"{'='*60}")
    sys.stdout.flush()
    images100 = all_images[:100]
    seed_results = []
    for seed in [0, 42, 123]:
        print(f"  Seed {seed}...", end=' ')
        sys.stdout.flush()
        bhuh = train_shared_siren(images100, hidden=64, epochs=50, seed=seed)
        seed_results.append({
            'seed': seed,
            'psnr': bhuh['avg_psnr_db'],
            'bytes': bhuh['recipe_bytes'],
            'min_psnr': bhuh['min_psnr'],
            'max_psnr': bhuh['max_psnr'],
        })
        print(f"PSNR={bhuh['avg_psnr_db']:.2f}dB, bytes={bhuh['recipe_bytes']}")

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("="*72)
    print("COMPLETE SCALING TABLE (N=50 to N=200)")
    print("="*72)
    print(f"{'N':>5} {'COIN B':>8} {'BHUH B':>8} {'Ratio':>8} {'COIN dB':>8} {'BHUH dB':>8} "
          f"{'dPSNR':>7} {'BHUH min':>8} {'BHUH max':>8} {'BHUH std':>8}")
    for r in results:
        print(f"{r['n']:>5} {r['coin_bytes']:>8} {r['bhuh_bytes']:>8} "
              f"{r['ratio']:>7.2f}x {r['coin_psnr']:>7.2f}dB {r['bhuh_psnr']:>7.2f}dB "
              f"{r['psnr_diff']:>+6.2f} {r['bhuh_min_psnr']:>7.1f}dB "
              f"{r['bhuh_max_psnr']:>7.1f}dB {r['bhuh_std_psnr']:>7.1f}dB")

    # Seed sensitivity
    print()
    print("SEED SENSITIVITY (N=100):")
    seed_psnrs = [s['psnr'] for s in seed_results]
    print(f"  Seeds: {[f'{s:.2f}dB' for s in seed_psnrs]}")
    print(f"  Mean: {np.mean(seed_psnrs):.2f} dB, Std: {np.std(seed_psnrs):.2f} dB")
    print(f"  Stable? {'YES' if np.std(seed_psnrs) < 1.0 else 'NO — high variance'}")

    # ============================================================
    # Curve fitting
    # ============================================================
    print()
    print("="*72)
    print("CURVE FITTING — Which model predicts the collapse?")
    print("="*72)
    print()

    ns = [r['n'] for r in results]
    ratios = [r['ratio'] for r in results]
    psnrs = [r['bhuh_psnr'] for r in results]

    # Include previous data points
    all_ns = [3, 5, 8, 10, 20] + ns
    all_ratios = [0.87, 1.44, 2.20, 2.69, 4.77] + ratios
    all_psnrs = [24.24, 23.86, 23.91, 23.11, 29.76] + psnrs

    # Fit models
    # 1. Linear: ratio = a + b*N
    a_lin, b_lin = np.polyfit(all_ns, all_ratios, 1)

    # 2. Logarithmic: ratio = a + b*ln(N)
    log_ns = np.log(all_ns)
    a_log, b_log = np.polyfit(log_ns, all_ratios, 1)

    # 3. Power law: ratio = a * N^b → ln(ratio) = ln(a) + b*ln(N)
    log_ratios = np.log(all_ratios)
    b_pow, log_a_pow = np.polyfit(log_ns, log_ratios, 1)
    a_pow = np.exp(log_a_pow)

    # R² for each
    r2_lin = 1 - np.sum((np.array(all_ratios) - (a_lin + b_lin*np.array(all_ns)))**2) / np.sum((np.array(all_ratios) - np.mean(all_ratios))**2)
    r2_log = 1 - np.sum((np.array(all_ratios) - (a_log + b_log*np.array(log_ns)))**2) / np.sum((np.array(all_ratios) - np.mean(all_ratios))**2)
    r2_pow = 1 - np.sum((np.array(log_ratios) - (log_a_pow + b_pow*np.array(log_ns)))**2) / np.sum((np.array(log_ratios) - np.mean(log_ratios))**2)

    print(f"  Linear:      ratio = {a_lin:.3f} + {b_lin:.3f}*N   (R²={r2_lin:.4f})")
    print(f"  Logarithmic: ratio = {a_log:.3f} + {b_log:.3f}*ln(N)  (R²={r2_log:.4f})")
    print(f"  Power law:   ratio = {a_pow:.3f} * N^{b_pow:.3f}   (R²={r2_pow:.4f})")
    print()

    # Best fit
    best_fit = max([('Linear', r2_lin, a_lin, b_lin),
                    ('Logarithmic', r2_log, a_log, b_log),
                    ('Power law', r2_pow, a_pow, b_pow)], key=lambda x: x[1])
    print(f"  Best fit: {best_fit[0]} (R²={best_fit[1]:.4f})")
    print()

    # Projections
    print("  Projections:")
    for n_proj in [300, 500, 1000]:
        if best_fit[0] == 'Linear':
            proj = best_fit[2] + best_fit[3] * n_proj
        elif best_fit[0] == 'Logarithmic':
            proj = best_fit[2] + best_fit[3] * np.log(n_proj)
        else:
            proj = best_fit[2] * n_proj ** best_fit[3]
        print(f"    N={n_proj}: projected ratio = {proj:.1f}x")

    # PSNR trend
    print()
    print("  PSNR trend (BHUH quality vs N):")
    for n_val, p in zip(all_ns, all_psnrs):
        print(f"    N={n_val:>3}: {p:.2f} dB")
    psnr_corr = np.corrcoef(all_ns, all_psnrs)[0,1]
    print(f"  Correlation N vs PSNR: {psnr_corr:.3f}")
    if psnr_corr < -0.5:
        print("  ✅ PSNR DECREASES with N — confirms capacity saturation")
    elif psnr_corr > 0.5:
        print("  ⚠️ PSNR INCREASES with N — unexpected")
    else:
        print("  ⚠️ No clear PSNR trend")

    # ============================================================
    # Analysis
    # ============================================================
    print()
    print("="*72)
    print("ANALYSIS")
    print("="*72)
    print()
    print(f"  Seed stability: std={np.std(seed_psnrs):.2f} dB "
          f"({'STABLE' if np.std(seed_psnrs) < 1.0 else 'UNSTABLE'})")
    print(f"  Best curve fit: {best_fit[0]} (R²={best_fit[1]:.4f})")
    print(f"  PSNR correlation with N: {psnr_corr:.3f}")
    print()

    # Check if quality is usable at each N
    for r in results:
        usable = "USABLE" if r['bhuh_psnr'] > 20 else "UNUSABLE"
        print(f"  N={r['n']:>3}: {r['bhuh_psnr']:.1f} dB ({usable}), "
              f"ratio={r['ratio']:.1f}x, range=[{r['bhuh_min_psnr']:.1f}, {r['bhuh_max_psnr']:.1f}]")

    print()
    verdict = (f"MAPPED: Scaling curve from N=50 to N=200. "
               f"Best fit: {best_fit[0]} (R²={best_fit[1]:.4f}). "
               f"Seed stability: {np.std(seed_psnrs):.2f} dB. "
               f"Quality at N=200: {results[-1]['bhuh_psnr']:.1f} dB. "
               f"PSNR-N correlation: {psnr_corr:.3f}.")
    print(f"  VERDICT: {verdict}")

    # Output JSON
    output = {
        'experiment': 22,
        'name': 'N=150/200 Sublinear Collapse Mapping',
        'protocol': 'CPU-only, torch.set_num_threads(8), seed=0 default',
        'params': {'hidden_bhuh': 64, 'hidden_coin': 32, 'omega': 15,
                   'epochs': 50, 'lr': 1e-3, 'size': 64},
        'scaling_results': results,
        'seed_sensitivity': seed_results,
        'curve_fits': {
            'linear': {'a': float(a_lin), 'b': float(b_lin), 'r2': float(r2_lin)},
            'logarithmic': {'a': float(a_log), 'b': float(b_log), 'r2': float(r2_log)},
            'power_law': {'a': float(a_pow), 'b': float(b_pow), 'r2': float(r2_pow)},
        },
        'best_fit': best_fit[0],
        'psnr_correlation': float(psnr_corr),
        'verdict': verdict,
    }
    print(f"\n--- JSON ---")
    print(json.dumps(output, indent=2, default=str))

    return output

if __name__ == '__main__':
    run()
