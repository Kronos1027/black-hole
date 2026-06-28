#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
BHUH Experiment 11: KAN vs SIREN — Does KAN solve the Mandelbrot problem?

Based on: Liu et al., "KAN: Kolmogorov-Arnold Networks" (ICLR 2025, arXiv 2404.19756)

KAN uses LEARNABLE splines as activation functions instead of fixed sines.
Theory: Kolmogorov-Arnold theorem says any continuous function can be
represented as sum of univariate compositions — exactly what KAN implements.

HYPOTHESIS: KAN can represent sharp/fractal signals (Mandelbrot) that SIREN cannot.
If true: K_BHUH(Mandelbrot) << K_SIREN(Mandelbrot) → Axiom 14 validated.

TEST:
  1. Generate Mandelbrot, smooth gaussian, and natural photo
  2. Fit SIREN (sin activations) and KAN-style (spline activations) to each
  3. Compare: PSNR, parameter count, convergence

NOTE: We implement a SIMPLIFIED KAN (learnable piecewise-linear) since
we don't have the full KAN library. This tests the CONCEPT honestly.
"""
import os, sys, time, json, zlib, struct
import numpy as np
from PIL import Image

def make_mandelbrot(n, max_iter=30):
    x = np.linspace(-2, 1, n); y = np.linspace(-1.5, 1.5, n)
    X, Y = np.meshgrid(x, y)
    C = X + 1j*Y; Z = np.zeros_like(C)
    img = np.zeros_like(X, dtype=np.float32)
    for i in range(max_iter):
        Z = Z*Z + C
        mask = (np.abs(Z) < 2) & (img == 0)
        img[mask] = i / max_iter
    return (img * 255).astype(np.uint8)

def make_gaussian(n):
    x = np.linspace(0, 1, n); y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    return (np.exp(-((X-0.5)**2 + (Y-0.5)**2)*8) * 255).astype(np.uint8)

def make_natural(n):
    """Real photo crop."""
    from skimage.data import astronaut
    arr = astronaut()
    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    pil = Image.fromarray(gray.astype(np.uint8))
    pil = pil.resize((n, n), Image.LANCZOS)
    return np.array(pil)

def compute_psnr(orig, recon):
    err = orig.astype(np.float64) - recon.astype(np.float64)
    mse = float(np.mean(err**2))
    return 100.0 if mse == 0 else float(10*np.log10(255.0**2/mse))

def train_siren(image, hidden=32, omega=15.0, epochs=300, lr=1e-3):
    """Standard SIREN with sin activations."""
    import torch, torch.nn as nn
    torch.manual_seed(0)
    n = image.shape[0]
    coords = np.stack(np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n)),axis=-1).reshape(-1,2)
    target = (image.astype(np.float32)/255.0).flatten()
    class Siren(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, hidden); self.fc2 = nn.Linear(hidden, hidden); self.fc3 = nn.Linear(hidden, 1)
            self.omega = omega
            for layer in [self.fc1, self.fc2]:
                bound = 1.0/layer.in_features if layer is self.fc1 else np.sqrt(6.0/hidden)/omega
                layer.weight.data.uniform_(-bound,bound); layer.bias.data.uniform_(-bound,bound)
            bound = np.sqrt(6.0/hidden)/omega
            self.fc3.weight.data.uniform_(-bound,bound); self.fc3.bias.data.uniform_(-bound,bound)
        def forward(self, x):
            h = torch.sin(self.omega * self.fc1(x))
            h = torch.sin(self.omega * self.fc2(h))
            return self.fc3(h)
    model = Siren(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32); yt = torch.tensor(target, dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad(); pred = model(xt).squeeze(-1)
        loss = ((pred-yt)**2).mean(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): recon = model(xt).squeeze(-1).numpy()
    recon_img = (recon*255).clip(0,255).astype(np.uint8).reshape(n,n)
    psnr = compute_psnr(image, recon_img)
    params = sum(int(np.prod(p.shape)) for p in model.parameters())
    return {'psnr_db': psnr, 'n_params': params, 'loss': float(loss.detach())}

def train_kan_simple(image, hidden=32, n_knots=5, epochs=300, lr=1e-3):
    """Simplified KAN: learnable piecewise-linear activation (spline approximation).
    
    Instead of sin(omega * Wx + b), use:
      spline(Wx + b) where spline is learned piecewise-linear
    
    This captures the ESSENCE of KAN: learnable activation functions.
    Full KAN uses B-splines; we use piecewise-linear for simplicity.
    """
    import torch, torch.nn as nn
    torch.manual_seed(0)
    n = image.shape[0]
    coords = np.stack(np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n)),axis=-1).reshape(-1,2)
    target = (image.astype(np.float32)/255.0).flatten()
    
    class LearnableSpline(nn.Module):
        """Piecewise-linear activation with learnable knot values."""
        def __init__(self, n_knots=5, range_val=3.0):
            super().__init__()
            self.n_knots = n_knots
            self.range_val = range_val
            # Learnable knot values (heights at each knot)
            self.knot_values = nn.Parameter(torch.linspace(-1, 1, n_knots))
            # Fixed knot positions
            knots = torch.linspace(-range_val, range_val, n_knots)
            self.register_buffer('knots', knots)
        
        def forward(self, x):
            # Piecewise linear interpolation
            # Clamp to range, then interpolate
            x_clamped = torch.clamp(x, -self.range_val, self.range_val)
            # Find which interval each value falls in
            idx = torch.searchsorted(self.knots, x_clamped) - 1
            idx = torch.clamp(idx, 0, self.n_knots - 2)
            # Linear interpolation
            x0 = self.knots[idx]
            x1 = self.knots[idx + 1]
            y0 = self.knot_values[idx]
            y1 = self.knot_values[idx + 1]
            t = (x_clamped - x0) / (x1 - x0 + 1e-8)
            return y0 + t * (y1 - y0)
    
    class KANSimple(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, hidden)
            self.act1 = LearnableSpline(n_knots)
            self.fc2 = nn.Linear(hidden, hidden)
            self.act2 = LearnableSpline(n_knots)
            self.fc3 = nn.Linear(hidden, 1)
            # Initialize
            for layer in [self.fc1, self.fc2, self.fc3]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        def forward(self, x):
            h = self.act1(self.fc1(x))
            h = self.act2(self.fc2(h))
            return self.fc3(h)
    
    model = KANSimple(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32); yt = torch.tensor(target, dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad(); pred = model(xt).squeeze(-1)
        loss = ((pred-yt)**2).mean(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): recon = model(xt).squeeze(-1).numpy()
    recon_img = (recon*255).clip(0,255).astype(np.uint8).reshape(n,n)
    psnr = compute_psnr(image, recon_img)
    params = sum(int(np.prod(p.shape)) for p in model.parameters())
    return {'psnr_db': psnr, 'n_params': params, 'loss': float(loss.detach())}

def run():
    print("="*72)
    print("BHUH EXPERIMENT 11: KAN vs SIREN")
    print("="*72)
    print()
    print("Based on: Liu et al., KAN (ICLR 2025, arXiv 2404.19756)")
    print("Hypothesis: KAN (learnable splines) beats SIREN (fixed sin)")
    print("             on sharp/fractal signals (Mandelbrot)")
    print()
    import torch

    N = 32  # small for speed
    print(f"--- Generating test images ({N}×{N}) ---")
    images = {
        'mandelbrot': make_mandelbrot(N),
        'gaussian': make_gaussian(N),
        'natural': make_natural(N),
    }
    for name, img in images.items():
        print(f"  {name}: shape={img.shape}, range=[{img.min()}, {img.max()}]")
    print()

    results = {}
    for name, img in images.items():
        print(f"\n--- {name} ---")
        sys.stdout.flush()
        
        # SIREN
        print(f"  SIREN (h=32, sin, 300 epochs)...", end=' ')
        sys.stdout.flush()
        t0 = time.time()
        siren_r = train_siren(img, hidden=32, epochs=300)
        t_siren = time.time()-t0
        print(f"PSNR={siren_r['psnr_db']:.2f}dB, params={siren_r['n_params']}, {t_siren:.1f}s")
        
        # KAN (simplified)
        print(f"  KAN (h=32, spline, 300 epochs)...", end=' ')
        sys.stdout.flush()
        t0 = time.time()
        kan_r = train_kan_simple(img, hidden=32, n_knots=5, epochs=300)
        t_kan = time.time()-t0
        print(f"PSNR={kan_r['psnr_db']:.2f}dB, params={kan_r['n_params']}, {t_kan:.1f}s")
        
        psnr_diff = kan_r['psnr_db'] - siren_r['psnr_db']
        print(f"  → KAN - SIREN: {psnr_diff:+.2f} dB")
        
        results[name] = {
            'siren_psnr': siren_r['psnr_db'], 'siren_params': siren_r['n_params'],
            'kan_psnr': kan_r['psnr_db'], 'kan_params': kan_r['n_params'],
            'psnr_diff': float(psnr_diff),
        }

    # Summary
    print()
    print("="*72)
    print("RESULTS: SIREN vs KAN")
    print("="*72)
    print(f"{'Image':<14} {'SIREN dB':>9} {'KAN dB':>9} {'Diff':>8} {'SIREN params':>13} {'KAN params':>11}")
    for name, r in results.items():
        print(f"{name:<14} {r['siren_psnr']:>8.2f} {r['kan_psnr']:>8.2f} "
              f"{r['psnr_diff']:>+7.2f} {r['siren_params']:>13} {r['kan_params']:>11}")

    # Analysis
    print()
    print("="*72)
    print("ANALYSIS")
    print("="*72)
    print()
    
    mandel_diff = results['mandelbrot']['psnr_diff']
    gaussian_diff = results['gaussian']['psnr_diff']
    natural_diff = results['natural']['psnr_diff']
    
    print(f"  Mandelbrot (fractal): KAN vs SIREN = {mandel_diff:+.2f} dB")
    print(f"  Gaussian (smooth):    KAN vs SIREN = {gaussian_diff:+.2f} dB")
    print(f"  Natural (photo):      KAN vs SIREN = {natural_diff:+.2f} dB")
    print()
    
    if mandel_diff > 3:
        print(f"  ✅ KAN SOLVES MANDELBROT PROBLEM — {mandel_diff:.1f} dB better than SIREN")
        print(f"     Claude's suggestion VALIDATED: KAN handles fractals where SIREN fails")
        print(f"     This validates Axiom 14 (Kolmogorov Twin) — K_KAN can represent")
        print(f"     high-complexity signals that K_SIREN cannot.")
        verdict = (f"VALIDATED — KAN beats SIREN by {mandel_diff:.1f} dB on Mandelbrot. "
                   "Claude's KAN suggestion is correct. KAN solves the fractal problem.")
    elif mandel_diff > 0:
        print(f"  ⚠️ KAN slightly better on Mandelbrot ({mandel_diff:.1f} dB)")
        verdict = f"PARTIAL — KAN {mandel_diff:.1f} dB better, marginal"
    else:
        print(f"  ❌ KAN does NOT help on Mandelbrot")
        verdict = "INVALID — KAN doesn't solve fractal problem"
    
    print(f"\n  VERDICT: {verdict}")
    print()
    
    # Overall assessment
    avg_diff = np.mean([mandel_diff, gaussian_diff, natural_diff])
    print(f"  Average KAN-SIREN diff across all images: {avg_diff:+.2f} dB")
    if avg_diff > 1:
        print(f"  ✅ KAN is generally better — consider as BHUH generator alternative")
    elif avg_diff > -1:
        print(f"  ⚠️ KAN comparable to SIREN — no clear winner")
    else:
        print(f"  ❌ KAN worse than SIREN overall — SIREN is better choice")

    return {'experiment': 11, 'name': 'KAN vs SIREN',
            'results': results, 'verdict': verdict,
            'mandelbrot_diff': float(mandel_diff)}

if __name__ == '__main__':
    result = run()
    print("\n--- JSON ---")
    print(json.dumps(result, indent=2, default=str))
