# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 20: Adaptive Quality via Perceptual Loss
================================================
Tests whether SIREN can adapt quality based on perceptual importance.

CONCEPT:
  Not all pixels are equally important. Human vision is more sensitive
  to edges than flat regions. SIREN can be trained with perceptual
  weighting:
  - Edge regions: higher weight (more SIREN capacity)
  - Flat regions: lower weight (less capacity)

  This should achieve better visual quality at same file size.

HYPOTHESIS:
  Perceptual-weighted SIREN will achieve higher PSNR-SSIM (perceptual
  quality) than uniform MSE SIREN at the same file size.

METHOD:
  1. Generate image with edges + flat regions
  2. Train SIREN with uniform MSE (baseline)
  3. Train SIREN with edge-weighted MSE (adaptive)
  4. Compare SSIM (structural similarity, not just MSE)

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, measure_model_size_compressed


def compute_edge_weights(image, sigma=1.0):
    """Compute edge-based importance weights for each pixel."""
    img_gray = np.mean(image, axis=2).astype(np.float32) / 255.0

    # Sobel edges
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    from scipy.ndimage import convolve
    ix = convolve(img_gray, gx)
    iy = convolve(img_gray, gy)
    edge_magnitude = np.sqrt(ix**2 + iy**2)

    # Normalize to [0.5, 3.0] range (edges get 3x weight, flat 0.5x)
    if edge_magnitude.max() > 0:
        edge_norm = edge_magnitude / edge_magnitude.max()
    else:
        edge_norm = np.zeros_like(edge_magnitude)

    weights = 0.5 + 2.5 * edge_norm  # [0.5, 3.0]
    return weights


def train_siren_weighted(image, weights, epochs=100, lr=3e-3, device='cpu', verbose=False):
    """Train SIREN with perceptual weighting."""
    size = image.shape[0]
    coords = get_coordinates(size, device)
    pixels = torch.from_numpy(image.astype(np.float32) / 255.0).reshape(-1, 3).to(device)
    w = torch.from_numpy(weights.reshape(-1, 1).astype(np.float32)).to(device)

    model = SIREN(in_features=2, hidden_features=32, hidden_layers=2, out_features=3, omega_0=30.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(coords)
        # Weighted MSE
        diff = (pred - pixels) ** 2
        loss = (diff * w).mean()
        loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"  Weighted Epoch {epoch}: loss={loss.item():.6f}")

    return model, loss.item()


def compute_psnr(orig, recon):
    """Compute PSNR."""
    mse = np.mean((orig.astype(float) - recon.astype(float))**2)
    if mse == 0:
        return 99.0
    return 10 * np.log10(255**2 / mse)


def compute_simple_ssim(orig, recon):
    """Simplified SSIM (global, not windowed)."""
    orig = orig.astype(np.float32)
    recon = recon.astype(np.float32)
    mu1 = orig.mean()
    mu2 = recon.mean()
    sigma1 = orig.std()
    sigma2 = recon.std()
    sigma12 = np.mean((orig - mu1) * (recon - mu2))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    return ssim


def run_phase20_experiment(verbose=True):
    """Run Phase 20 Adaptive Quality experiment."""
    print("=" * 80)
    print("🧪 Phase 20: Adaptive Quality via Perceptual Loss")
    print("=" * 80)

    device = 'cpu'
    size = 128

    # Generate image with edges + flat regions
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Add sharp edges (square boundary)
    img[30:98, 30:98, :] = 200  # bright square
    img[32:96, 32:96, :] = 30   # dark inside (edge at boundary)

    # Add smooth gradient (flat region)
    for c in range(3):
        img[:, :, c] += 30 * np.sin(2 * np.pi * 2 * xs) * np.cos(2 * np.pi * 2 * ys)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Compute edge weights
    weights = compute_edge_weights(img)

    zip_size = len(zlib.compress(img.tobytes(), 9))

    # Baseline: uniform MSE
    print("\n🔵 Baseline: Uniform MSE SIREN...")
    from phase1_multi_file_siren import train_single_siren
    uniform_model, uniform_loss = train_single_siren(img, epochs=120, device=device, verbose=verbose)
    uniform_size = measure_model_size_compressed(uniform_model)

    # Get prediction
    coords = get_coordinates(size, device)
    with torch.no_grad():
        uniform_pred = uniform_model(coords)
    uniform_img = (uniform_pred.cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)

    uniform_psnr = compute_psnr(img, uniform_img)
    uniform_ssim = compute_simple_ssim(img, uniform_img)

    # Adaptive: edge-weighted
    print("\n🌌 Adaptive: Edge-weighted SIREN...")
    weighted_model, weighted_loss = train_siren_weighted(img, weights, epochs=120, device=device, verbose=verbose)
    weighted_size = measure_model_size_compressed(weighted_model)

    with torch.no_grad():
        weighted_pred = weighted_model(coords)
    weighted_img = (weighted_pred.cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)

    weighted_psnr = compute_psnr(img, weighted_img)
    weighted_ssim = compute_simple_ssim(img, weighted_img)

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 20 RESULTS — ADAPTIVE QUALITY")
    print(f"{'='*80}")
    print(f"\n  {'Method':<25} {'Size':>8} {'PSNR':>8} {'SSIM':>8} {'vs ZIP':>8}")
    print(f"  {'-'*60}")
    print(f"  {'Uniform MSE':<25} {uniform_size:>7,}B {uniform_psnr:>6.1f}dB {uniform_ssim:>7.4f} {zip_size/uniform_size:>7.2f}x")
    print(f"  {'Edge-weighted':<25} {weighted_size:>7,}B {weighted_psnr:>6.1f}dB {weighted_ssim:>7.4f} {zip_size/weighted_size:>7.2f}x")

    ssim_improvement = (weighted_ssim - uniform_ssim) / uniform_ssim * 100

    print(f"\n  📋 Comparison:")
    print(f"  - SSIM: {uniform_ssim:.4f} → {weighted_ssim:.4f} ({'+' if ssim_improvement > 0 else ''}{ssim_improvement:.1f}%)")
    print(f"  - PSNR: {uniform_psnr:.1f} → {weighted_psnr:.1f}dB")
    print(f"  - Size: {uniform_size:,}B → {weighted_size:,}B")

    if weighted_ssim > uniform_ssim:
        print(f"\n  ✅ Edge-weighted SIREN achieves better perceptual quality!")
        print(f"     SSIM improved by {ssim_improvement:.1f}%")
    else:
        print(f"\n  ⚠️  No perceptual improvement (may need more training or different weighting)")

    return {
        'uniform_size': uniform_size,
        'weighted_size': weighted_size,
        'uniform_psnr': uniform_psnr,
        'weighted_psnr': weighted_psnr,
        'uniform_ssim': uniform_ssim,
        'weighted_ssim': weighted_ssim,
        'ssim_improvement': ssim_improvement,
    }


if __name__ == '__main__':
    results = run_phase20_experiment(verbose=True)
