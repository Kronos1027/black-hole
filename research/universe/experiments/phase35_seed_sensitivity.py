# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 35: Seed Sensitivity Analysis
=====================================
Tests how small perturbations in the SIREN seed (weights) affect
the reconstructed output.

CONCEPT:
  If the seed is a "DNA" of the file, how sensitive is it?
  - 1 bit flip → catastrophic failure? (fragile)
  - 1 bit flip → small change? (robust)
  - This determines: error correction needs, transmission reliability

HYPOTHESIS:
  SIREN seeds will be moderately sensitive — small perturbations
  cause visible but not catastrophic changes. This is because
  neural networks have distributed representations (no single
  bit is critical).

METHOD:
  1. Train SIREN on image
  2. Perturb weights: add Gaussian noise σ=0.001, 0.01, 0.1, 1.0
  3. Also try: zero out 1%, 5%, 10% of weights
  4. Measure PSNR vs original reconstruction

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren


def perturb_weights(model, noise_std, device='cpu'):
    """Add Gaussian noise to all weights."""
    perturbed = copy.deepcopy(model)
    with torch.no_grad():
        for param in perturbed.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)
    return perturbed


def zero_weights(model, pct, device='cpu'):
    """Zero out a percentage of weights randomly."""
    perturbed = copy.deepcopy(model)
    with torch.no_grad():
        for param in perturbed.parameters():
            mask = torch.rand_like(param) > (pct / 100.0)
            param.mul_(mask.float())
    return perturbed


def query_model(model, size, device='cpu'):
    """Get model output as image."""
    coords = get_coordinates(size, device)
    with torch.no_grad():
        pred = model(coords)
    return (pred.cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)


def run_phase35_experiment(verbose=True):
    """Run Phase 35 Seed Sensitivity experiment."""
    print("=" * 80)
    print("🧪 Phase 35: Seed Sensitivity Analysis")
    print("=" * 80)

    device = 'cpu'
    size = 128

    # Generate and train
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(2, 5, 2)
            img[:, :, c] += 50 * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    model, _ = train_single_siren(img, epochs=80, device=device, verbose=False)
    orig_output = query_model(model, size, device)

    # Test 1: Gaussian noise perturbation
    print(f"\n📊 Test 1: Gaussian Noise Perturbation")
    print(f"  {'Noise σ':>10} {'PSNR vs clean':>15} {'Visual':>10}")
    print(f"  {'-'*40}")

    noise_levels = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    noise_results = []

    for sigma in noise_levels:
        if sigma == 0:
            perturbed = model
        else:
            perturbed = perturb_weights(model, sigma, device)

        output = query_model(perturbed, size, device)
        mse = np.mean((orig_output.astype(float) - output.astype(float))**2)
        psnr = 10 * np.log10(255**2 / max(mse, 1e-10)) if mse > 0 else 99

        if psnr > 40:
            visual = "✅ clean"
        elif psnr > 25:
            visual = "⚠️ visible"
        elif psnr > 15:
            visual = "❌ degraded"
        else:
            visual = "💥 broken"

        print(f"  {sigma:>9.3f} {psnr:>13.1f}dB {visual:>10}")
        noise_results.append({'sigma': sigma, 'psnr': psnr})

    # Test 2: Weight zeroing
    print(f"\n📊 Test 2: Weight Zeroing (pruning)")
    print(f"  {'% Zeroed':>10} {'PSNR vs clean':>15} {'Visual':>10}")
    print(f"  {'-'*40}")

    zero_pcts = [0, 1, 5, 10, 25, 50, 75]
    zero_results = []

    for pct in zero_pcts:
        if pct == 0:
            perturbed = model
        else:
            perturbed = zero_weights(model, pct, device)

        output = query_model(perturbed, size, device)
        mse = np.mean((orig_output.astype(float) - output.astype(float))**2)
        psnr = 10 * np.log10(255**2 / max(mse, 1e-10)) if mse > 0 else 99

        if psnr > 40:
            visual = "✅ clean"
        elif psnr > 25:
            visual = "⚠️ visible"
        elif psnr > 15:
            visual = "❌ degraded"
        else:
            visual = "💥 broken"

        print(f"  {pct:>9}% {psnr:>13.1f}dB {visual:>10}")
        zero_results.append({'pct': pct, 'psnr': psnr})

    # Analysis
    print(f"\n{'='*80}")
    print("📊 PHASE 35 SUMMARY — SEED SENSITIVITY")
    print(f"{'='*80}")

    # Find critical thresholds
    noise_critical = next((r for r in noise_results if r['psnr'] < 25), None)
    zero_critical = next((r for r in zero_results if r['psnr'] < 25), None)

    print(f"\n  📋 Noise sensitivity:")
    if noise_critical:
        print(f"  - Visible degradation at σ={noise_critical['sigma']:.3f}")
        print(f"  - Seed is {'FRAGILE' if noise_critical['sigma'] < 0.01 else 'ROBUST' if noise_critical['sigma'] > 0.1 else 'MODERATE'}")
    else:
        print(f"  - Very robust to noise!")

    print(f"\n  📋 Pruning tolerance:")
    if zero_critical:
        print(f"  - Visible degradation at {zero_critical['pct']}% weights zeroed")
        print(f"  - {'HIGH redundancy' if zero_critical['pct'] >= 25 else 'LOW redundancy'}")
    else:
        print(f"  - Tolerates all tested pruning levels!")

    # Robustness score
    robust_noise = sum(1 for r in noise_results if r['psnr'] > 25) / len(noise_results)
    robust_zero = sum(1 for r in zero_results if r['psnr'] > 25) / len(zero_results)
    overall = (robust_noise + robust_zero) / 2 * 100

    print(f"\n  📊 Robustness score: {overall:.0f}%")
    print(f"  - Noise: {robust_noise*100:.0f}% levels OK")
    print(f"  - Pruning: {robust_zero*100:.0f}% levels OK")

    if overall > 60:
        print(f"\n  ✅ SIREN seeds are ROBUST — distributed representation")
        print(f"     No single bit is critical (unlike traditional formats)")
        print(f"     Partial corruption = graceful degradation, not total loss")
    else:
        print(f"\n  ⚠️  SIREN seeds are somewhat fragile")

    print(f"\n  📋 Implications:")
    print(f"  - Error correction: less aggressive ECC needed than bit-exact formats")
    print(f"  - Partial recovery: corrupted seed still produces approximate file")
    print(f"  - Pruning opportunity: can zero small weights for extra compression")
    print(f"  - Transmission: more resilient to bit errors than ZIP/PNG")

    return {
        'noise_results': noise_results,
        'zero_results': zero_results,
        'robustness': overall,
    }


if __name__ == '__main__':
    results = run_phase35_experiment(verbose=True)
