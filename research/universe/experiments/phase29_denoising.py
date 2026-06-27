# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 29: Image Denoising via SIREN
=====================================
Tests whether SIREN can act as a denoiser.

CONCEPT:
  SIREN fits a smooth function to data. If we train it on a NOISY image,
  the SIREN may learn the signal but not the noise (because noise is
  high-frequency and SIREN has limited capacity).

  This makes SIREN a natural denoiser: train on noisy → query → get clean.

HYPOTHESIS:
  SIREN trained on noisy image will produce cleaner output than input,
  because SIREN's limited capacity filters out high-frequency noise.

METHOD:
  1. Generate clean image
  2. Add Gaussian noise (σ=20, 30, 50)
  3. Train SIREN on noisy image
  4. Compare SIREN output with clean original
  5. Measure PSNR improvement

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren


def run_phase29_experiment(verbose=True):
    """Run Phase 29 Denoising experiment."""
    print("=" * 80)
    print("🧪 Phase 29: Image Denoising via SIREN")
    print("=" * 80)

    device = 'cpu'
    size = 128

    # Generate clean image
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    clean = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(2, 5, 2)
            amp = rng.uniform(40, 80)
            clean[:, :, c] += amp * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
    clean = ((clean - clean.min()) / (clean.max() - clean.min()) * 255).astype(np.uint8)

    # Test different noise levels
    noise_levels = [10, 20, 30, 50]
    results = []

    print(f"\n{'Noise σ':>8} {'Noisy PSNR':>12} {'SIREN PSNR':>12} {'Improvement':>12} {'Denoised?':>10}")
    print("-" * 60)

    for sigma in noise_levels:
        # Add noise
        noise = np.random.normal(0, sigma, clean.shape)
        noisy = np.clip(clean.astype(float) + noise, 0, 255).astype(np.uint8)

        # PSNR of noisy vs clean
        mse_noisy = np.mean((clean.astype(float) - noisy.astype(float))**2)
        psnr_noisy = 10 * np.log10(255**2 / max(mse_noisy, 1e-10))

        # Train SIREN on noisy image
        model, loss = train_single_siren(noisy, epochs=80, device=device, verbose=False)

        # Get SIREN output (denoised)
        coords = get_coordinates(size, device)
        with torch.no_grad():
            pred = model(coords)
        denoised = (pred.cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)

        # PSNR of denoised vs clean
        mse_denoised = np.mean((clean.astype(float) - denoised.astype(float))**2)
        psnr_denoised = 10 * np.log10(255**2 / max(mse_denoised, 1e-10))

        improvement = psnr_denoised - psnr_noisy
        denoised_label = "✅ Yes" if improvement > 0.5 else "❌ No"

        print(f"{sigma:>7} {psnr_noisy:>10.1f}dB {psnr_denoised:>10.1f}dB {improvement:>+10.1f}dB {denoised_label:>10}")

        results.append({
            'sigma': sigma,
            'psnr_noisy': psnr_noisy,
            'psnr_denoised': psnr_denoised,
            'improvement': improvement,
        })

    # Summary
    print(f"\n{'='*80}")
    print("📊 PHASE 29 SUMMARY — SIREN DENOISING")
    print(f"{'='*80}")

    avg_improvement = np.mean([r['improvement'] for r in results])
    best = max(results, key=lambda x: x['improvement'])

    if avg_improvement > 0.5:
        print(f"\n  ✅ SIREN denoises! Average improvement: {avg_improvement:+.1f}dB")
        print(f"  Best: σ={best['sigma']} → {best['improvement']:+.1f}dB improvement")
        print(f"\n  📋 Mechanism:")
        print(f"  - SIREN has limited capacity (fixed network size)")
        print(f"  - It fits the SMOOTH signal (low frequency)")
        print(f"  - Noise (high frequency) is filtered out naturally")
        print(f"  - This is 'capacity-based denoising' — no explicit noise model needed")
    elif avg_improvement > -1:
        print(f"\n  ⚠️  SIREN has minimal denoising effect ({avg_improvement:+.1f}dB)")
    else:
        print(f"\n  ❌ SIREN doesn't denoise ({avg_improvement:+.1f}dB)")

    print(f"\n  📋 Applications:")
    print(f"  - Medical imaging (MRI/CT noise removal)")
    print(f"  - Low-light photography (ISO noise)")
    print(f"  - Scientific instruments (sensor noise)")
    print(f"  - Combined: compress AND denoise in one step!")

    return results


if __name__ == '__main__':
    results = run_phase29_experiment(verbose=True)
