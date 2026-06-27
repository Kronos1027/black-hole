# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 28: Image Super-Resolution via SIREN
=============================================
Tests whether SIREN can upscale images beyond their original resolution
with meaningful detail (not just interpolation).

CONCEPT:
  Traditional upscaling (bilinear, bicubic) interpolates between pixels.
  SIREN learns the underlying FUNCTION, so it can generate new detail
  at higher resolutions — potentially sharper than bicubic.

HYPOTHESIS:
  SIREN trained on 64x64, queried at 256x256, will produce sharper
  results than bicubic interpolation, because SIREN learns the signal's
  frequency content, not just pixel values.

METHOD:
  1. Generate 128x128 reference image
  2. Downsample to 32x32 (low-res input)
  3. Train SIREN on 32x32
  4. Query SIREN at 128x128 (4x upscale)
  5. Compare with bicubic interpolation
  6. Measure PSNR vs original 128x128

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, zlib, io
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren


def run_phase28_experiment(verbose=True):
    """Run Phase 28 Super-Resolution experiment."""
    print("=" * 80)
    print("🧪 Phase 28: Image Super-Resolution via SIREN")
    print("=" * 80)

    device = 'cpu'

    # Generate reference image at 128x128
    size_ref = 128
    size_low = 32  # 4x downscale
    size_up = 128   # 4x upscale

    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size_ref, 0:size_ref].astype(np.float32) / size_ref
    ref_img = np.zeros((size_ref, size_ref, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(5):
            kx, ky = rng.integers(2, 8, 2)
            amp = rng.uniform(40, 80)
            phase = rng.uniform(0, 2 * np.pi)
            ref_img[:, :, c] += amp * np.sin(2 * np.pi * kx * xs + phase) * np.cos(2 * np.pi * ky * ys + phase)
    ref_img = ((ref_img - ref_img.min()) / (ref_img.max() - ref_img.min()) * 255).astype(np.uint8)

    # Downsample to 32x32
    low_img = np.array(Image.fromarray(ref_img).resize((size_low, size_low), Image.LANCZOS))

    print(f"\n  Reference: {size_ref}x{size_ref}")
    print(f"  Low-res:   {size_low}x{size_low} (4x downscale)")
    print(f"  Upscale:   {size_up}x{size_up} (4x upscale)")

    # Method 1: Bicubic interpolation
    print(f"\n🔵 Baseline: Bicubic interpolation...")
    bicubic_img = np.array(Image.fromarray(low_img).resize((size_up, size_up), Image.BICUBIC))
    mse_bicubic = np.mean((ref_img.astype(float) - bicubic_img.astype(float))**2)
    psnr_bicubic = 10 * np.log10(255**2 / max(mse_bicubic, 1e-10))

    # Method 2: Bilinear interpolation
    bilinear_img = np.array(Image.fromarray(low_img).resize((size_up, size_up), Image.BILINEAR))
    mse_bilinear = np.mean((ref_img.astype(float) - bilinear_img.astype(float))**2)
    psnr_bilinear = 10 * np.log10(255**2 / max(mse_bilinear, 1e-10))

    # Method 3: LANCZOS interpolation
    lanczos_img = np.array(Image.fromarray(low_img).resize((size_up, size_up), Image.LANCZOS))
    mse_lanczos = np.mean((ref_img.astype(float) - lanczos_img.astype(float))**2)
    psnr_lanczos = 10 * np.log10(255**2 / max(mse_lanczos, 1e-10))

    # Method 4: SIREN super-resolution
    print(f"\n🌌 BHUH: SIREN super-resolution...")
    print(f"  Training SIREN on {size_low}x{size_low}...")
    model, loss = train_single_siren(low_img, epochs=100, device=device, verbose=verbose)

    # Query at 128x128
    print(f"  Querying SIREN at {size_up}x{size_up} (4x upscale)...")
    coords = get_coordinates(size_up, device)
    with torch.no_grad():
        pred = model(coords)
    siren_img = (pred.cpu().numpy().reshape(size_up, size_up, 3) * 255).clip(0, 255).astype(np.uint8)
    mse_siren = np.mean((ref_img.astype(float) - siren_img.astype(float))**2)
    psnr_siren = 10 * np.log10(255**2 / max(mse_siren, 1e-10))

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 28 RESULTS — SUPER-RESOLUTION (4x upscale, 32→128)")
    print(f"{'='*80}")
    print(f"\n  {'Method':<25} {'PSNR vs Original':>18} {'MSE':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Bilinear':<25} {psnr_bilinear:>16.1f}dB {mse_bilinear:>10.2f}")
    print(f"  {'Bicubic':<25} {psnr_bicubic:>16.1f}dB {mse_bicubic:>10.2f}")
    print(f"  {'LANCZOS':<25} {psnr_lanczos:>16.1f}dB {mse_lanczos:>10.2f}")
    print(f"  {'SIREN (BHUH)':<25} {psnr_siren:>16.1f}dB {mse_siren:>10.2f}")

    best_method = max([('Bilinear', psnr_bilinear), ('Bicubic', psnr_bicubic),
                       ('LANCZOS', psnr_lanczos), ('SIREN', psnr_siren)],
                      key=lambda x: x[1])

    print(f"\n  🏆 Best: {best_method[0]} ({best_method[1]:.1f}dB)")

    if psnr_siren > psnr_bicubic:
        diff = psnr_siren - psnr_bicubic
        print(f"\n  ✅ SIREN beats bicubic by {diff:.1f}dB!")
        print(f"     SIREN learns the signal's frequency content,")
        print(f"     generating detail that interpolation cannot.")
    elif psnr_siren > psnr_bilinear:
        print(f"\n  ⚠️  SIREN beats bilinear but not bicubic")
    else:
        print(f"\n  ❌ SIREN doesn't beat interpolation at 4x upscale")

    print(f"\n  📋 Key insight:")
    print(f"  - SIREN trained on low-res learns the underlying FUNCTION")
    print(f"  - Querying at high-res generates new detail (not just interpolation)")
    print(f"  - This is impossible with traditional codecs (they store pixels, not functions)")
    print(f"  - Applications: AI upscaling, retro gaming, medical imaging")

    return {
        'psnr_bilinear': psnr_bilinear,
        'psnr_bicubic': psnr_bicubic,
        'psnr_lanczos': psnr_lanczos,
        'psnr_siren': psnr_siren,
    }


if __name__ == '__main__':
    results = run_phase28_experiment(verbose=True)
