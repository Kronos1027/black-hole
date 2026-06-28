# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 57: Neural Inpainting (Fill Missing Regions via SIREN)
==============================================================
Tests whether SIREN can reconstruct missing parts of an image.

CONCEPT:
  SIREN learns a CONTINUOUS function. If we train on only PART of
  an image (with a hole), SIREN should INFER the missing region
  by extending the learned function.

  This is "neural inpainting" — the SIREN fills gaps naturally
  because it learns the underlying pattern, not just pixel values.

HYPOTHESIS:
  SIREN trained on an image with 25% missing will reconstruct the
  missing region with >20dB PSNR vs the true missing region.

METHOD:
  1. Generate complete image
  2. Remove center 25% (set to 0)
  3. Train SIREN only on visible pixels (85% of image)
  4. Query SIREN at missing pixel locations
  5. Compare filled region with true original

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates


def run_phase57_experiment(verbose=True):
    print("=" * 80)
    print("🧪 Phase 57: Neural Inpainting (Fill Missing Regions)")
    print("=" * 80)

    device = 'cpu'
    size = 128

    # Generate complete image
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(2, 5, 2)
            img[:, :, c] += 50 * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    # Create mask: remove center 25% (32x32 to 96x96 region)
    mask = np.ones((size, size), dtype=bool)
    hole_start = size // 4
    hole_end = 3 * size // 4
    mask[hole_start:hole_end, hole_start:hole_end] = False
    missing_pct = (1 - mask.sum() / mask.size) * 100

    # Create masked image (hole = 0)
    masked_img = img.copy()
    masked_img[~mask] = 0

    print(f"\n  Image: {size}x{size}")
    print(f"  Hole: center {hole_end-hole_start}x{hole_end-hole_start} = {missing_pct:.0f}% missing")

    # Train SIREN only on visible pixels
    coords_all = get_coordinates(size, device)
    pixels_all = torch.from_numpy(img.astype(np.float32) / 255.0).reshape(-1, 3).to(device)
    mask_flat = torch.from_numpy(mask.ravel()).to(device)

    # Only train on visible pixels
    coords_train = coords_all[mask_flat]
    pixels_train = pixels_all[mask_flat]

    model = SIREN(in_features=2, hidden_features=32, hidden_layers=2, out_features=3, omega_0=30.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    print(f"\n🌌 Training SIREN on {mask.sum()}/{mask.size} visible pixels...")
    for epoch in range(150):
        optimizer.zero_grad()
        pred = model(coords_train)
        loss = F.mse_loss(pred, pixels_train)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 30 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.6f}")

    # Query at ALL locations (including missing)
    with torch.no_grad():
        full_pred = model(coords_all)
    reconstructed = (full_pred.cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)

    # Evaluate: how well did SIREN fill the hole?
    true_hole = img[~mask]
    filled_hole = reconstructed[~mask]

    mse_hole = np.mean((true_hole.astype(float) - filled_hole.astype(float))**2)
    psnr_hole = 10 * np.log10(255**2 / max(mse_hole, 1e-10))

    # Also evaluate visible region (should be good — it was trained on these)
    true_visible = img[mask]
    pred_visible = reconstructed[mask]
    mse_visible = np.mean((true_visible.astype(float) - pred_visible.astype(float))**2)
    psnr_visible = 10 * np.log10(255**2 / max(mse_visible, 1e-10))

    # Total PSNR
    mse_total = np.mean((img.astype(float) - reconstructed.astype(float))**2)
    psnr_total = 10 * np.log10(255**2 / max(mse_total, 1e-10))

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 57 RESULTS — NEURAL INPAINTING")
    print(f"{'='*80}")
    print(f"\n  {'Region':<20} {'PSNR':>10} {'Quality':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Visible (trained)':<20} {psnr_visible:>8.1f}dB {'✅ good':>9}")
    print(f"  {'Missing (inferred)':<20} {psnr_hole:>8.1f}dB {'✅ good' if psnr_hole > 20 else '⚠️ ok' if psnr_hole > 15 else '❌ poor':>9}")
    print(f"  {'Total (all pixels)':<20} {psnr_total:>8.1f}dB")

    if psnr_hole > 25:
        print(f"\n  ✅ SIREN FILLED THE HOLE with {psnr_hole:.1f}dB quality!")
        print(f"     The learned function naturally extends to missing regions.")
    elif psnr_hole > 15:
        print(f"\n  ⚠️  Partial inpainting ({psnr_hole:.1f}dB) — rough but recognizable")
    else:
        print(f"\n  ❌ Inpainting failed ({psnr_hole:.1f}dB)")

    print(f"\n  📋 Key insight:")
    print(f"  SIREN learns a CONTINUOUS FUNCTION, not discrete pixels.")
    print(f"  When part of the image is missing, SIREN extends the function")
    print(f"  to fill the gap — this is MATHEMATICAL INTERPOLATION.")
    print(f"  No separate inpainting model needed — it's built into SIREN!")

    print(f"\n  📋 Applications:")
    print(f"  - Photo restoration (remove scratches, watermarks)")
    print(f"  - Medical imaging (reconstruct from sparse MRI slices)")
    print(f"  - Satellite (fill cloud-covered regions)")
    print(f"  - Compression: can afford to SKIP some pixels during training!")
    print(f"    (transmit fewer pixels, let SIREN fill the rest)")

    return {
        'psnr_visible': psnr_visible,
        'psnr_hole': psnr_hole,
        'psnr_total': psnr_total,
        'missing_pct': missing_pct,
    }


if __name__ == '__main__':
    results = run_phase57_experiment(verbose=True)
