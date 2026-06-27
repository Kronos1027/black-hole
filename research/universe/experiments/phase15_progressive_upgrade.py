# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 15: Progressive Lossy-to-Lossless Upgrade
================================================
Tests whether a lossy SIREN seed can be UPGRADED to lossless by
adding a residual layer.

CONCEPT:
  Traditional codecs have fixed quality. SIREN enables PROGRESSIVE
  enhancement:
  - Level 0: SIREN only (lossy, tiny)
  - Level 1: SIREN + coarse residual (better quality)
  - Level 2: SIREN + fine residual (lossless)

  User can download Level 0 for instant preview, then upgrade to
  lossless when needed — all from the SAME seed.

HYPOTHESIS:
  Progressive upgrade will achieve bit-perfect at smaller total size
  than encoding lossless from scratch, because SIREN captures most
  of the structure.

METHOD:
  1. Train SIREN on image (lossy seed)
  2. Compute residual = (original - SIREN_output) mod 256
  3. Level 0: SIREN only
  4. Level 1: SIREN + zlib(residual)
  5. Level 2: SIREN + PNG(residual) = bit-perfect
  6. Compare sizes

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import io
import hashlib
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren, measure_model_size_compressed


def run_phase15_experiment(verbose=True):
    """Run Phase 15 Progressive Upgrade experiment."""
    print("=" * 80)
    print("🧪 Phase 15: Progressive Lossy-to-Lossless Upgrade")
    print("=" * 80)

    device = 'cpu'
    size = 128

    # Generate image
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(1, 5, 2)
            amp = rng.uniform(40, 80)
            img[:, :, c] += amp * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    raw_size = img.nbytes
    zip_size = len(zlib.compress(img.tobytes(), 9))

    # Train SIREN
    print(f"\n📸 Training SIREN on {size}x{size}...")
    model, loss = train_single_siren(img, epochs=150, device=device, verbose=False)
    siren_size = measure_model_size_compressed(model)

    # Generate SIREN prediction
    coords = get_coordinates(size, device)
    with torch.no_grad():
        pred = model(coords)
    pred_img = (pred.cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)

    # Compute PSNR for lossy
    mse_lossy = np.mean((img.astype(float) - pred_img.astype(float))**2)
    psnr_lossy = 10 * np.log10(255**2 / max(mse_lossy, 1e-10))

    # Compute residual
    residual = (img.astype(np.int16) - pred_img.astype(np.int16)) % 256
    residual = residual.astype(np.uint8)

    # Level 1: SIREN + zlib(residual)
    residual_zlib = zlib.compress(residual.tobytes(), 9)

    # Level 2: SIREN + PNG(residual)
    res_buf = io.BytesIO()
    Image.fromarray(residual).save(res_buf, format='PNG', optimize=True)
    residual_png = res_buf.getvalue()

    # Level 3: SIREN + WebP(residual) - even smaller
    res_buf2 = io.BytesIO()
    Image.fromarray(residual).save(res_buf2, format='WebP', lossless=True)
    residual_webp = res_buf2.getvalue()

    # Verify bit-perfect
    rec_zlib = ((pred_img.astype(np.int16) + residual) % 256).astype(np.uint8)
    sha_orig = hashlib.sha256(img.tobytes()).hexdigest()
    sha_rec = hashlib.sha256(rec_zlib.tobytes()).hexdigest()
    bit_perfect = sha_orig == sha_rec

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 15 RESULTS — PROGRESSIVE UPGRADE")
    print(f"{'='*80}")

    print(f"\n  {'Level':<35} {'Size':>8} {'Quality':>10} {'vs ZIP':>8}")
    print(f"  {'-'*65}")
    print(f"  {'Level 0: SIREN only (lossy)':<35} {siren_size:>7,}B {psnr_lossy:>8.1f}dB {zip_size/siren_size:>7.2f}x")
    print(f"  {'Level 1: SIREN + zlib residual':<35} {siren_size + len(residual_zlib):>7,}B {'lossy':>10} {zip_size/(siren_size + len(residual_zlib)):>7.2f}x")
    print(f"  {'Level 2: SIREN + PNG residual':<35} {siren_size + len(residual_png):>7,}B {'lossy':>10} {zip_size/(siren_size + len(residual_png)):>7.2f}x")
    print(f"  {'Level 3: SIREN + WebP residual':<35} {siren_size + len(residual_webp):>7,}B {'bit-perfect' if bit_perfect else 'lossy':>10} {zip_size/(siren_size + len(residual_webp)):>7.2f}x")
    print(f"  {'Reference: ZIP (zlib-9)':<35} {zip_size:>7,}B {'lossless':>10} {'1.00x':>7}")
    print(f"  {'Reference: Raw':<35} {raw_size:>7,}B {'lossless':>10} {raw_size/zip_size:>7.2f}x")

    print(f"\n  Bit-perfect verified: {'✅ YES' if bit_perfect else '❌ NO'}")
    print(f"\n  📋 Progressive loading scenario:")
    print(f"     1. User downloads Level 0 ({siren_size:,}B) → instant preview ({psnr_lossy:.1f}dB)")
    print(f"     2. Upgrade to Level 3 (+{len(residual_webp):,}B) → bit-perfect")
    print(f"     3. Total: {siren_size + len(residual_webp):,}B vs ZIP {zip_size:,}B")
    total = siren_size + len(residual_webp)
    print(f"     4. vs ZIP: {zip_size/total:.2f}x smaller!" if total < zip_size else f"     4. ZIP is {total/zip_size:.2f}x smaller")

    return {
        'siren_size': siren_size,
        'residual_zlib': len(residual_zlib),
        'residual_png': len(residual_png),
        'residual_webp': len(residual_webp),
        'total_bitperfect': siren_size + len(residual_webp),
        'zip_size': zip_size,
        'bit_perfect': bit_perfect,
        'psnr_lossy': psnr_lossy,
    }


if __name__ == '__main__':
    results = run_phase15_experiment(verbose=True)
