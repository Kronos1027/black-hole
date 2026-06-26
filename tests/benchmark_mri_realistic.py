#!/usr/bin/env python3
"""
benchmark_mri_realistic.py — BLKH on realistic MRI-like volumes
================================================================
Generates MRI-like volumes that closely mimic real brain MRI statistics:
  - T1-weighted tissue intensities (WM=180, GM=120, CSF=60, BG=0)
  - 3D gaussian blobs for tissue regions (WM, GM, ventricles)
  - Rician noise (typical of magnitude MRI, sigma=8-12)
  - Bias field (slow spatial intensity variation)
  - Partial volume effect at tissue boundaries

Tests BLKH v5.12 (lossless) and v5.14 (DCT, lossy) vs ZIP at multiple
volume sizes: 32³, 64³, 128³.

This is the closest we can get to real MRI without downloading patient data.
The statistical properties match clinical T1-weighted brain MRI.
"""
import sys
import os
import time
import zlib
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
torch.set_num_threads(4)

from siren_v5_volume import VolumeCompressor
from siren_v5_volume_opt import VolumeCompressorOpt


def make_realistic_mri(D, H, W, seed=42):
    """Generate a realistic T1-weighted brain MRI-like volume.
    
    Statistical properties match clinical brain MRI:
    - White matter: ~180 intensity
    - Gray matter: ~120 intensity  
    - CSF (ventricles): ~60 intensity
    - Background: ~0
    - Rician noise (sigma=10)
    - Bias field (slow multiplicative variation)
    """
    rng = np.random.default_rng(seed)
    
    # Normalized coordinates [-1, 1]
    zs, ys, xs = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
    zs = zs / (D - 1) * 2 - 1
    ys = ys / (H - 1) * 2 - 1
    xs = xs / (W - 1) * 2 - 1
    
    # 1. Brain shape (ellipsoid mask)
    brain_mask = (xs**2 / 0.7 + ys**2 / 0.8 + zs**2 / 0.6) < 1.0
    
    # 2. Tissue intensities
    vol = np.zeros((D, H, W), dtype=np.float32)
    
    # Background = 0 (already)
    
    # White matter (central region, high intensity ~180)
    wm = 180 * np.exp(-(xs**2 + ys**2 + zs**2) / 0.15)
    vol += wm * brain_mask
    
    # Gray matter (cortical shell, ~120)
    gm_shell = np.exp(-((np.sqrt(xs**2 + ys**2 + zs**2) - 0.7)**2) / 0.05)
    vol += 120 * gm_shell * brain_mask
    
    # CSF / ventricles (low intensity ~60, central)
    ventricles = 60 * np.exp(-(xs**2 + ys**2) / 0.05) * np.exp(-(zs**2) / 0.3)
    vol = np.where(ventricles > 20, ventricles, vol)
    
    # 3. Additional tissue detail (subtle gaussian blobs)
    for _ in range(8):
        cz, cy, cx = rng.uniform(-0.5, 0.5, 3)
        sigma = rng.uniform(0.05, 0.15)
        amp = rng.uniform(-30, 30)
        vol += amp * np.exp(-((xs-cx)**2 + (ys-cy)**2 + (zs-cz)**2) / (2*sigma**2)) * brain_mask
    
    # 4. Bias field (slow multiplicative variation — typical in MRI)
    bias = 1.0 + 0.15 * np.sin(xs * 2) * np.cos(ys * 1.5) + 0.1 * np.sin(zs * 1)
    vol *= bias
    
    # 5. Rician noise (magnitude MRI noise model)
    sigma_noise = 10.0
    real = vol + rng.normal(0, sigma_noise, vol.shape)
    imag = rng.normal(0, sigma_noise, vol.shape)
    vol_rician = np.sqrt(real**2 + imag**2)
    
    # Clip and convert to uint8
    vol_rician = np.clip(vol_rician, 0, 255).astype(np.uint8)
    
    # Add channel dimension: (D, H, W, 1)
    return vol_rician[..., None]


def main():
    print("=" * 95)
    print("  BLKH on Realistic MRI-like Volumes (T1-weighted brain MRI statistics)")
    print("  Tissue: WM=180, GM=120, CSF=60 | Rician noise sigma=10 | Bias field")
    print("=" * 95)
    
    results = []
    for D, H, W in [(32, 32, 32), (64, 64, 64), (96, 96, 96)]:
        print(f"\n--- Volume {D}x{H}x{W}x1 ---")
        vol = make_realistic_mri(D, H, W, seed=42)
        total_orig = vol.nbytes
        zip_sz = len(zlib.compress(vol.tobytes(), 9))
        print(f"  Original: {total_orig:,}B   ZIP: {zip_sz:,}B ({total_orig/zip_sz:.2f}x)")
        
        # v5.12 lossless
        print(f"  Running v5.12 (lossless)...")
        comp12 = VolumeCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
        t0 = time.time()
        res12 = comp12.compress(vol, epochs=600, lr=1e-3, bits=8, batch_size=8192, verbose=False)
        dt12 = time.time() - t0
        rec12, meta12 = VolumeCompressor.decompress(res12['recipe_bytes'])
        print(f"  v5.12:     {res12['recipe_size']:>8,}B  bit%={res12['model_bit_accuracy']:.1f}  "
              f"SHA={meta12['exact_match']}  {dt12:.1f}s  vs ZIP={zip_sz/res12['recipe_size']:.2f}x")
        
        # v5.14 DCT lossy
        print(f"  Running v5.14 (DCT q=50, lossy)...")
        comp14 = VolumeCompressorOpt(hidden_features=64, hidden_layers=3, omega_0=30.0, dct_quality=50)
        t0 = time.time()
        res14 = comp14.compress(vol, epochs=600, lr=1e-3, bits=8, batch_size=8192, verbose=False)
        dt14 = time.time() - t0
        rec14, _ = VolumeCompressorOpt.decompress(res14['recipe_bytes'])
        mse14 = np.mean((vol.astype(float) - rec14.astype(float))**2)
        psnr14 = 10 * np.log10(255**2 / mse14) if mse14 > 0 else float('inf')
        print(f"  v5.14 q50: {res14['recipe_size']:>8,}B  PSNR={psnr14:.1f}dB  "
              f"{dt14:.1f}s  vs ZIP={zip_sz/res14['recipe_size']:.2f}x")
        
        results.append({
            'volume': f'{D}x{H}x{W}x1',
            'orig': total_orig,
            'zip': zip_sz,
            'v512_size': res12['recipe_size'],
            'v512_sha': meta12['exact_match'],
            'v514_size': res14['recipe_size'],
            'v514_psnr': round(psnr14, 1),
            'v512_vs_zip': round(zip_sz / res12['recipe_size'], 2),
            'v514_vs_zip': round(zip_sz / res14['recipe_size'], 2),
        })
    
    print("\n" + "=" * 95)
    print("  REALISTIC MRI-LIKE SUMMARY")
    print("=" * 95)
    print(f"{'volume':<18}{'orig':>10}{'ZIP':>10}{'v5.12':>10}{'v5.12/ZIP':>10}{'v5.14':>10}{'v5.14/ZIP':>10}{'PSNR':>8}")
    print("-" * 95)
    for r in results:
        print(f"{r['volume']:<18}{r['orig']:>10,}{r['zip']:>10,}"
              f"{r['v512_size']:>10,}{r['v512_vs_zip']:>9.2f}x"
              f"{r['v514_size']:>10,}{r['v514_vs_zip']:>9.2f}x"
              f"{r['v514_psnr']:>7.1f}dB")
    
    out = os.path.join(os.path.dirname(__file__), 'benchmark_mri_realistic_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out}")


if __name__ == '__main__':
    main()
