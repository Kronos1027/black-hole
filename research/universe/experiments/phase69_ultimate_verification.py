# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 69: Ultimate Verification (Re-Verify ALL Key Claims)
============================================================
Independently re-verifies every major BHUH claim with fresh experiments.

This is the scientific integrity check — running key experiments again
to confirm results are reproducible and not artifacts.

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, zlib, io
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))


def run_phase69_experiment(verbose=True):
    print("=" * 80)
    print("🧪 Phase 69: Ultimate Verification (Re-Verify ALL Key Claims)")
    print("=" * 80)

    passed = 0
    failed = 0
    results = []

    def verify(name, condition, details=""):
        nonlocal passed, failed
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"  {status}: {name} {details}")
        results.append({'name': name, 'passed': condition, 'details': details})
        if condition:
            passed += 1
        else:
            failed += 1

    # === CLAIM 1: Shared Roots (Phase 1) ===
    print("\n📋 Claim 1: Shared Roots (Multi-File SIREN)")
    from phase1_multi_file_siren import run_experiment
    r = run_experiment(n_images=10, size=64, epochs_single=30, epochs_multi=60, verbose=False)
    verify("Multi-File improvement > 1.5x", r['improvement'] > 1.5, f"(got {r['improvement']:.2f}x)")
    verify("BHUH smaller than separate", r['bhuh_size'] < r['baseline_size'], f"({r['bhuh_size']:,}B < {r['baseline_size']:,}B)")
    verify("BHUH beats ZIP", r['bhuh_size'] < r['zip_size'], f"({r['bhuh_size']:,}B < {r['zip_size']:,}B)")

    # === CLAIM 2: RLE on Real Photos (Phase 24) ===
    print("\n📋 Claim 2: RLE Beats ZIP on Real Photos")
    img = np.array(Image.open(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'docs', 'assets', 'sample_photos', 'sky_128.png')).convert('RGB'))
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    from siren_v5_rle import RLEDCTCompressor
    comp = RLEDCTCompressor(quality=0.9, speed='balanced')
    res = comp.compress(img, verbose=False)
    rec, _ = RLEDCTCompressor.decompress(res['recipe_bytes'])
    mse = np.mean((img.astype(float) - rec.astype(float))**2)
    psnr = 10*np.log10(255**2 / max(mse, 1e-10))
    verify("RLE < ZIP on real photo", res['recipe_size'] < zip_sz, f"({res['recipe_size']:,}B < {zip_sz:,}B)")
    verify("RLE PSNR > 25dB", psnr > 25, f"(got {psnr:.1f}dB)")

    # === CLAIM 3: Cross-Domain (Phase 6) ===
    print("\n📋 Claim 3: Cross-Domain Transfer")
    from phase6_cross_domain import run_phase6_experiment
    r6 = run_phase6_experiment(verbose=False)
    verify("Cross-domain < separate", r6['cross_domain'] < r6['baseline_separate'], f"({r6['cross_domain']:,}B < {r6['baseline_separate']:,}B)")
    verify("Cross-domain < domain-specific", r6['cross_domain'] < r6['domain_specific'], f"({r6['cross_domain']:,}B < {r6['domain_specific']:,}B)")

    # === CLAIM 4: BHUH Equation |s|=O(1) (Phase 21/43) ===
    print("\n📋 Claim 4: BHUH Equation (seed size constant)")
    from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren, measure_model_size_compressed
    seed_sizes = []
    for size in [64, 128, 256]:
        rng = np.random.default_rng(42)
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
        img_t = np.zeros((size, size, 3), dtype=np.float32)
        for c in range(3):
            for _ in range(3):
                kx, ky = rng.integers(2, 5, 2)
                img_t[:, :, c] += 50 * np.sin(2*np.pi*kx*xs) * np.cos(2*np.pi*ky*ys)
        img_t = ((img_t - img_t.min()) / (img_t.max() - img_t.min()) * 255).astype(np.uint8)
        model, _ = train_single_siren(img_t, epochs=40, device='cpu', verbose=False)
        seed_sizes.append(measure_model_size_compressed(model))
    cv = np.std(seed_sizes) / np.mean(seed_sizes)
    verify("Seed size CV < 0.1 (constant)", cv < 0.1, f"(CV={cv:.4f}, sizes={[f'{s:,}B' for s in seed_sizes]})")

    # === CLAIM 5: Denoising (Phase 29) ===
    print("\n📋 Claim 5: SIREN Denoising")
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:128, 0:128].astype(np.float32) / 128
    clean = np.zeros((128, 128, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(2, 5, 2)
            clean[:, :, c] += 50 * np.sin(2*np.pi*kx*xs) * np.cos(2*np.pi*ky*ys)
    clean = ((clean - clean.min()) / (clean.max() - clean.min()) * 255).astype(np.uint8)
    noisy = np.clip(clean.astype(float) + np.random.normal(0, 30, clean.shape), 0, 255).astype(np.uint8)
    mse_noisy = np.mean((clean.astype(float) - noisy.astype(float))**2)
    psnr_noisy = 10*np.log10(255**2 / max(mse_noisy, 1e-10))
    model, _ = train_single_siren(noisy, epochs=60, device='cpu', verbose=False)
    coords = get_coordinates(128, 'cpu')
    with torch.no_grad():
        pred = model(coords)
    denoised = (pred.cpu().numpy().reshape(128, 128, 3) * 255).clip(0, 255).astype(np.uint8)
    mse_den = np.mean((clean.astype(float) - denoised.astype(float))**2)
    psnr_den = 10*np.log10(255**2 / max(mse_den, 1e-10))
    improvement = psnr_den - psnr_noisy
    verify("Denoising improvement > 3dB", improvement > 3, f"(got +{improvement:.1f}dB)")

    # === CLAIM 6: Production Tests ===
    print("\n📋 Claim 6: Production Tests (165/165)")
    import subprocess
    result = subprocess.run(['python', '-m', 'pytest', 'tests/', '-q', '--tb=no'],
                          capture_output=True, text=True, timeout=120, cwd=os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    test_output = result.stdout
    verify("165 tests pass", '165 passed' in test_output, f"({test_output.strip().split(chr(10))[-2] if test_output else 'no output'})")

    # === SUMMARY ===
    print(f"\n{'='*80}")
    print("📊 PHASE 69 SUMMARY — ULTIMATE VERIFICATION")
    print(f"{'='*80}")
    print(f"\n  Total claims verified: {passed + failed}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  Success rate: {passed/(passed+failed)*100:.0f}%")

    if failed == 0:
        print(f"\n  🎉 ALL CLAIMS VERIFIED! BHUH is scientifically sound!")
    else:
        print(f"\n  ⚠️  {failed} claims failed verification — needs investigation")

    return {'passed': passed, 'failed': failed, 'results': results}


if __name__ == '__main__':

    results = run_phase69_experiment(verbose=True)
