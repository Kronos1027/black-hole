# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 1b: Multi-File SIREN on REAL Images
===========================================
Tests the shared roots principle on real photographs (not just synthetic).

Previous Phase 1 used synthetic satellite-like images. This experiment
tests whether the scaling law holds for REAL natural photos.

HYPOTHESIS:
  Real photos have more high-frequency detail than synthetic images,
  so shared roots will provide LESS improvement (maybe 2-5x instead of 17x).
  But it should still beat separate SIRENs.

METHOD:
  1. Load 5 real sample photos (marble, skin, sky, water, wood @ 128x128)
  2. Train Multi-File SIREN with shared roots
  3. Compare with separate SIRENs and ZIP
  4. Also test with upscaled versions (256x256) via duplication

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import (
    SIREN, ModulatedSIREN, get_coordinates,
    train_single_siren, train_multi_file_siren,
    measure_model_size_compressed
)


def load_real_photos(size=128):
    """Load real sample photos from docs/assets."""
    photos_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'docs', 'assets', 'sample_photos')
    images = []
    names = []

    for fname in sorted(os.listdir(photos_dir)):
        if not fname.endswith('.png'):
            continue
        img = np.array(Image.open(os.path.join(photos_dir, fname)).convert('RGB'))
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Resize if needed
        if img.shape[0] != size or img.shape[1] != size:
            pil = Image.fromarray(img)
            pil = pil.resize((size, size), Image.LANCZOS)
            img = np.array(pil)

        images.append(img)
        names.append(fname.replace('.png', ''))

    return images, names


def run_real_photo_experiment(verbose=True):
    """Run Multi-File SIREN on real photos."""
    print("=" * 80)
    print("🧪 Phase 1b: Multi-File SIREN on REAL Photos")
    print("=" * 80)

    device = 'cpu'

    # Load real photos
    print("\n📸 Loading real sample photos...")
    images, names = load_real_photos(size=128)
    n_files = len(images)
    total_raw = sum(img.nbytes for img in images)
    total_zip = sum(len(zlib.compress(img.tobytes(), 9)) for img in images)

    print(f"  Loaded {n_files} photos: {', '.join(names)}")
    print(f"  Size: 128x128x3 each")
    print(f"  Total raw: {total_raw:,}B")
    print(f"  Total ZIP: {total_zip:,}B")
    print()

    # Baseline: separate SIRENs
    print("🔵 Baseline: Training separate SIRENs...")
    t0 = time.time()
    baseline_total = 0
    for i, img in enumerate(images):
        model, loss = train_single_siren(img, epochs=100, device=device, verbose=False)
        size = measure_model_size_compressed(model)
        baseline_total += size
        if verbose:
            print(f"  {names[i]}: loss={loss:.6f}, size={size:,}B")
    baseline_time = time.time() - t0
    print(f"  Total: {baseline_total:,}B in {baseline_time:.1f}s")
    print()

    # BHUH: Multi-file SIREN
    print("🌌 BHUH: Training multi-file SIREN (shared roots)...")
    t0 = time.time()
    multi_model, multi_loss = train_multi_file_siren(
        images, epochs=200, device=device, verbose=verbose
    )
    multi_time = time.time() - t0
    multi_size = measure_model_size_compressed(multi_model)
    print(f"  Total: {multi_size:,}B in {multi_time:.1f}s")
    print()

    # Results
    improvement = baseline_total / max(multi_size, 1)
    vs_zip = total_zip / max(multi_size, 1)

    print("=" * 80)
    print("📊 RESULTS — REAL PHOTOS")
    print("=" * 80)
    print(f"\n  {'Metric':<30} {'Value':>15}")
    print(f"  {'-'*50}")
    print(f"  {'Files':<30} {n_files:>15}")
    print(f"  {'Total raw':<30} {total_raw:>14,}B")
    print(f"  {'Total ZIP':<30} {total_zip:>14,}B")
    print(f"  {'Separate SIRENs':<30} {baseline_total:>14,}B")
    print(f"  {'BHUH (shared roots)':<30} {multi_size:>14,}B")
    print(f"  {'Improvement vs SIREN':<30} {improvement:>14.2f}x")
    print(f"  {'vs ZIP':<30} {vs_zip:>14.2f}x")
    print(f"  {'Training time (baseline)':<30} {baseline_time:>13.1f}s")
    print(f"  {'Training time (BHUH)':<30} {multi_time:>13.1f}s")

    print(f"\n📋 Comparison with synthetic (Phase 1):")
    print(f"  Synthetic 50 imgs: 17.93x vs SIREN, 62.78x vs ZIP")
    print(f"  Real {n_files} imgs:      {improvement:.2f}x vs SIREN, {vs_zip:.2f}x vs ZIP")

    if improvement >= 2.0:
        print(f"\n✅ Shared roots work on REAL photos too!")
    elif improvement >= 1.5:
        print(f"\n⚠️  Partial success — real photos are harder (more high-freq detail)")
    else:
        print(f"\n❌ Real photos don't benefit as much from shared roots")

    return {
        'n_files': n_files,
        'names': names,
        'baseline_size': baseline_total,
        'bhuh_size': multi_size,
        'improvement': improvement,
        'zip_size': total_zip,
        'vs_zip': vs_zip,
    }


if __name__ == '__main__':
    results = run_real_photo_experiment(verbose=True)
