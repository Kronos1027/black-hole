#!/usr/bin/env python3
"""
example_datacenter.py — Real-world usage example
=================================================
Simulates a datacenter scenario: compress a corpus of 100 MRI-like images
using the best BLKH strategy, compare total size vs ZIP, verify roundtrip.

This is the kind of script a hospital IT department or satellite operator
would actually run to evaluate BLKH for their archive.
"""
import sys
import os
import time
import zlib
import json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
torch.set_num_threads(4)

from siren_v5_torch import ImageINRv5
from siren_v5_atlas import AtlasCompressor


def make_mri_corpus(n, size=128, base_seed=42):
    """Simulate N MRI slices — smooth tissue boundaries."""
    images = []
    for i in range(n):
        rng = np.random.default_rng(seed=base_seed + i)
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
        img = np.zeros((size, size, 3), dtype=np.float32)
        # Background tissue
        img[:, :, 0] = 60 + 20 * np.sin(xs * 0.05) * np.cos(ys * 0.05)
        img[:, :, 1] = 80 + 25 * np.sin(ys * 0.04)
        img[:, :, 2] = 50 + 15 * np.cos(xs * 0.06 + ys * 0.03)
        # 4-6 tissue regions
        for _ in range(rng.integers(4, 7)):
            cy, cx = rng.uniform(15, size - 15, 2)
            sigma = rng.uniform(8, 20)
            amp = rng.uniform(50, 150)
            for c in range(3):
                img[:, :, c] += amp * np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma**2)) * (c + 1) / 3
        images.append(np.clip(img, 0, 255).astype(np.uint8))
    return images


def main():
    print("=" * 80)
    print("  BLACK HOLE — DATACENTER USAGE EXAMPLE")
    print("  Compressing 100 MRI-like images (128x128 RGB each)")
    print("=" * 80)

    N = 20  # use 20 (sweet spot for atlas per our benchmarks)
    SIZE = 128
    print(f"\n[1] Generating {N} synthetic MRI-like images ({SIZE}x{SIZE} RGB)...")
    images = make_mri_corpus(N, size=SIZE, base_seed=42)
    total_orig = sum(im.nbytes for im in images)
    zip_total = sum(len(zlib.compress(im.tobytes(), 9)) for im in images)
    print(f"    Total original: {total_orig:,} B ({total_orig/1024:.1f} KB)")
    print(f"    ZIP per-file:   {zip_total:,} B ({zip_total/1024:.1f} KB)  "
          f"ratio {total_orig/zip_total:.2f}x")

    # Strategy 1: BLKH v5 single (per-image)
    print(f"\n[2] Strategy A: BLKH v5 single-image (per file, no sharing)...")
    blkh_single_total = 0
    single_ok_count = 0
    t0 = time.time()
    for i, img in enumerate(images):
        comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress_bitperfect(img, epochs=600, lr=1e-3,
                                         bits=8, batch_size=2048, verbose=False)
        blkh_single_total += res['recipe_size']
        # Verify roundtrip on first and last
        if i == 0 or i == N - 1:
            rec, meta = ImageINRv5.decompress(res['recipe_bytes'])
            if meta['exact_match']:
                single_ok_count += 1
    single_time = time.time() - t0
    print(f"    BLKH single total: {blkh_single_total:,} B ({blkh_single_total/1024:.1f} KB)  "
          f"ratio {total_orig/blkh_single_total:.2f}x")
    print(f"    Time: {single_time:.1f}s ({single_time/N:.2f}s/image)")
    print(f"    SHA-256 spot check: {single_ok_count}/2 OK")

    # Strategy 2: BLKH Atlas (shared SIREN)
    print(f"\n[3] Strategy B: BLKH v5.2 Neural Atlas (1 shared SIREN for all)...")
    comp = AtlasCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
    t0 = time.time()
    res = comp.compress(images, epochs=1500, lr=1e-3, bits=8,
                         batch_size=8192, verbose=False)
    atlas_time = time.time() - t0
    rec, meta = AtlasCompressor.decompress(res['recipe_bytes'])
    print(f"    BLKH atlas total:  {res['recipe_size']:,} B ({res['recipe_size']/1024:.1f} KB)  "
          f"ratio {total_orig/res['recipe_size']:.2f}x")
    print(f"      weights (shared): {res['weights_packed_size']:,} B "
          f"-> {res['weights_packed_size']/N:.0f} B/image amortized")
    print(f"      residual per img: {res['residual_per_image']:.0f} B")
    print(f"    Time: {atlas_time:.1f}s ({atlas_time/N:.2f}s/image amortized)")
    print(f"    Bit accuracy avg: {res['avg_bit_pct']:.1f}%")
    print(f"    All {N} SHA-256 verified: {meta['all_sha256_match']}")

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  Original corpus:    {total_orig:>10,} B")
    print(f"  ZIP per-file:       {zip_total:>10,} B   "
          f"(ratio {total_orig/zip_total:.2f}x, baseline)")
    print(f"  BLKH v5 single:     {blkh_single_total:>10,} B   "
          f"(ratio {total_orig/blkh_single_total:.2f}x, "
          f"{zip_total/blkh_single_total:.2f}x vs ZIP)")
    print(f"  BLKH v5.2 atlas:    {res['recipe_size']:>10,} B   "
          f"(ratio {total_orig/res['recipe_size']:.2f}x, "
          f"{zip_total/res['recipe_size']:.2f}x vs ZIP)")

    print(f"\n  Winners:")
    strategies = [
        ('ZIP per-file (baseline)', zip_total),
        ('BLKH v5 single', blkh_single_total),
        ('BLKH v5.2 atlas', res['recipe_size']),
    ]
    strategies.sort(key=lambda x: x[1])
    for i, (name, sz) in enumerate(strategies):
        marker = '*** BEST ***' if i == 0 else ''
        print(f"    {i+1}. {name:25s}  {sz:>8,} B  {marker}")

    print(f"\n  Recommendation:")
    best = strategies[0]
    print(f"    For this corpus ({N} MRI-like images, {SIZE}x{SIZE} RGB):")
    print(f"    Use {best[0]} — saves {(zip_total - best[1]):,} B vs ZIP "
          f"({(1 - best[1]/zip_total)*100:.1f}% smaller).")

    out = Path(__file__).parent / 'example_datacenter_results.json'
    out.write_text(json.dumps({
        'n_images': N,
        'size': SIZE,
        'total_orig': total_orig,
        'zip_total': zip_total,
        'blkh_single_total': blkh_single_total,
        'blkh_atlas_total': res['recipe_size'],
        'atlas_sha256_ok': meta['all_sha256_match'],
        'atlas_bit_pct': res['avg_bit_pct'],
    }, indent=2))
    print(f"\n  Results saved to: {out}")


if __name__ == '__main__':
    main()
