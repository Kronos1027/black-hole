#!/usr/bin/env python3
"""
benchmark_atlas.py — Atlas scaling test: 5/10/20/50 similar images
====================================================================
Proves that BLKH Atlas beats ZIP per-file at scale, with growing advantage
as N increases (because weights are amortized).
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

from siren_v5_atlas import AtlasCompressor


def make_smooth_image(size, seed):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        cy, cx = rng.uniform(size * 0.2, size * 0.8, 2)
        sigma = rng.uniform(size * 0.1, size * 0.25)
        amp = rng.uniform(80, 200)
        img[:, :, c] = amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    print("=" * 95)
    print("  BLKH Neural Atlas — scaling test (bit-perfect, SHA-256 verified)")
    print(f"  PyTorch {torch.__version__} | device: {('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 95)

    results = []
    SIZE = 64  # keep image size constant, vary N
    EPOCHS = 1000

    for N in [5, 10, 20, 50]:
        print(f"\n--- N = {N} images ({SIZE}x{SIZE}x3) ---")
        images = [make_smooth_image(SIZE, seed=42 + i) for i in range(N)]
        total_orig = sum(im.nbytes for im in images)
        zip_total = sum(len(zlib.compress(im.tobytes(), 9)) for im in images)
        print(f"  total_orig: {total_orig:,}B   ZIP per-file: {zip_total:,}B "
              f"(ratio {total_orig/zip_total:.2f}x)")

        comp = AtlasCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
        t0 = time.time()
        try:
            res = comp.compress(images, epochs=EPOCHS, lr=1e-3, bits=8,
                                 batch_size=8192, verbose=False)
            dt = time.time() - t0
            rec, meta = AtlasCompressor.decompress(res['recipe_bytes'])
            ok = meta['all_sha256_match']
            winner = "BLKH" if res['recipe_size'] < zip_total else "ZIP"
            amort_w = res['weights_packed_size'] / N
            print(f"  ATLAS: {res['recipe_size']:,}B  (ratio {total_orig/res['recipe_size']:.2f}x)")
            print(f"    weights: {res['weights_packed_size']:,}B ({amort_w:.0f}/image amortized)")
            print(f"    residual: {res['residual_total']:,}B ({res['residual_per_image']:.0f}/image)")
            print(f"    bit acc: {res['avg_bit_pct']:.1f}%   SHA-256: {sum(meta['sha256_per_image'])}/{N}")
            print(f"    train: {res['train_time_s']:.1f}s   total: {dt:.1f}s")
            print(f"  vs ZIP: {zip_total/res['recipe_size']:.3f}x  -> {winner}")

            results.append({
                'n': N,
                'total_orig': total_orig,
                'zip_total': zip_total,
                'atlas_total': res['recipe_size'],
                'atlas_weights': res['weights_packed_size'],
                'atlas_residual': res['residual_total'],
                'amortized_weight_per_image': amort_w,
                'avg_bit_pct': res['avg_bit_pct'],
                'all_sha256_match': ok,
                'train_time_s': res['train_time_s'],
                'total_time_s': dt,
                'winner': winner,
                'atlas_vs_zip': zip_total / res['recipe_size'],
                'atlas_ratio': total_orig / res['recipe_size'],
                'zip_ratio': total_orig / zip_total,
            })
        except Exception as e:
            import traceback; traceback.print_exc()

    print("\n" + "=" * 95)
    print("  ATLAS SCALING SUMMARY")
    print("=" * 95)
    print(f"{'N':>5}{'orig':>10}{'ZIP':>10}{'ATLAS':>10}{'ratio':>8}{'vs ZIP':>10}{'ok':>5}")
    print("-" * 95)
    for r in results:
        print(f"{r['n']:>5}{r['total_orig']:>10,}{r['zip_total']:>10,}"
              f"{r['atlas_total']:>10,}{r['atlas_ratio']:>7.2f}x"
              f"{r['atlas_vs_zip']:>9.3f}x{'OK' if r['all_sha256_match'] else 'FAIL':>5}")

    out = Path(__file__).parent / 'benchmark_atlas_results.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")


if __name__ == '__main__':
    main()
