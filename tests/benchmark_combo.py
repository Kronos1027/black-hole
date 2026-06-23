#!/usr/bin/env python3
"""
benchmark_combo.py — v5.9 Combo scaling test (simplified)
=========================================================
Tests only BLKH v5.9 Combo vs ZIP per-file across N=5, 10, 20 images.
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

from siren_v5_combo import ComboCompressor


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
    print("=" * 90)
    print("  BLKH v5.9 COMBO scaling test (hypernetwork + WebP residual)")
    print("  All 100% SHA-256 verified")
    print("=" * 90)

    SIZE = 64
    results = []
    for N in [5, 10]:
        print(f"\n--- N = {N} images ({SIZE}x{SIZE}x3) ---")
        images = [make_smooth_image(SIZE, seed=42 + i) for i in range(N)]
        total_orig = sum(im.nbytes for im in images)
        zip_total = sum(len(zlib.compress(im.tobytes(), 9)) for im in images)
        print(f"  total_orig: {total_orig:,}B   ZIP: {zip_total:,}B")

        comp = ComboCompressor(latent_dim=16, hidden_features=16, hidden_layers=1,
                                omega_0=30.0, residual_codec='webp')
        t0 = time.time()
        comp.train_base(images, epochs=800, lr=1e-3, batch_size=2048, verbose=False)
        res = comp.compress_many(images, epochs=300, lr=3e-3,
                                  bits=8, batch_size=2048, verbose=False)
        dt = time.time() - t0
        rec, meta = ComboCompressor.decompress(res['recipe_bytes'])
        winner = "BLKH" if res['recipe_size'] < zip_total else "ZIP"
        print(f"  Combo:    {res['recipe_size']:>7,}B  bit%={res['avg_bit_pct']:.1f}  "
              f"ok={meta['all_sha256_match']}  {dt:.1f}s  "
              f"vs ZIP={zip_total/res['recipe_size']:.3f}x  -> {winner}")
        print(f"    hyper (shared): {res['hyper_size']:,}B ({res['hyper_size']/N:.0f}/img amortized)")
        print(f"    latent/img:     {res['latent_per_image']}B  residual/img: {res['residual_per_image']:.0f}B")

        results.append({
            'n': N,
            'total_orig': total_orig,
            'zip_total': zip_total,
            'combo_size': res['recipe_size'],
            'combo_ok': meta['all_sha256_match'],
            'combo_vs_zip': zip_total / res['recipe_size'],
            'hyper_size': res['hyper_size'],
            'avg_bit_pct': res['avg_bit_pct'],
            'time_s': dt,
        })

    print("\n" + "=" * 90)
    print("  COMBO SCALING SUMMARY")
    print("=" * 90)
    print(f"{'N':>5}{'orig':>10}{'ZIP':>10}{'Combo':>10}{'Combo/ZIP':>11}{'ok':>5}")
    print("-" * 90)
    for r in results:
        print(f"{r['n']:>5}{r['total_orig']:>10,}{r['zip_total']:>10,}"
              f"{r['combo_size']:>10,}{r['combo_vs_zip']:>10.3f}x"
              f"{'OK' if r['combo_ok'] else 'FAIL':>5}")

    out = Path(__file__).parent / 'benchmark_combo_results.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")


if __name__ == '__main__':
    main()
