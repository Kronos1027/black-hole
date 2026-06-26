#!/usr/bin/env python3
"""
benchmark_large.py — BLKH v5 on larger images (256x256, 512x512)
=================================================================
The key hypothesis: as image size grows, BLKH advantage over ZIP grows
because the SIREN recipe is roughly fixed-size while ZIP grows linearly
with content.
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


def make_smooth_image(size, seed=42):
    """Smooth 2D image — case where SIREN beats ZIP."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        cy, cx = rng.uniform(size * 0.2, size * 0.8, 2)
        sigma = rng.uniform(size * 0.1, size * 0.25)
        amp = rng.uniform(80, 200)
        img[:, :, c] = amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_pure_gradient(size):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            img[i, j] = [int(i * 255 / size), int(j * 255 / size), int((i + j) * 255 / (2 * size))]
    return img


def main():
    print("=" * 95)
    print("  BLKH v5 — Large Image Benchmark (scaling test)")
    print(f"  PyTorch {torch.__version__} | device: {('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 95)

    sizes = [64, 128, 256, 512]
    results = []

    for size in sizes:
        print(f"\n--- size {size}x{size} ---")
        # Test both gradient and blobs
        for name, img in [
            (f'gradient_{size}', make_pure_gradient(size)),
            (f'blobs_{size}', make_smooth_image(size, seed=42)),
        ]:
            orig = img.nbytes
            zip_size = len(zlib.compress(img.tobytes(), 9))
            print(f"  {name}: orig={orig:,}B  ZIP={zip_size:,}B ({orig/zip_size:.2f}x)")

            # Use bigger network for bigger images
            if size >= 256:
                hidden, layers = 64, 3
                epochs = 1500
            else:
                hidden, layers = 32, 2
                epochs = 1000

            comp = ImageINRv5(hidden_features=hidden, hidden_layers=layers, omega_0=30.0)
            t0 = time.time()
            try:
                res = comp.compress_bitperfect(img, epochs=epochs, lr=1e-3,
                                                 bits=8, batch_size=4096, verbose=False)
                dt = time.time() - t0
                recon, meta = ImageINRv5.decompress(res['recipe_bytes'])
                winner = "BLKH" if res['recipe_size'] < zip_size else "ZIP"
                print(f"    BLKH: {res['recipe_size']:,}B ({orig/res['recipe_size']:.2f}x)  "
                      f"bit%={res['model_bit_accuracy']:.1f}  "
                      f"PSNR={res['psnr_db']:.1f}dB  "
                      f"SHA={'OK' if meta['exact_match'] else 'FAIL'}  "
                      f"{dt:.1f}s  -> {winner}")
                results.append({
                    'name': name,
                    'size': size,
                    'orig_size': orig,
                    'zip_size': zip_size,
                    'blkh_size': res['recipe_size'],
                    'weights_size': res['weights_packed_size'],
                    'residual_size': res['residual_compressed_size'],
                    'bit_pct': round(res['model_bit_accuracy'], 2),
                    'psnr_db': round(res['psnr_db'], 2),
                    'sha256_ok': meta['exact_match'],
                    'time_s': round(dt, 2),
                    'winner': winner,
                    'blkh_vs_zip': round(zip_size / res['recipe_size'], 3),
                })
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback; traceback.print_exc()

    print("\n" + "=" * 95)
    print("  LARGE IMAGE SUMMARY")
    print("=" * 95)
    print(f"{'name':<20}{'orig':>10}{'zip':>10}{'blkh':>10}{'bit%':>8}{'vs zip':>10}{'win':>8}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<20}{r['orig_size']:>10,}{r['zip_size']:>10,}"
              f"{r['blkh_size']:>10,}{r['bit_pct']:>7.1f}%{r['blkh_vs_zip']:>9.3f}x"
              f"{r['winner']:>8}")

    out = Path(__file__).parent / 'benchmark_large_results.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")


if __name__ == '__main__':
    main()
