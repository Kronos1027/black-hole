#!/usr/bin/env python3
"""
benchmark_lossy.py — BLKH v5 LOSSY mode vs JPEG vs WebP
=========================================================
The fair comparison for BLKH's lossy mode (no residual) is against
JPEG and WebP lossy. This benchmark runs all three at similar PSNR levels
and compares file sizes.

BLKH lossy mode uses:
  - INT4 quantization (4-bit weights)
  - Optional magnitude pruning
  - No residual layer (no XOR correction)
  - 1.5KB-2KB typical recipe size regardless of image size

This is the mode where BLKH might actually compete with JPEG/WebP!
"""
import sys
import os
import io
import time
import zlib
import json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
torch.set_num_threads(4)

from siren_v5_torch import ImageINRv5
from benchmark_real_photos import (
    make_sky_photo, make_wood_photo, make_water_photo, make_skin_photo, make_marble_photo
)


def jpeg_compress(img, quality=85):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def webp_compress_lossy(img, quality=85):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='WebP', lossless=False, quality=quality)
    return buf.getvalue()


def compute_psnr(orig, recon):
    mse = np.mean((orig.astype(np.float32) - recon.astype(np.float32)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf'))


def jpeg_psnr(img, quality=85):
    """Compress and decompress with JPEG, return (size, psnr)."""
    data = jpeg_compress(img, quality)
    from PIL import Image
    recon = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
    return len(data), compute_psnr(img, recon)


def webp_psnr(img, quality=85):
    data = webp_compress_lossy(img, quality)
    from PIL import Image
    recon = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
    return len(data), compute_psnr(img, recon)


def blkh_lossy(img, hidden=32, layers=2, epochs=1500, bits=4, prune=0.005):
    """Returns (size, psnr, time)."""
    comp = ImageINRv5(hidden_features=hidden, hidden_layers=layers, omega_0=30.0)
    t0 = time.time()
    res = comp.compress_lossy(img, epochs=epochs, lr=1e-3,
                                bits=bits, prune_threshold=prune,
                                batch_size=2048, verbose=False)
    dt = time.time() - t0
    return res['recipe_size'], float(res['psnr_db']), dt


def main():
    print("=" * 110)
    print("  BLKH v5 LOSSY vs JPEG vs WebP vs ZIP")
    print("  (all lossy formats — fair comparison)")
    print("=" * 110)

    photos = [
        ('Sky',    make_sky_photo(128, seed=1)),
        ('Wood',   make_wood_photo(128, seed=2)),
        ('Water',  make_water_photo(128, seed=3)),
        ('Skin',   make_skin_photo(128, seed=4)),
        ('Marble', make_marble_photo(128, seed=5)),
    ]

    results = []
    for name, img in photos:
        print(f"\n--- {name} ({img.nbytes:,}B) ---")
        orig = img.nbytes
        zip_size = len(zlib.compress(img.tobytes(), 9))
        print(f"  ZIP (lossless):    {zip_size:>7,}B  ({orig/zip_size:.2f}x)")

        jpeg_q85_sz, jpeg_q85_psnr = jpeg_psnr(img, 85)
        jpeg_q75_sz, jpeg_q75_psnr = jpeg_psnr(img, 75)
        webp_q85_sz, webp_q85_psnr = webp_psnr(img, 85)
        webp_q75_sz, webp_q75_psnr = webp_psnr(img, 75)

        print(f"  JPEG q=85:         {jpeg_q85_sz:>7,}B  PSNR={jpeg_q85_psnr:.1f}dB  ({orig/jpeg_q85_sz:.2f}x)")
        print(f"  JPEG q=75:         {jpeg_q75_sz:>7,}B  PSNR={jpeg_q75_psnr:.1f}dB  ({orig/jpeg_q75_sz:.2f}x)")
        print(f"  WebP q=85:         {webp_q85_sz:>7,}B  PSNR={webp_q85_psnr:.1f}dB  ({orig/webp_q85_sz:.2f}x)")
        print(f"  WebP q=75:         {webp_q75_sz:>7,}B  PSNR={webp_q75_psnr:.1f}dB  ({orig/webp_q75_sz:.2f}x)")

        # BLKH lossy at different quality levels
        for label, bits, prune, epochs in [
            ('BLKH INT4 aggressive', 4, 0.01, 1500),
            ('BLKH INT4 balanced',   4, 0.005, 1500),
            ('BLKH INT8 high-q',     8, 0.0, 2000),
        ]:
            sz, psnr, dt = blkh_lossy(img, bits=bits, prune=prune, epochs=epochs)
            winner = "BEST LOSSY" if sz < min(jpeg_q85_sz, webp_q85_sz) else ""
            print(f"  {label}:    {sz:>7,}B  PSNR={psnr:.1f}dB  ({orig/sz:.2f}x)  "
                  f"{dt:.1f}s  {winner}")

            results.append({
                'name': name,
                'orig_size': orig,
                'zip_size': zip_size,
                'mode': label,
                'blkh_size': sz,
                'blkh_psnr_db': round(psnr, 2),
                'blkh_time_s': round(dt, 2),
                'blkh_ratio': round(orig / sz, 2),
                'jpeg_q85_size': jpeg_q85_sz,
                'jpeg_q85_psnr': round(jpeg_q85_psnr, 2),
                'jpeg_q75_size': jpeg_q75_sz,
                'jpeg_q75_psnr': round(jpeg_q75_psnr, 2),
                'webp_q85_size': webp_q85_sz,
                'webp_q85_psnr': round(webp_q85_psnr, 2),
                'webp_q75_size': webp_q75_sz,
                'webp_q75_psnr': round(webp_q75_psnr, 2),
            })

    # Summary: best lossy size per photo
    print("\n" + "=" * 110)
    print("  LOSSY SUMMARY — Best size per photo (with PSNR for context)")
    print("=" * 110)
    print(f"{'photo':<10}{'ZIP':>10}{'JPEG q85':>14}{'WebP q85':>14}{'BLKH best':>20}{'winner':>12}")
    print("-" * 110)
    for name, _ in photos:
        photo_results = [r for r in results if r['name'] == name]
        if not photo_results:
            continue
        # Best BLKH = smallest size
        blkh_best = min(photo_results, key=lambda r: r['blkh_size'])
        zip_sz = blkh_best['zip_size']
        jpeg_sz = blkh_best['jpeg_q85_size']
        webp_sz = blkh_best['webp_q85_size']
        blkh_sz = blkh_best['blkh_size']
        competitors = {'ZIP': zip_sz, 'JPEG': jpeg_sz, 'WebP': webp_sz, 'BLKH': blkh_sz}
        winner = min(competitors, key=competitors.get)
        print(f"{name:<10}{zip_sz:>10,}"
              f"{jpeg_sz:>8,}B@{blkh_best['jpeg_q85_psnr']:.0f}dB"
              f"{webp_sz:>8,}B@{blkh_best['webp_q85_psnr']:.0f}dB"
              f"{blkh_sz:>10,}B@{blkh_best['blkh_psnr_db']:.0f}dB"
              f"{winner:>12}")

    out = Path(__file__).parent / 'benchmark_lossy_results.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")


if __name__ == '__main__':
    main()
