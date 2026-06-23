#!/usr/bin/env python3
"""
benchmark_vs_image_formats.py — BLKH v5 vs PNG vs JPEG vs WebP vs ZIP
======================================================================
The previous benchmarks only compared BLKH to ZIP. But for image data, the
real competitors are PNG (lossless standard), JPEG (lossy standard), and
WebP (modern, both modes). This benchmark compares BLKH against all of them.

Note: BLKH is BIT-PERFECT (lossless). Fair comparison:
  - PNG (lossless)
  - WebP lossless
  - ZIP on raw bytes (lossless)
  - BLKH v5 bit-perfect (lossless)
And for context (NOT fair, BLKH is lossless):
  - JPEG q=85 (lossy)
  - WebP q=85 (lossy)
"""
import sys
import os
import io
import time
import zlib
import json
import tempfile
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
torch.set_num_threads(4)

from siren_v5_torch import ImageINRv5
from benchmark_real_photos import (
    make_sky_photo, make_wood_photo, make_water_photo, make_skin_photo, make_marble_photo
)


def png_size(img):
    """Lossless PNG size."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='PNG', optimize=True)
    return len(buf.getvalue())


def jpeg_size(img, quality=85):
    """Lossy JPEG size at given quality."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=quality)
    return len(buf.getvalue())


def webp_lossless_size(img):
    """Lossless WebP size."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='WebP', lossless=True)
    return len(buf.getvalue())


def webp_lossy_size(img, quality=85):
    """Lossy WebP size at given quality."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='WebP', lossless=False, quality=quality)
    return len(buf.getvalue())


def zip_size(img):
    """ZIP (zlib) on raw bytes."""
    return len(zlib.compress(img.tobytes(), 9))


def blkh_size(img, hidden=32, layers=2, epochs=1500):
    """BLKH v5 bit-perfect recipe size."""
    comp = ImageINRv5(hidden_features=hidden, hidden_layers=layers, omega_0=30.0)
    res = comp.compress_bitperfect(img, epochs=epochs, lr=1e-3,
                                     bits=8, batch_size=2048, verbose=False)
    # Verify
    recon, meta = ImageINRv5.decompress(res['recipe_bytes'])
    assert meta['exact_match'], "BLKH roundtrip failed"
    return res['recipe_size'], res['model_bit_accuracy'], res['psnr_db']


def main():
    print("=" * 100)
    print("  BLKH v5 vs PNG vs JPEG vs WebP vs ZIP")
    print("  * = lossless (fair comparison with BLKH)")
    print("=" * 100)

    photos = [
        ('Sky',    make_sky_photo(128, seed=1)),
        ('Wood',   make_wood_photo(128, seed=2)),
        ('Water',  make_water_photo(128, seed=3)),
        ('Skin',   make_skin_photo(128, seed=4)),
        ('Marble', make_marble_photo(128, seed=5)),
    ]

    results = []
    for name, img in photos:
        print(f"\n--- {name} ({img.shape}, {img.nbytes:,}B) ---")
        orig = img.nbytes

        png_sz = png_size(img)
        jpeg_sz = jpeg_size(img, 85)
        webp_l_sz = webp_lossless_size(img)
        webp_lossy_sz = webp_lossy_size(img, 85)
        zip_sz = zip_size(img)
        blkh_sz, blkh_bit, blkh_psnr = blkh_size(img)

        print(f"  *PNG (lossless):       {png_sz:>7,}B  ({orig/png_sz:.2f}x)")
        print(f"  *WebP lossless:        {webp_l_sz:>7,}B  ({orig/webp_l_sz:.2f}x)")
        print(f"  *ZIP (zlib-9):         {zip_sz:>7,}B  ({orig/zip_sz:.2f}x)")
        print(f"  *BLKH v5 bit-perfect:  {blkh_sz:>7,}B  ({orig/blkh_sz:.2f}x)  bit%={blkh_bit:.1f}  PSNR={blkh_psnr:.1f}dB")
        print(f"   JPEG q=85 (lossy):    {jpeg_sz:>7,}B  ({orig/jpeg_sz:.2f}x)  -- NOT lossless")
        print(f"   WebP q=85 (lossy):    {webp_lossy_sz:>7,}B  ({orig/webp_lossy_sz:.2f}x)  -- NOT lossless")

        results.append({
            'name': name,
            'orig_size': orig,
            'png_lossless': png_sz,
            'webp_lossless': webp_l_sz,
            'zip_lossless': zip_sz,
            'blkh_lossless': blkh_sz,
            'blkh_bit_pct': round(blkh_bit, 2),
            'blkh_psnr_db': round(blkh_psnr, 2),
            'jpeg_lossy': jpeg_sz,
            'webp_lossy': webp_lossy_sz,
        })

    print("\n" + "=" * 100)
    print("  LOSSLESS COMPARISON (fair — all preserve original bytes)")
    print("=" * 100)
    print(f"{'photo':<10}{'orig':>9}{'PNG':>9}{'WebP-L':>9}{'ZIP':>9}{'BLKH':>9}{'BLKH wins':>11}")
    print("-" * 100)
    blkh_wins = 0
    for r in results:
        competitors = {'PNG': r['png_lossless'], 'WebP-L': r['webp_lossless'],
                       'ZIP': r['zip_lossless'], 'BLKH': r['blkh_lossless']}
        winner = min(competitors, key=competitors.get)
        if winner == 'BLKH':
            blkh_wins += 1
        print(f"{r['name']:<10}{r['orig_size']:>9,}{r['png_lossless']:>9,}"
              f"{r['webp_lossless']:>9,}{r['zip_lossless']:>9,}{r['blkh_lossless']:>9,}"
              f"{winner:>11}")

    print(f"\n  BLKH wins {blkh_wins}/{len(results)} lossless comparisons.")

    print("\n" + "=" * 100)
    print("  NOTE: JPEG/WebP lossy are NOT comparable to BLKH (which is lossless)")
    print("  They're shown for context — BLKH preserves 100% of bytes via SHA-256.")
    print("=" * 100)

    out = Path(__file__).parent / 'benchmark_vs_image_formats_results.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")


if __name__ == '__main__':
    main()
