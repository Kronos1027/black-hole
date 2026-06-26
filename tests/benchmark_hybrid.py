#!/usr/bin/env python3
"""
benchmark_hybrid.py — BLKH v5.8 Hybrid mode vs everything else
================================================================
Tests the new hybrid mode (SIREN + WebP/PNG residual) on real photos
and compares against:
  - ZIP (lossless baseline)
  - PNG (lossless image codec)
  - WebP lossless
  - BLKH v5 bit-perfect (XOR + zlib residual)
  - BLKH v5.8 hybrid (WebP residual) <- NEW

The hypothesis: hybrid mode should beat both ZIP AND the original
BLKH bit-perfect mode, because it uses image-codec compression on
the residual (which has 2D structure) instead of zlib on random XOR bytes.
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
from siren_v5_hybrid import HybridCompressor
from benchmark_real_photos import (
    make_sky_photo, make_wood_photo, make_water_photo, make_skin_photo, make_marble_photo
)


def png_size(img):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='PNG', optimize=True)
    return len(buf.getvalue())


def webp_lossless_size(img):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='WebP', lossless=True)
    return len(buf.getvalue())


def main():
    print("=" * 110)
    print("  BLKH v5.8 HYBRID vs ZIP/PNG/WebP/BLKH-v5 (lossless, all bit-perfect)")
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
        zip_sz = len(zlib.compress(img.tobytes(), 9))
        png_sz = png_size(img)
        webp_l_sz = webp_lossless_size(img)
        print(f"  ZIP:          {zip_sz:>7,}B")
        print(f"  PNG:          {png_sz:>7,}B")
        print(f"  WebP-L:       {webp_l_sz:>7,}B")

        # BLKH v5 bit-perfect (XOR + zlib residual)
        comp_v5 = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res_v5 = comp_v5.compress_bitperfect(img, epochs=1000, lr=1e-3,
                                                bits=8, batch_size=2048, verbose=False)
        print(f"  BLKH v5:      {res_v5['recipe_size']:>7,}B  (XOR+zlib residual: {res_v5['residual_compressed_size']:,}B)")

        # BLKH v5.8 hybrid (WebP residual)
        comp_hyb = HybridCompressor(hidden_features=32, hidden_layers=2,
                                      omega_0=30.0, residual_codec='webp')
        res_hyb = comp_hyb.compress_bitperfect(img, epochs=1000, lr=1e-3,
                                                  bits=8, batch_size=2048, verbose=False)
        recon, meta = HybridCompressor.decompress(res_hyb['recipe_bytes'])
        assert meta['exact_match'], f"hybrid SHA-256 failed for {name}"
        print(f"  BLKH hybrid:  {res_hyb['recipe_size']:>7,}B  (WebP residual: {res_hyb['residual_compressed_size']:,}B)  ***")

        # Also test PNG residual
        comp_hyb_png = HybridCompressor(hidden_features=32, hidden_layers=2,
                                          omega_0=30.0, residual_codec='png')
        res_hyb_png = comp_hyb_png.compress_bitperfect(img, epochs=1000, lr=1e-3,
                                                          bits=8, batch_size=2048, verbose=False)
        print(f"  BLKH png-res: {res_hyb_png['recipe_size']:>7,}B  (PNG residual: {res_hyb_png['residual_compressed_size']:,}B)")

        results.append({
            'name': name,
            'orig': orig,
            'zip': zip_sz,
            'png': png_sz,
            'webp_lossless': webp_l_sz,
            'blkh_v5_xor_zlib': res_v5['recipe_size'],
            'blkh_hybrid_webp': res_hyb['recipe_size'],
            'blkh_hybrid_png': res_hyb_png['recipe_size'],
            'v5_residual': res_v5['residual_compressed_size'],
            'hybrid_webp_residual': res_hyb['residual_compressed_size'],
            'hybrid_png_residual': res_hyb_png['residual_compressed_size'],
            'bit_pct': round(res_hyb['model_bit_accuracy'], 2),
            'sha256_ok': meta['exact_match'],
        })

    print("\n" + "=" * 110)
    print("  HYBRID MODE SUMMARY — all lossless, all bit-perfect")
    print("=" * 110)
    print(f"{'photo':<10}{'orig':>9}{'ZIP':>9}{'PNG':>9}{'WebP-L':>9}{'BLKH v5':>10}{'BLKH hyb':>11}{'winner':>10}")
    print("-" * 110)
    n_hybrid_wins = 0
    for r in results:
        competitors = {
            'ZIP': r['zip'], 'PNG': r['png'], 'WebP-L': r['webp_lossless'],
            'BLKH v5': r['blkh_v5_xor_zlib'], 'BLKH hyb': r['blkh_hybrid_webp'],
        }
        winner = min(competitors, key=competitors.get)
        if winner == 'BLKH hyb':
            n_hybrid_wins += 1
        print(f"{r['name']:<10}{r['orig']:>9,}{r['zip']:>9,}{r['png']:>9,}"
              f"{r['webp_lossless']:>9,}{r['blkh_v5_xor_zlib']:>10,}"
              f"{r['blkh_hybrid_webp']:>11,}{winner:>10}")

    print(f"\n  BLKH hybrid wins {n_hybrid_wins}/{len(results)} photos (lossless comparison)")

    out = Path(__file__).parent / 'benchmark_hybrid_results.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")


if __name__ == '__main__':
    main()
