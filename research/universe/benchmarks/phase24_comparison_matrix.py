# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 24: Universal Codec Comparison Matrix
=============================================
Final comprehensive comparison of ALL BHUH methods vs ALL standards.

This is the DEFINITIVE benchmark — every method, every metric, one table.

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import io
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))


def run_phase24_experiment():
    """Run comprehensive codec comparison."""
    print("=" * 90)
    print("🧪 Phase 24: Universal Codec Comparison Matrix")
    print("=" * 90)

    # Load sample photos
    photos_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'docs', 'assets', 'sample_photos')
    if not os.path.exists(photos_dir):
        print("Sample photos not found")
        return

    results = []

    for fname in sorted(os.listdir(photos_dir)):
        if not fname.endswith('.png'):
            continue
        img = np.array(Image.open(os.path.join(photos_dir, fname)).convert('RGB'))
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        orig = img.nbytes
        zip_sz = len(zlib.compress(img.tobytes(), 9))
        png_buf = io.BytesIO()
        Image.fromarray(img).save(png_buf, format='PNG', optimize=True)
        png_sz = png_buf.tell()

        # JPEG
        jpg_buf = io.BytesIO()
        Image.fromarray(img).save(jpg_buf, format='JPEG', quality=80)
        jpg_sz = jpg_buf.tell()

        # WebP
        webp_buf = io.BytesIO()
        Image.fromarray(img).save(webp_buf, format='WebP', quality=80)
        webp_sz = webp_buf.tell()

        # BLKH DCT
        from siren_v5_dct import DCTCompressor
        comp = DCTCompressor(quality=0.9, codec='brotli')
        res = comp.compress(img, verbose=False)
        dct_sz = res['recipe_size']

        # BLKH Fast
        from siren_v5_fast import FastDCTCompressor
        comp = FastDCTCompressor(quality=0.9, speed='fast')
        res = comp.compress(img, verbose=False)
        fast_sz = res['recipe_size']

        # BLKH RLE
        from siren_v5_rle import RLEDCTCompressor
        comp = RLEDCTCompressor(quality=0.9, speed='balanced')
        res = comp.compress(img, verbose=False)
        rle_sz = res['recipe_size']

        # BLKH Photo
        from siren_v5_photo import PhotoCompressor
        comp = PhotoCompressor(subsampling='420', codec='brotli')
        res = comp.compress(img, verbose=False)
        photo_sz = res['recipe_size']

        # BLKH AVIF (if available)
        try:
            from siren_v5_avif import AVIFCompressor
            comp = AVIFCompressor(quality=0.9)
            res = comp.compress(img, verbose=False)
            avif_sz = res['recipe_size']
        except Exception:
            avif_sz = 0

        results.append({
            'name': fname,
            'original': orig,
            'zip': zip_sz,
            'png': png_sz,
            'jpeg': jpg_sz,
            'webp': webp_sz,
            'dct': dct_sz,
            'fast': fast_sz,
            'rle': rle_sz,
            'photo': photo_sz,
            'avif': avif_sz,
        })

    # Print comparison table
    print(f"\n{'Image':<18} {'ZIP':>7} {'PNG':>7} {'JPEG':>7} {'WebP':>7} {'DCT':>7} {'Fast':>7} {'RLE':>7} {'Photo':>7} {'AVIF':>7}")
    print("-" * 85)
    for r in results:
        print(f"{r['name']:<18} {r['zip']:>6,}B {r['png']:>6,}B {r['jpeg']:>6,}B {r['webp']:>6,}B "
              f"{r['dct']:>6,}B {r['fast']:>6,}B {r['rle']:>6,}B {r['photo']:>6,}B {r['avif']:>6,}B")

    # Averages
    n = len(results)
    print("-" * 85)
    print(f"{'AVERAGE':<18}", end="")
    for key in ['zip', 'png', 'jpeg', 'webp', 'dct', 'fast', 'rle', 'photo', 'avif']:
        avg = sum(r[key] for r in results) / n
        print(f" {avg:>6,.0f}B", end="")
    print()

    # BLKH vs standards
    print(f"\n{'='*90}")
    print("📊 BLKH vs STANDARDS — Average improvement")
    print(f"{'='*90}")

    avg_zip = sum(r['zip'] for r in results) / n
    for method in ['dct', 'fast', 'rle', 'photo', 'avif']:
        avg = sum(r[method] for r in results) / n
        if avg > 0:
            vs_zip = avg_zip / avg
            print(f"  BLKH {method.upper():>5}: {avg:>6,.0f}B avg  vs ZIP: {vs_zip:.2f}x")

    # Best method per image
    print(f"\n📋 Best method per image:")
    for r in results:
        methods = {k: r[k] for k in ['zip', 'png', 'jpeg', 'webp', 'dct', 'fast', 'rle', 'photo', 'avif'] if r[k] > 0}
        best = min(methods, key=methods.get)
        print(f"  {r['name']:<18} → {best.upper():>5} ({methods[best]:,}B)")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'phase24_comparison_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = run_phase24_experiment()
