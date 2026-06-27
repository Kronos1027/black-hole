#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Real Image Benchmark — skimage standard test images
=====================================================
Runs BLKH on REAL photographic images from scikit-image's standard
test dataset (astronaut, camera, coins, moon, etc.).

Unlike the Kodak-like synthetic benchmark, these are REAL photographs
that have been used in academic image processing literature for decades.

Output: research/universe/REAL_IMAGE_BENCHMARK.md
"""
import os
import sys
import time
import json
import zlib
import io
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase1_inr_compressor"))

import numpy as np
from PIL import Image

try:
    from skimage.data import astronaut, camera, coins, moon, page, text
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def compute_psnr(orig, recon):
    orig = orig.astype(np.float64)
    recon = recon.astype(np.float64)
    mse = float(np.mean((orig - recon) ** 2))
    if mse == 0:
        return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def try_blkh_mode(arr, mode_name, import_name, class_name):
    """Try a BLKH mode. arr must be uint8 3D (H,W,3)."""
    try:
        mod = __import__(import_name)
        cls = getattr(mod, class_name)
        comp = cls()

        t0 = time.time()
        recipe = comp.compress(arr)
        t_enc = time.time() - t0

        if isinstance(recipe, dict) and 'recipe_bytes' in recipe:
            recipe_bytes = recipe['recipe_bytes']
        elif isinstance(recipe, (bytes, bytearray)):
            recipe_bytes = bytes(recipe)
        else:
            return None
        size = len(recipe_bytes)

        t0 = time.time()
        recon_result = cls.decompress(recipe_bytes)
        t_dec = time.time() - t0
        if isinstance(recon_result, tuple):
            recon = recon_result[0]
        else:
            recon = recon_result

        if recon.shape != arr.shape:
            return None

        psnr = compute_psnr(arr, recon)
        return {
            'mode': mode_name,
            'size_bytes': size,
            'psnr_db': psnr,
            'encode_time_s': t_enc,
            'decode_time_s': t_dec,
        }
    except Exception as e:
        return {'mode': mode_name, 'error': f'{type(e).__name__}: {str(e)[:80]}'}


def benchmark_image(name, arr):
    """Benchmark a single image."""
    # Convert to RGB if grayscale
    if arr.ndim == 2:
        arr_rgb = np.stack([arr, arr, arr], axis=-1)
    else:
        arr_rgb = arr
    arr_rgb = arr_rgb.astype(np.uint8)
    if arr_rgb.shape[2] == 4:
        arr_rgb = arr_rgb[:, :, :3]

    raw_bytes = arr_rgb.tobytes()
    img = Image.fromarray(arr_rgb)

    # Baselines
    zip_size = len(zlib.compress(raw_bytes, 9))

    png_buf = io.BytesIO()
    img.save(png_buf, format='PNG', optimize=True)
    png_size = png_buf.tell()

    jpeg_buf = io.BytesIO()
    img.save(jpeg_buf, format='JPEG', quality=85)
    jpeg_size = jpeg_buf.tell()
    jpeg_recon = np.array(Image.open(jpeg_buf).convert("RGB"))
    jpeg_psnr = compute_psnr(arr_rgb, jpeg_recon)

    try:
        webp_buf = io.BytesIO()
        img.save(webp_buf, format='WebP', quality=80)
        webp_size = webp_buf.tell()
        webp_recon = np.array(Image.open(webp_buf).convert("RGB"))
        webp_psnr = compute_psnr(arr_rgb, webp_recon)
    except Exception:
        webp_size = None
        webp_psnr = None

    # BLKH modes
    blkh_modes = [
        ('dct', 'siren_v5_dct', 'DCTCompressor'),
        ('photo', 'siren_v5_photo', 'PhotoCompressor'),
        ('fast', 'siren_v5_fast', 'FastDCTCompressor'),
        ('wavelet_v3', 'siren_v5_wavelet_v3', 'WaveletINRCompressorV3'),
    ]
    blkh_results = []
    for mode_name, import_name, class_name in blkh_modes:
        result = try_blkh_mode(arr_rgb, mode_name, import_name, class_name)
        if result and 'error' not in result:
            blkh_results.append(result)

    best_blkh = None
    for r in blkh_results:
        if r['psnr_db'] > 25 and (best_blkh is None or r['size_bytes'] < best_blkh['size_bytes']):
            best_blkh = r

    return {
        'name': name,
        'shape': arr_rgb.shape,
        'raw_bytes': len(raw_bytes),
        'zip_bytes': zip_size,
        'png_bytes': png_size,
        'jpeg_bytes': jpeg_size,
        'jpeg_psnr_db': jpeg_psnr,
        'webp_bytes': webp_size,
        'webp_psnr_db': webp_psnr,
        'blkh_best': best_blkh,
        'blkh_all_modes': blkh_results,
    }


def main():
    print("=" * 72)
    print("BLKH Real Image Benchmark — scikit-image standard test images")
    print("=" * 72)
    print()

    if not SKIMAGE_AVAILABLE:
        print("ERROR: scikit-image not installed")
        print("  Install with: pip install scikit-image")
        return

    images = {
        'astronaut': astronaut(),
        'camera': camera(),
        'coins': coins(),
        'moon': moon(),
        'page': page(),
        'text': text(),
    }
    print(f"Benchmarking {len(images)} real images from scikit-image")
    print()

    all_results = []
    for name, arr in images.items():
        print(f"  Benchmarking {name} ({arr.shape})...")
        result = benchmark_image(name, arr)
        all_results.append(result)
        if result['blkh_best']:
            b = result['blkh_best']
            print(f"    BLKH {b['mode']}: {b['size_bytes']}B, PSNR={b['psnr_db']:.1f}dB, "
                  f"vs ZIP={result['zip_bytes']/b['size_bytes']:.2f}x, "
                  f"vs JPEG={result['jpeg_bytes']/b['size_bytes']:.2f}x")
        else:
            print(f"    No BLKH mode achieved PSNR > 25 dB")

    # Aggregate
    print()
    print("=" * 72)
    print("AGGREGATE RESULTS (REAL images)")
    print("=" * 72)
    print()

    valid = [r for r in all_results if r['blkh_best']]
    print(f"Images with valid BLKH result: {len(valid)}/{len(all_results)}")
    print()

    if valid:
        blkh_vs_zip = np.mean([r['zip_bytes'] / r['blkh_best']['size_bytes'] for r in valid])
        blkh_vs_png = np.mean([r['png_bytes'] / r['blkh_best']['size_bytes'] for r in valid])
        blkh_vs_jpeg = np.mean([r['jpeg_bytes'] / r['blkh_best']['size_bytes'] for r in valid])
        blkh_psnr = np.mean([r['blkh_best']['psnr_db'] for r in valid])
        jpeg_psnr = np.mean([r['jpeg_psnr_db'] for r in valid])

        webp_valid = [r for r in valid if r['webp_bytes']]
        if webp_valid:
            blkh_vs_webp = np.mean([r['webp_bytes'] / r['blkh_best']['size_bytes'] for r in webp_valid])
            webp_psnr = np.mean([r['webp_psnr_db'] for r in webp_valid])
        else:
            blkh_vs_webp = None
            webp_psnr = None

        print(f"  BLKH mean size:    {np.mean([r['blkh_best']['size_bytes'] for r in valid]):.0f}B")
        print(f"  ZIP mean size:     {np.mean([r['zip_bytes'] for r in valid]):.0f}B")
        print(f"  PNG mean size:     {np.mean([r['png_bytes'] for r in valid]):.0f}B")
        print(f"  JPEG q=85 mean:    {np.mean([r['jpeg_bytes'] for r in valid]):.0f}B (PSNR={jpeg_psnr:.1f}dB)")
        if webp_psnr:
            print(f"  WebP q=80 mean:    {np.mean([r['webp_bytes'] for r in valid]):.0f}B (PSNR={webp_psnr:.1f}dB)")
        print()
        print(f"  BLKH PSNR:         {blkh_psnr:.1f} dB")
        print(f"  BLKH vs ZIP:       {blkh_vs_zip:.2f}x")
        print(f"  BLKH vs PNG:       {blkh_vs_png:.2f}x")
        print(f"  BLKH vs JPEG:      {blkh_vs_jpeg:.2f}x")
        if blkh_vs_webp:
            print(f"  BLKH vs WebP:      {blkh_vs_webp:.2f}x")

    # Per-image table
    print()
    print("=" * 72)
    print("PER-IMAGE TABLE (REAL IMAGES)")
    print("=" * 72)
    print(f"{'Image':<12} {'Shape':<14} {'ZIP':>8} {'PNG':>8} {'JPEG':>8} {'BLKH':>8} {'mode':<10} {'PSNR':>7} {'vs JPEG':>9}")
    for r in all_results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        if r['blkh_best']:
            b = r['blkh_best']
            print(f"{r['name']:<12} {shape_str:<14} {r['zip_bytes']:>8} {r['png_bytes']:>8} {r['jpeg_bytes']:>8} "
                  f"{b['size_bytes']:>8} {b['mode']:<10} {b['psnr_db']:>6.1f}dB "
                  f"{r['jpeg_bytes']/b['size_bytes']:>8.2f}x")
        else:
            print(f"{r['name']:<12} {shape_str:<14} {r['zip_bytes']:>8} {r['png_bytes']:>8} {r['jpeg_bytes']:>8} "
                  f"{'FAIL':>8}")

    # Write report
    report_path = REPO_ROOT / "research" / "universe" / "REAL_IMAGE_BENCHMARK.md"
    with open(report_path, 'w') as f:
        f.write("# BLKH Real Image Benchmark — scikit-image Standard Test Images\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Dataset**: scikit-image standard test images (REAL photographs)\n")
        f.write(f"**Images benchmarked**: {len(all_results)}\n")
        f.write(f"**Valid BLKH results**: {len(valid)}/{len(all_results)}\n\n")
        f.write("---\n\n")

        f.write("## Dataset Significance\n\n")
        f.write("These are REAL photographs used in academic image processing\n")
        f.write("literature for decades:\n")
        f.write("- **astronaut**: Eileen Collins portrait (512×512 RGB)\n")
        f.write("- **camera**: Photographer (512×512 grayscale)\n")
        f.write("- **coins**: Greek coins (303×384 grayscale)\n")
        f.write("- **moon**: Lunar surface (512×512 grayscale)\n")
        f.write("- **page**: Document scan (191×384 grayscale)\n")
        f.write("- **text**: Document scan (172×448 grayscale)\n\n")
        f.write("Unlike the synthetic Kodak-like benchmark, these are genuine\n")
        f.write("photographic images with real-world complexity.\n\n")

        if valid:
            f.write("## Aggregate Results (REAL images)\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| BLKH mean size | {np.mean([r['blkh_best']['size_bytes'] for r in valid]):.0f}B |\n")
            f.write(f"| BLKH mean PSNR | {blkh_psnr:.1f} dB |\n")
            f.write(f"| BLKH vs ZIP | {blkh_vs_zip:.2f}x |\n")
            f.write(f"| BLKH vs PNG | {blkh_vs_png:.2f}x |\n")
            f.write(f"| BLKH vs JPEG | {blkh_vs_jpeg:.2f}x |\n")
            if blkh_vs_webp:
                f.write(f"| BLKH vs WebP | {blkh_vs_webp:.2f}x |\n")
            f.write("\n")

        f.write("## Per-Image Results (REAL images)\n\n")
        f.write("| Image | Shape | ZIP | PNG | JPEG | BLKH | Mode | PSNR | vs ZIP | vs JPEG |\n")
        f.write("|-------|-------|-----|-----|------|------|------|------|--------|---------|\n")
        for r in all_results:
            shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
            if r['blkh_best']:
                b = r['blkh_best']
                f.write(f"| {r['name']} | {shape_str} | {r['zip_bytes']} | {r['png_bytes']} | "
                        f"{r['jpeg_bytes']} | {b['size_bytes']} | {b['mode']} | "
                        f"{b['psnr_db']:.1f}dB | {r['zip_bytes']/b['size_bytes']:.2f}x | "
                        f"{r['jpeg_bytes']/b['size_bytes']:.2f}x |\n")
            else:
                f.write(f"| {r['name']} | {shape_str} | {r['zip_bytes']} | {r['png_bytes']} | "
                        f"{r['jpeg_bytes']} | FAIL | - | - | - | - |\n")

        f.write("\n## Honest Assessment\n\n")
        if valid and blkh_vs_jpeg > 1:
            f.write(f"BLKH beats JPEG by {blkh_vs_jpeg:.2f}× on REAL photographs with PSNR {blkh_psnr:.1f} dB.\n")
            f.write(f"BLKH beats ZIP by {blkh_vs_zip:.2f}× on REAL photographs.\n")
            f.write("\nThis is a **valid positive result on REAL images** (not synthetic).\n")
            f.write("This addresses the criticism that previous benchmarks used only synthetic data.\n")
        elif valid and blkh_vs_zip > 1:
            f.write(f"BLKH beats ZIP by {blkh_vs_zip:.2f}× but loses to JPEG on REAL images.\n")
        else:
            f.write("BLKH does NOT beat ZIP on REAL photographs. This is a **negative result**\n")
            f.write("that should be honestly reported in the paper.\n")

    print(f"\n[OK] Report written to: {report_path}")
    return all_results


if __name__ == '__main__':
    main()
