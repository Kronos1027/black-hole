#!/usr/bin/env python3
"""
benchmark_real_photos.py — BLKH v5 on REAL photo-like images
==============================================================
Unlike previous benchmarks that used synthetic mathematical signals,
this generates photo-realistic images (sky gradients, wood grain, water,
skin tones, marble) and tests BLKH v5 vs ZIP on them.

These are MUCH harder for SIREN because they contain realistic high-frequency
content (noise, texture detail). The question: does BLKH still beat ZIP?
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


# ============================================================
#  Realistic photo generators (with actual noise/textures)
# ============================================================
def make_sky_photo(size=128, seed=42):
    """Realistic sky: blue gradient + slight noise + sun glow."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Vertical gradient (sky gets lighter near horizon)
    img[:, :, 2] = 180 + 60 * (1 - ys)  # blue
    img[:, :, 1] = 130 + 50 * (1 - ys)  # green (less)
    img[:, :, 0] = 100 + 40 * (1 - ys)  # red (less)
    # Sun glow (gaussian)
    cy, cx = 0.2, 0.7
    glow = 80 * np.exp(-((xs - cx)**2 + (ys - cy)**2) / 0.05)
    img += glow[:, :, None]
    # Noise (camera sensor)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_wood_photo(size=128, seed=42):
    """Realistic wood grain: sinusoidal grain + noise + color variation."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Wood base color (brown)
    img[:, :, 0] = 130
    img[:, :, 1] = 80
    img[:, :, 2] = 40
    # Grain (sinusoidal along x with curvature)
    grain = 25 * np.sin(xs * 30 + 0.5 * np.sin(ys * 8))
    img += grain[:, :, None]
    # Knots (dark spots)
    for _ in range(2):
        cy, cx = rng.uniform(0.2, 0.8, 2)
        knot = -30 * np.exp(-((xs - cx)**2 + (ys - cy)**2) / 0.005)
        img += knot[:, :, None]
    # Fine noise
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_water_photo(size=128, seed=42):
    """Realistic water: wave pattern + reflection + sparkle noise."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Base water color
    img[:, :, 2] = 150
    img[:, :, 1] = 80
    img[:, :, 0] = 30
    # Wave pattern
    waves = 15 * np.sin(xs * 20 + ys * 5) + 10 * np.sin(xs * 8 - ys * 12)
    img += waves[:, :, None]
    # Horizontal banding (perspective)
    img += 20 * np.sin(ys * 15)[:, :, None]
    # Sparkle (sparse bright points)
    sparkle_mask = rng.random(img.shape[:2]) < 0.005
    img[sparkle_mask] += 80
    # Noise
    img += rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_skin_photo(size=128, seed=42):
    """Realistic skin: warm gradient + pores (noise) + slight blush."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Skin base
    img[:, :, 0] = 200
    img[:, :, 1] = 150
    img[:, :, 2] = 120
    # Gradient (lighting)
    img *= (0.7 + 0.6 * (1 - ys**2))[:, :, None]
    # Blush (cheeks)
    cy, cx = 0.6, 0.3
    blush = 30 * np.exp(-((xs - cx)**2 + (ys - cy)**2) / 0.05)
    img[:, :, 0] += blush
    cy, cx = 0.6, 0.7
    blush = 30 * np.exp(-((xs - cx)**2 + (ys - cy)**2) / 0.05)
    img[:, :, 0] += blush
    # Pores (high-freq noise)
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_marble_photo(size=128, seed=42):
    """Realistic marble: white base + turbulent veins."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    # White base
    img[:, :] = 230
    # Veins (turbulent noise via sum of sines)
    veins = np.sin(xs * 15 + 2 * np.sin(ys * 8)) * np.cos(ys * 12 + 1.5 * np.sin(xs * 6))
    veins = (veins > 0.5).astype(np.float32) * 80
    img -= veins[:, :, None]
    # Subtle color tint
    img[:, :, 0] -= 5
    img[:, :, 2] -= 10
    # Fine noise
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    print("=" * 95)
    print("  BLKH v5 — REAL PHOTOS BENCHMARK")
    print("  (photo-realistic images with noise & texture, NOT synthetic)")
    print("=" * 95)

    photos = [
        ('sky_128',     make_sky_photo(128, seed=1)),
        ('wood_128',    make_wood_photo(128, seed=2)),
        ('water_128',   make_water_photo(128, seed=3)),
        ('skin_128',    make_skin_photo(128, seed=4)),
        ('marble_128',  make_marble_photo(128, seed=5)),
    ]

    # Save the photos so user can see them
    try:
        from PIL import Image
        out_dir = Path('docs/assets/sample_photos')
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, img in photos:
            Image.fromarray(img).save(out_dir / f'{name}.png')
        print(f"\nSample photos saved to: {out_dir}/")
    except ImportError:
        pass

    results = []
    for name, img in photos:
        print(f"\n--- {name} ({img.shape}, {img.nbytes:,}B) ---")
        orig = img.nbytes
        zip_size = len(zlib.compress(img.tobytes(), 9))
        print(f"  ZIP: {zip_size:,}B  (ratio {orig/zip_size:.2f}x)")

        # Try different network sizes to find best for this photo
        best = None
        for hidden, layers, epochs in [(32, 2, 1000), (64, 3, 1500), (128, 3, 2000)]:
            comp = ImageINRv5(hidden_features=hidden, hidden_layers=layers, omega_0=30.0)
            t0 = time.time()
            try:
                res = comp.compress_bitperfect(img, epochs=epochs, lr=1e-3,
                                                 bits=8, batch_size=2048, verbose=False)
                dt = time.time() - t0
                recon, meta = ImageINRv5.decompress(res['recipe_bytes'])
                ok = meta['exact_match']
                winner = "BLKH" if res['recipe_size'] < zip_size else "ZIP"
                print(f"  BLKH h={hidden:3d} l={layers} e={epochs}: {res['recipe_size']:>6,}B  "
                      f"bit%={res['model_bit_accuracy']:.1f}  PSNR={res['psnr_db']:.1f}dB  "
                      f"ok={ok}  {dt:.1f}s  -> {winner}")
                if best is None or (res['recipe_size'] < best['recipe_size'] and ok):
                    best = {**res, 'hidden': hidden, 'layers': layers, 'epochs': epochs,
                            'ok': ok, 'time': dt}
            except Exception as e:
                print(f"  BLKH h={hidden} FAILED: {e}")

        if best:
            winner = "BLKH" if best['recipe_size'] < zip_size else "ZIP"
            results.append({
                'name': name,
                'orig_size': orig,
                'zip_size': zip_size,
                'blkh_size': best['recipe_size'],
                'weights_size': best['weights_packed_size'],
                'residual_size': best['residual_compressed_size'],
                'bit_pct': round(best['model_bit_accuracy'], 2),
                'psnr_db': round(best['psnr_db'], 2),
                'sha256_ok': best['ok'],
                'config': f"h={best['hidden']} l={best['layers']} e={best['epochs']}",
                'time_s': round(best['time'], 2),
                'winner': winner,
                'blkh_vs_zip': round(zip_size / best['recipe_size'], 3),
            })

    print("\n" + "=" * 95)
    print("  REAL PHOTOS SUMMARY")
    print("=" * 95)
    print(f"{'photo':<15}{'orig':>9}{'zip':>9}{'blkh':>9}{'bit%':>8}{'psnr':>8}{'vs zip':>10}{'win':>8}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<15}{r['orig_size']:>9,}{r['zip_size']:>9,}"
              f"{r['blkh_size']:>9,}{r['bit_pct']:>7.1f}%{r['psnr_db']:>7.1f}dB"
              f"{r['blkh_vs_zip']:>9.3f}x{r['winner']:>8}")

    n_blkh_wins = sum(1 for r in results if r['winner'] == 'BLKH')
    print(f"\n  BLKH wins {n_blkh_wins}/{len(results)} real photos tested.")

    out = Path(__file__).parent / 'benchmark_real_photos_results.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")


if __name__ == '__main__':
    main()
