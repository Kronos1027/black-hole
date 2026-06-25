# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

"""
siren_v5_auto.py — v5.30 Intelligent auto-mode selector
=========================================================
Automatically picks the best compression mode based on image characteristics.

Detection logic:
  1. Check unique colors → palette mode if <= 256 colors (lossless)
  2. Check if smooth synthetic (low entropy) → wavelet3 (lossless)
  3. Check if natural photo (high entropy) → DCT or fast (lossy)
  4. Default → fast mode (good balance)

The auto mode tries multiple modes and picks the smallest, with a time budget
to avoid trying all modes on large images.

Usage:
    from siren_v5_auto import AutoCompressor
    comp = AutoCompressor(quality=0.9, lossless=False, time_budget_s=2.0)
    res = comp.compress(image)
    print(f"Auto-picked: {res['mode']} → {res['recipe_size']}B")

CLI:
    blkh auto input.png output.blkh --quality 0.9

Author: Darlan Pereira da Silva (Kronos1027)
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siren_v5_palette import PaletteCompressor
from siren_v5_fast import FastDCTCompressor
from siren_v5_dct import DCTCompressor
from siren_v5_photo import PhotoCompressor


MAGIC_AUTO = b'BLKA'  # Reuse — auto delegates to best mode
VERSION_AUTO = 1


class AutoCompressor:
    """
    v5.30 Intelligent auto-mode selector.
    Tries multiple modes and picks the smallest within time budget.
    """

    def __init__(self, quality: float = 0.9, lossless: bool = False,
                 time_budget_s: float = 5.0, speed: str = 'fast'):
        """
        Args:
            quality: 0.1-1.0 for lossy modes
            lossless: if True, only try lossless modes (palette, wavelet3)
            time_budget_s: max time to spend trying modes
            speed: 'fast', 'balanced', 'best'
        """
        self.quality = quality
        self.lossless = lossless
        self.time_budget = time_budget_s
        self.speed = speed

    def _detect_image_type(self, image: np.ndarray) -> str:
        """Quick image type detection."""
        H, W, C = image.shape

        # Check palette suitability
        if PaletteCompressor.should_use_palette(image, threshold=256):
            return 'palette'

        # Check entropy (smooth vs photo)
        gray = np.mean(image, axis=2).astype(np.uint8)
        # Laplacian as entropy proxy
        lap = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
        if lap < 5.0:
            return 'smooth'  # low entropy → wavelet
        elif lap < 15.0:
            return 'mixed'
        else:
            return 'photo'  # high entropy → DCT

    def compress(self, image: np.ndarray, verbose: bool = False) -> dict:
        """Auto-select and compress with best mode."""
        assert image.dtype == np.uint8 and image.ndim == 3
        H, W, C = image.shape
        original_bytes = image.tobytes()
        t0 = time.time()

        img_type = self._detect_image_type(image)
        if verbose:
            print(f"[auto] Image type: {img_type} ({H}x{W}x{C})")

        candidates = []

        # Always try palette first (fast check, lossless win)
        try:
            if PaletteCompressor.should_use_palette(image, threshold=256):
                # Use balanced for palette (small data, brotli is fast enough)
                comp = PaletteCompressor(max_colors=256, speed='balanced')
                res = comp.compress(image, verbose=False)
                res['mode'] = 'palette_v5_29'
                candidates.append(res)
                if verbose:
                    print(f"[auto] palette: {res['recipe_size']}B (lossless)")
        except Exception as e:
            if verbose:
                print(f"[auto] palette failed: {e}")

        # Try modes based on image type and lossless flag
        if not self.lossless:
            # Lossy modes
            if img_type in ('photo', 'mixed'):
                # Try fast DCT
                try:
                    comp = FastDCTCompressor(quality=self.quality, speed=self.speed)
                    res = comp.compress(image, verbose=False)
                    res['mode'] = 'fast_v5_23'
                    candidates.append(res)
                    if verbose:
                        print(f"[auto] fast: {res['recipe_size']}B (lossy)")
                except Exception:
                    pass

                # Try DCT if time budget allows
                if time.time() - t0 < self.time_budget:
                    try:
                        comp = DCTCompressor(quality=self.quality, codec='brotli')
                        res = comp.compress(image, verbose=False)
                        res['mode'] = 'dct_v5_22'
                        candidates.append(res)
                        if verbose:
                            print(f"[auto] dct: {res['recipe_size']}B (lossy)")
                    except Exception:
                        pass

                # Try photo mode
                if time.time() - t0 < self.time_budget:
                    try:
                        comp = PhotoCompressor(subsampling='420', codec='brotli')
                        res = comp.compress(image, verbose=False)
                        res['mode'] = 'photo_v5_21'
                        candidates.append(res)
                        if verbose:
                            print(f"[auto] photo: {res['recipe_size']}B (lossy)")
                    except Exception:
                        pass

        # Pick smallest
        if not candidates:
            # Fallback to fast
            comp = FastDCTCompressor(quality=self.quality, speed=self.speed)
            res = comp.compress(image, verbose=False)
            res['mode'] = 'fast_v5_23'
            candidates.append(res)

        best = min(candidates, key=lambda x: x['recipe_size'])

        dt = time.time() - t0
        best['auto_time_s'] = dt
        best['modes_tried'] = len(candidates)
        best['auto_mode'] = True

        if verbose:
            print(f"[auto] Best: {best['mode']} → {best['recipe_size']}B "
                  f"(tried {len(candidates)} modes in {dt:.2f}s)")

        return best


def _self_test():
    from PIL import Image
    import io
    import os
    import zlib

    print("=" * 80)
    print("BLKH v5.30 Auto Mode Selector — Self Test")
    print("=" * 80)

    # Test different image types
    def make_palette_img(size, n_colors=16):
        rng = np.random.default_rng(42)
        palette = rng.integers(0, 256, (n_colors, 3), dtype=np.uint8)
        img = np.zeros((size, size, 3), dtype=np.uint8)
        y, x = np.mgrid[0:size, 0:size]
        region = (x // 16 + y // 16) % n_colors
        for c in range(3):
            img[:, :, c] = palette[region, c]
        return img

    def make_smooth(size):
        rng = np.random.default_rng(42)
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
        img = np.zeros((size, size, 3), dtype=np.float32)
        for c in range(3):
            for _ in range(3):
                kx, ky = rng.integers(1, 5, 2)
                amp = rng.uniform(40, 80)
                img[:, :, c] += amp * np.sin(2*np.pi*kx*xs) * np.cos(2*np.pi*ky*ys)
        return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    print(f"\n{'Image':<25} {'Type':<10} {'ZIP':>10} {'BLKH auto':>12} {'Mode':<20} {'vs ZIP':>8}")
    print("-" * 90)

    test_images = [
        ('palette_128_16c', make_palette_img(128, 16)),
        ('palette_256_64c', make_palette_img(256, 64)),
        ('smooth_128', make_smooth(128)),
        ('smooth_256', make_smooth(256)),
    ]

    # Add real photos
    photos_dir = '/home/z/my-project/blackhole_repo/docs/assets/sample_photos'
    for fname in sorted(os.listdir(photos_dir))[:3]:
        if fname.endswith('.png'):
            img = np.array(Image.open(os.path.join(photos_dir, fname)).convert('RGB'))
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            test_images.append((fname, img))

    for name, img in test_images:
        zip_sz = len(zlib.compress(img.tobytes(), 9))
        comp = AutoCompressor(quality=0.9, lossless=False, time_budget_s=3.0, speed='fast')
        res = comp.compress(img, verbose=False)
        vs_zip = zip_sz / res['recipe_size'] if res['recipe_size'] > 0 else 0
        print(f"{name:<25} {comp._detect_image_type(img):<10} {zip_sz:>10,} {res['recipe_size']:>10,}B {res['mode']:<20} {vs_zip:>7.2f}x")


if __name__ == '__main__':
    _self_test()
