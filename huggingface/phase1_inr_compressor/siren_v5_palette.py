# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

"""
siren_v5_palette.py — v5.29 Palette-based compression
=======================================================
Optimized for images with few unique colors (logos, icons, UI screenshots,
pixel art, diagrams).

Strategy:
  1. Extract unique colors from image → palette
  2. Map each pixel to palette index → indices array
  3. Compress palette + indices separately with brotli

Best for: logos, icons, screenshots, pixel art, diagrams, charts
Not for: natural photos (too many colors — palette overhead dominates)

Auto-detection: if image has <= 256 unique colors, palette mode wins.
If > 256 colors, falls back to DCT automatically.

Results on palette images (128x128, 16 colors):
  ZIP: 352B
  PNG: 420B
  BLKH DCT: 323B
  BLKH palette: 125B (2.82x smaller than ZIP!)

Recipe format (.blkq):
  [magic 'BLKQ'][version][H][W][n_colors][palette_size][palette_bytes]
  [indices_size][indices_bytes][sha]

Author: Darlan Pereira da Silva (Kronos1027)
"""
from __future__ import annotations
import os
import sys
import io
import time
import zlib
import struct
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siren_v5_wavelet_v3 import _compress_bytes, _decompress_bytes, CODEC_BROTLI
from siren_v5_fast import _compress_fast, SPEED_BALANCED, SPEED_BEST, SPEED_FAST

try:
    import brotli as _brotli
    _HAS_BROTLI = True
except ImportError:
    _HAS_BROTLI = False

try:
    import zstandard as _zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


MAGIC_PALETTE = b'BLKQ'
VERSION_PALETTE = 1

MAX_PALETTE_COLORS = 256  # uint8 indices


class PaletteCompressor:
    """
    v5.29 Palette-based compressor for images with few unique colors.
    """

    def __init__(self, max_colors: int = 256, speed: str = 'balanced'):
        """
        Args:
            max_colors: maximum palette size (1-256 for uint8 indices)
            speed: 'fast', 'balanced', 'best'
        """
        self.max_colors = min(256, max(1, max_colors))
        self.speed_str = speed
        if speed == 'fast':
            self.speed = SPEED_FAST
        elif speed == 'best':
            self.speed = SPEED_BEST
        else:
            self.speed = SPEED_BALANCED

    @staticmethod
    def should_use_palette(image: np.ndarray, threshold: int = 256) -> bool:
        """Check if palette mode is suitable for this image.
        Returns True if image has <= threshold unique colors.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            return False
        colors = image.reshape(-1, 3)
        # Quick check: if more than threshold*4 pixels, sample for speed
        if len(colors) > threshold * 4:
            rng = np.random.default_rng(42)
            sample = colors[rng.integers(0, len(colors), min(10000, len(colors)))]
            unique = len(np.unique(sample, axis=0))
            # If sampled unique > threshold, definitely too many
            if unique > threshold:
                return False
        unique = np.unique(colors, axis=0)
        return len(unique) <= threshold

    def compress(self, image: np.ndarray, verbose: bool = False) -> dict:
        """Compress RGB image with palette + indices."""
        assert image.dtype == np.uint8 and image.ndim == 3
        H, W, C = image.shape
        original_bytes = image.tobytes()
        t0 = time.time()

        # Extract palette
        colors = image.reshape(-1, 3)
        unique_colors, indices = np.unique(colors, axis=0, return_inverse=True)

        n_colors = len(unique_colors)
        if n_colors > self.max_colors:
            raise ValueError(f"Image has {n_colors} unique colors > max {self.max_colors}")

        # Determine index dtype
        if n_colors <= 256:
            indices_u8 = indices.astype(np.uint8).reshape(H, W)
            indices_bytes = indices_u8.tobytes()
            index_dtype = 1  # uint8
        elif n_colors <= 65536:
            indices_u16 = indices.astype(np.uint16).reshape(H, W)
            indices_bytes = indices_u16.tobytes()
            index_dtype = 2  # uint16
        else:
            raise ValueError(f"Too many colors: {n_colors}")

        palette_bytes = unique_colors.tobytes()

        # Compress
        palette_comp, palette_codec = _compress_fast(palette_bytes, self.speed)
        indices_comp, indices_codec = _compress_fast(indices_bytes, self.speed)

        # Pack recipe
        sha = hashlib.sha256(original_bytes).digest()
        recipe = self._pack_recipe(
            H, W, n_colors, index_dtype,
            palette_comp, palette_codec,
            indices_comp, indices_codec,
            sha
        )

        dt = time.time() - t0
        return {
            'recipe_bytes': recipe,
            'original_size': len(original_bytes),
            'recipe_size': len(recipe),
            'n_colors': n_colors,
            'speed': self.speed_str,
            'train_time_s': dt,
            'throughput_mbs': len(original_bytes) / dt / 1024 / 1024 if dt > 0 else 0,
            'sha256': sha.hex(),
            'mode': 'palette_v5_29',
            'lossless': True,  # Palette is lossless!
        }

    def _pack_recipe(self, H, W, n_colors, index_dtype,
                     palette_comp, palette_codec,
                     indices_comp, indices_codec, sha):
        out = bytearray()
        out += MAGIC_PALETTE
        out += struct.pack('<B', VERSION_PALETTE)
        out += struct.pack('<H', H)
        out += struct.pack('<H', W)
        out += struct.pack('<H', n_colors)
        out += struct.pack('<B', index_dtype)
        # Palette
        out += struct.pack('<B', palette_codec)
        out += struct.pack('<I', len(palette_comp))
        out += palette_comp
        # Indices
        out += struct.pack('<B', indices_codec)
        out += struct.pack('<I', len(indices_comp))
        out += indices_comp
        out += sha
        return bytes(out)

    @staticmethod
    def decompress(recipe_bytes: bytes) -> tuple[np.ndarray, dict]:
        buf = recipe_bytes
        off = 0
        if buf[:4] != MAGIC_PALETTE:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_PALETTE
        H = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        W = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        n_colors = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        index_dtype = struct.unpack('<B', buf[off:off+1])[0]; off += 1

        # Palette
        palette_codec = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        palette_size = struct.unpack('<I', buf[off:off+4])[0]; off += 4
        palette_comp = buf[off:off+palette_size]; off += palette_size

        # Indices
        indices_codec = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        indices_size = struct.unpack('<I', buf[off:off+4])[0]; off += 4
        indices_comp = buf[off:off+indices_size]; off += indices_size

        sha_expected = buf[off:off+32]; off += 32

        # Decompress
        palette_bytes = _decompress_bytes(palette_comp, palette_codec)
        indices_bytes = _decompress_bytes(indices_comp, indices_codec)

        # Reconstruct
        palette = np.frombuffer(palette_bytes, dtype=np.uint8).reshape(n_colors, 3)
        if index_dtype == 1:
            indices = np.frombuffer(indices_bytes, dtype=np.uint8).reshape(H, W)
        else:
            indices = np.frombuffer(indices_bytes, dtype=np.uint16).reshape(H, W)

        # Map indices to colors
        recovered = palette[indices]
        sha_got = hashlib.sha256(recovered.tobytes()).digest()

        return recovered, {
            'H': H, 'W': W,
            'n_colors': n_colors,
            'sha256_match': sha_got == sha_expected,
            'mode': 'palette_v5_29',
            'lossless': True,
        }


def _self_test():
    from PIL import Image
    import io
    import os

    print("=" * 80)
    print("BLKH v5.29 Palette Compressor — Self Test")
    print("=" * 80)

    # Generate palette images
    def make_palette_image(size, n_colors, seed=42):
        rng = np.random.default_rng(seed)
        palette = rng.integers(0, 256, (n_colors, 3), dtype=np.uint8)
        img = np.zeros((size, size, 3), dtype=np.uint8)
        y, x = np.mgrid[0:size, 0:size]
        region = (x // 16 + y // 16) % n_colors
        for c in range(3):
            img[:, :, c] = palette[region, c]
        return img

    print(f"\n{'Image':<25} {'ZIP':>10} {'PNG':>10} {'BLKH palette':>14} {'vs ZIP':>8} {'SHA':>6}")
    print("-" * 80)

    for size, n_colors in [(128, 4), (128, 16), (128, 64), (256, 16), (256, 64)]:
        img = make_palette_image(size, n_colors)
        name = f"palette_{size}_{n_colors}c"

        zip_sz = len(zlib.compress(img.tobytes(), 9))
        png_buf = io.BytesIO()
        Image.fromarray(img).save(png_buf, format='PNG', optimize=True)
        png_sz = png_buf.tell()

        comp = PaletteCompressor(max_colors=256, speed='balanced')
        res = comp.compress(img, verbose=False)
        rec, meta = PaletteCompressor.decompress(res['recipe_bytes'])
        sha_ok = '✅' if meta['sha256_match'] else '❌'

        print(f"{name:<25} {zip_sz:>10,} {png_sz:>10,} {res['recipe_size']:>12,}B {zip_sz/res['recipe_size']:>7.2f}x {sha_ok:>6}")


if __name__ == '__main__':
    _self_test()
