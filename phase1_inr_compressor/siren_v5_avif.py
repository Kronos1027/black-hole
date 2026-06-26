# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

"""
siren_v5_avif.py — v5.26 AVIF/HEIF wrapper
=============================================
Wraps industry-standard AVIF/HEIF encoders in BLKH unified API.

Why: AVIF is the modern standard (replaces JPEG), supported by all major browsers.
This module provides a unified BLKH interface so users can choose AVIF when they
need maximum compatibility with existing tools.

Strategy:
  1. Use pillow-avif-plugin for AVIF encoding
  2. Use pillow-heif for HEIF encoding (optional)
  3. Wrap in BLKH API with quality parameter (0.0-1.0 → AVIF quality 0-100)
  4. Provide same compress/decompress interface as other BLKH modes

Quality mapping:
  quality=1.0 → AVIF quality=95 (best)
  quality=0.9 → AVIF quality=80
  quality=0.5 → AVIF quality=50
  quality=0.1 → AVIF quality=10 (most lossy)

Recipe format (.blka):
  [magic 'BLKA'][version][quality_int][H][W][avif_data]

Author: Darlan Pereira da Silva (Kronos1027)
"""
from __future__ import annotations
import os
import sys
import io
import time
import struct
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


MAGIC_AVIF = b'BLKA'  # Note: this conflicts with audio's BLKA, but they're in different contexts
MAGIC_AVIF_V2 = b'BLHV'  # BLKH Heif/AVIF
VERSION_AVIF = 1

# Try imports
try:
    import pillow_avif  # noqa: F401 - registers AVIF with PIL
    _HAS_AVIF = True
except ImportError:
    _HAS_AVIF = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HAS_HEIF = True
except ImportError:
    _HAS_HEIF = False


def _quality_to_avif_quality(quality: float) -> int:
    """Convert BLKH quality (0.1-1.0) to AVIF quality (10-95)."""
    q = max(0.1, min(1.0, quality))
    return int(10 + q * 85)  # 0.1 → 18, 1.0 → 95


class AVIFCompressor:
    """
    v5.26 AVIF/HEIF wrapper for BLKH.
    Provides modern format support alongside BLKH's native modes.
    """

    def __init__(self, quality: float = 0.9, format: str = 'AVIF'):
        """
        Args:
            quality: 0.1-1.0 (1.0=best)
            format: 'AVIF' or 'HEIF'
        """
        if format.upper() == 'AVIF' and not _HAS_AVIF:
            raise ImportError("AVIF not available. Install: pip install pillow-avif-plugin")
        if format.upper() == 'HEIF' and not _HAS_HEIF:
            raise ImportError("HEIF not available. Install: pip install pillow-heif")
        self.quality = float(max(0.1, min(1.0, quality)))
        self.format = format.upper()
        self.avif_quality = _quality_to_avif_quality(self.quality)

    @staticmethod
    def is_avif_available() -> bool:
        return _HAS_AVIF

    @staticmethod
    def is_heif_available() -> bool:
        return _HAS_HEIF

    def compress(self, image: np.ndarray, verbose: bool = False) -> dict:
        """Compress RGB image with AVIF/HEIF."""
        from PIL import Image as PILImage

        assert image.dtype == np.uint8 and image.ndim == 3
        H, W, C = image.shape
        original_bytes = image.tobytes()
        t0 = time.time()

        # Encode with AVIF/HEIF
        buf = io.BytesIO()
        PILImage.fromarray(image).save(buf, format=self.format, quality=self.avif_quality)
        avif_data = buf.getvalue()

        # Pack recipe
        sha = hashlib.sha256(original_bytes).digest()
        recipe = self._pack_recipe(int(self.quality * 100), H, W, avif_data, sha)

        dt = time.time() - t0
        return {
            'recipe_bytes': recipe,
            'original_size': len(original_bytes),
            'recipe_size': len(recipe),
            'quality': self.quality,
            'avif_quality': self.avif_quality,
            'format': self.format,
            'train_time_s': dt,
            'throughput_mbs': len(original_bytes) / dt / 1024 / 1024 if dt > 0 else 0,
            'sha256': sha.hex(),
            'mode': 'avif_v5_26',
            'lossy': True,
        }

    def _pack_recipe(self, quality_int, H, W, avif_data, sha):
        out = bytearray()
        out += MAGIC_AVIF_V2
        out += struct.pack('<B', VERSION_AVIF)
        out += struct.pack('<B', quality_int)
        out += struct.pack('<H', H)
        out += struct.pack('<H', W)
        out += struct.pack('<I', len(avif_data))
        out += avif_data
        out += sha
        return bytes(out)

    @staticmethod
    def decompress(recipe_bytes: bytes) -> tuple[np.ndarray, dict]:
        from PIL import Image as PILImage

        buf = recipe_bytes
        off = 0
        if buf[:4] != MAGIC_AVIF_V2:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_AVIF
        quality_int = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        H = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        W = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        data_size = struct.unpack('<I', buf[off:off+4])[0]; off += 4
        avif_data = buf[off:off+data_size]; off += data_size
        sha_expected = buf[off:off+32]; off += 32

        # Decode AVIF/HEIF
        img_buf = io.BytesIO(avif_data)
        # Detect format from data (AVIF and HEIF both start with ftyp box)
        # Try AVIF first, then HEIF
        try:
            img = PILImage.open(img_buf).convert('RGB')
        except Exception:
            img_buf.seek(0)
            if _HAS_HEIF:
                img = pillow_heif.open_heif(img_buf).convert('RGB')
                img = img.to_pil()
            else:
                raise

        recovered = np.array(img, dtype=np.uint8)
        sha_got = hashlib.sha256(recovered.tobytes()).digest()

        return recovered, {
            'H': H, 'W': W,
            'quality': quality_int / 100.0,
            'format': 'AVIF/HEIF',
            'sha256_match': sha_got == sha_expected,
            'mode': 'avif_v5_26',
            'lossy': True,
        }


def _self_test():
    from PIL import Image

    print("=" * 80)
    print("BLKH v5.26 AVIF/HEIF Wrapper — Self Test")
    print("=" * 80)
    print(f"AVIF available: {_HAS_AVIF}")
    print(f"HEIF available: {_HAS_HEIF}")

    if not _HAS_AVIF:
        print("\nInstall AVIF: pip install pillow-avif-plugin")
        return

    # Test on sample photos
    print(f"\n{'Image':<18} {'ZIP':>10} {'PNG':>10} {'BLKH AVIF q=0.9':>18} {'BLKH AVIF q=0.5':>18}")
    print("-" * 80)

    import os
    import zlib
    photos_dir = '/home/z/my-project/blackhole_repo/docs/assets/sample_photos'
    for fname in sorted(os.listdir(photos_dir))[:5]:
        if not fname.endswith('.png'):
            continue
        img = np.array(Image.open(os.path.join(photos_dir, fname)).convert('RGB'))
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        zip_sz = len(zlib.compress(img.tobytes(), 9))
        png_buf = io.BytesIO()
        Image.fromarray(img).save(png_buf, format='PNG', optimize=True)
        png_sz = png_buf.tell()

        results = []
        for q in [0.9, 0.5]:
            comp = AVIFCompressor(quality=q, format='AVIF')
            res = comp.compress(img, verbose=False)
            rec, _ = AVIFCompressor.decompress(res['recipe_bytes'])
            mse = np.mean((img.astype(float) - rec.astype(float))**2)
            psnr = 10*np.log10(255**2 / max(mse, 1e-10))
            results.append((res['recipe_size'], psnr))

        print(f"{fname:<18} {zip_sz:>10,} {png_sz:>10,} "
              f"{results[0][0]:>10,}B/{results[0][1]:.0f}dB "
              f"{results[1][0]:>10,}B/{results[1][1]:.0f}dB")


if __name__ == '__main__':
    _self_test()
