# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Pytest tests for siren_v5_avif.py (BLKH v5.26 AVIF/HEIF wrapper)."""
import sys
import os
import zlib
import numpy as np
import pytest
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_inr_compressor'))
from siren_v5_avif import AVIFCompressor, _HAS_AVIF


pytestmark = pytest.mark.skipif(not _HAS_AVIF, reason="AVIF not available (pip install pillow-avif-plugin)")


def _make_smooth_image(size=128, seed=42):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(1, 5, 2)
            amp = rng.uniform(40, 80)
            phase = rng.uniform(0, 2*np.pi)
            img[:, :, c] += amp * np.sin(2*np.pi*kx*xs + phase) * np.cos(2*np.pi*ky*ys)
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


def test_avif_compress_decompress_shape():
    img = _make_smooth_image(64, seed=42)
    comp = AVIFCompressor(quality=0.9)
    res = comp.compress(img, verbose=False)
    rec, meta = AVIFCompressor.decompress(res['recipe_bytes'])
    assert rec.shape == img.shape
    assert rec.dtype == np.uint8
    assert meta['lossy']


def test_avif_beats_zip():
    img = _make_smooth_image(128, seed=42)
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    comp = AVIFCompressor(quality=0.5)
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] < zip_sz


def test_avif_quality_preserved():
    img = _make_smooth_image(64, seed=42)
    for q in [0.9, 0.5]:
        comp = AVIFCompressor(quality=q)
        res = comp.compress(img, verbose=False)
        _, meta = AVIFCompressor.decompress(res['recipe_bytes'])
        assert abs(meta['quality'] - q) < 0.01


def test_avif_magic_bytes():
    img = _make_smooth_image(64, seed=1)
    comp = AVIFCompressor(quality=0.9)
    res = comp.compress(img, verbose=False)
    assert res['recipe_bytes'][:4] == b'BLHV'


def test_avif_bad_magic_raises():
    with pytest.raises(ValueError, match="bad magic"):
        AVIFCompressor.decompress(b'BADM' + b'\x00' * 100)


def test_avif_high_quality_psnr():
    img = _make_smooth_image(128, seed=42)
    comp = AVIFCompressor(quality=1.0)
    res = comp.compress(img, verbose=False)
    rec, _ = AVIFCompressor.decompress(res['recipe_bytes'])
    mse = np.mean((img.astype(float) - rec.astype(float))**2)
    psnr = 10*np.log10(255**2 / max(mse, 1e-10))
    assert psnr > 30


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
