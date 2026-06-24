# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Pytest tests for siren_v5_gpu.py (BLKH v5.25 GPU-ready DCT)."""
import sys
import os
import zlib
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_inr_compressor'))
from siren_v5_gpu import GPUDCTCompressor, _HAS_TORCH, _HAS_CUDA
from siren_v5_dct import _HAS_SCIPY


pytestmark = pytest.mark.skipif(not (_HAS_SCIPY or _HAS_CUDA), reason="scipy or torch required")


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


def test_gpu_compressor_basic():
    """GPU compressor must produce same shape output."""
    img = _make_smooth_image(64, seed=42)
    comp = GPUDCTCompressor(quality=0.9, speed='balanced')
    res = comp.compress(img, verbose=False)
    rec, meta = GPUDCTCompressor.decompress(res['recipe_bytes'])
    assert rec.shape == img.shape
    assert rec.dtype == np.uint8
    assert meta['lossy']


def test_gpu_quality_preserved():
    """Quality should be preserved through compress/decompress."""
    img = _make_smooth_image(64, seed=42)
    for q in [0.9, 0.5]:
        comp = GPUDCTCompressor(quality=q, speed='balanced')
        res = comp.compress(img, verbose=False)
        _, meta = GPUDCTCompressor.decompress(res['recipe_bytes'])
        assert abs(meta['quality'] - q) < 0.01


def test_gpu_beats_zip():
    """GPU compressor must beat ZIP on smooth content."""
    img = _make_smooth_image(128, seed=42)
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    comp = GPUDCTCompressor(quality=0.5, speed='fast')
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] < zip_sz


def test_gpu_magic_bytes():
    """Recipe must start with BLKG magic."""
    img = _make_smooth_image(64, seed=1)
    comp = GPUDCTCompressor(quality=0.9)
    res = comp.compress(img, verbose=False)
    assert res['recipe_bytes'][:4] == b'BLKG'


def test_gpu_bad_magic_raises():
    """Decompress with bad magic must raise."""
    with pytest.raises(ValueError, match="bad magic"):
        GPUDCTCompressor.decompress(b'BADM' + b'\x00' * 100)


def test_gpu_is_gpu_available_method():
    """is_gpu_available should return bool."""
    result = GPUDCTCompressor.is_gpu_available()
    assert isinstance(result, bool)


def test_gpu_high_quality_psnr():
    """High quality must give PSNR > 25 dB."""
    img = _make_smooth_image(128, seed=42)
    comp = GPUDCTCompressor(quality=1.0, speed='balanced')
    res = comp.compress(img, verbose=False)
    rec, _ = GPUDCTCompressor.decompress(res['recipe_bytes'])
    mse = np.mean((img.astype(float) - rec.astype(float))**2)
    psnr = 10*np.log10(255**2 / max(mse, 1e-10))
    assert psnr > 25


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
