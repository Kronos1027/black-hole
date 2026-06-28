# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Pytest tests for siren_v5_fast.py (BLKH v5.23 fast DCT)."""
import sys
import os
import zlib
import numpy as np
import pytest
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_inr_compressor'))
from siren_v5_fast import FastDCTCompressor, SPEED_FAST, SPEED_BALANCED, SPEED_BEST
from siren_v5_dct import _HAS_SCIPY


pytestmark = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy required")


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


def test_fast_compress_decompress_shape():
    """Fast compressor must produce same shape output."""
    img = _make_smooth_image(64, seed=42)
    for speed in ['fast', 'balanced', 'best']:
        comp = FastDCTCompressor(quality=0.9, speed=speed)
        res = comp.compress(img, verbose=False)
        rec, meta = FastDCTCompressor.decompress(res['recipe_bytes'])
        assert rec.shape == img.shape
        assert rec.dtype == np.uint8
        assert meta['lossy']


def test_fast_beats_zip_on_smooth():
    """Fast mode must beat ZIP on smooth content."""
    img = _make_smooth_image(128, seed=42)
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    comp = FastDCTCompressor(quality=0.5, speed='fast')
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] < zip_sz


def test_fast_mode_faster_than_best():
    """Fast mode should be faster than best mode."""
    import time
    img = _make_smooth_image(128, seed=42)
    # Fast
    comp = FastDCTCompressor(quality=0.9, speed='fast')
    t0 = time.time()
    for _ in range(20):
        res = comp.compress(img, verbose=False)
    fast_time = time.time() - t0
    # Best
    comp = FastDCTCompressor(quality=0.9, speed='best')
    t0 = time.time()
    for _ in range(20):
        res = comp.compress(img, verbose=False)
    best_time = time.time() - t0
    assert fast_time < best_time, f"fast ({fast_time:.3f}s) should be < best ({best_time:.3f}s)"


def test_fast_best_smaller_than_fast():
    """Best mode should produce smaller files than fast mode."""
    img = _make_smooth_image(128, seed=42)
    sizes = []
    for speed in ['fast', 'balanced', 'best']:
        comp = FastDCTCompressor(quality=0.9, speed=speed)
        res = comp.compress(img, verbose=False)
        sizes.append(res['recipe_size'])
    # Best should be smallest (or equal)
    assert sizes[2] <= sizes[0]
    assert sizes[1] <= sizes[0]


def test_fast_quality_preserved():
    """Quality setting should be preserved through compress/decompress."""
    img = _make_smooth_image(64, seed=42)
    for q in [0.9, 0.5, 0.25]:
        comp = FastDCTCompressor(quality=q, speed='fast')
        res = comp.compress(img, verbose=False)
        _, meta = FastDCTCompressor.decompress(res['recipe_bytes'])
        assert abs(meta['quality'] - q) < 0.01


def test_fast_speed_preserved():
    """Speed setting should be preserved through compress/decompress."""
    img = _make_smooth_image(64, seed=42)
    for speed in ['fast', 'balanced', 'best']:
        comp = FastDCTCompressor(quality=0.9, speed=speed)
        res = comp.compress(img, verbose=False)
        _, meta = FastDCTCompressor.decompress(res['recipe_bytes'])
        assert meta['speed'] == speed


def test_fast_magic_bytes():
    """Recipe must start with BLKF magic."""
    img = _make_smooth_image(64, seed=1)
    comp = FastDCTCompressor(quality=0.9, speed='fast')
    res = comp.compress(img, verbose=False)
    assert res['recipe_bytes'][:4] == b'BLKF'


def test_fast_bad_magic_raises():
    """Decompress with bad magic must raise."""
    with pytest.raises(ValueError, match="bad magic"):
        FastDCTCompressor.decompress(b'BADM' + b'\x00' * 100)


def test_fast_high_quality_psnr():
    """High quality fast mode must give PSNR > 25 dB."""
    img = _make_smooth_image(128, seed=42)
    comp = FastDCTCompressor(quality=1.0, speed='fast')
    res = comp.compress(img, verbose=False)
    rec, _ = FastDCTCompressor.decompress(res['recipe_bytes'])
    mse = np.mean((img.astype(float) - rec.astype(float))**2)
    psnr = 10*np.log10(255**2 / max(mse, 1e-10))
    assert psnr > 25


def test_fast_real_image():
    """Test on real photo."""
    path = '/home/z/my-project/blackhole_repo/docs/assets/sample_photos/sky_128.png'
    if not os.path.exists(path):
        pytest.skip("Sample photo not available")
    img = np.array(Image.open(path).convert('RGB'))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    comp = FastDCTCompressor(quality=0.9, speed='fast')
    res = comp.compress(img, verbose=False)
    rec, _ = FastDCTCompressor.decompress(res['recipe_bytes'])
    assert rec.shape == img.shape
    mse = np.mean((img.astype(float) - rec.astype(float))**2)
    psnr = 10*np.log10(255**2 / max(mse, 1e-10))
    assert psnr > 25


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
