# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Pytest tests for siren_v5_dct.py (BLKH v5.22 DCT compressor)."""
import sys
import os
import zlib
import numpy as np
import pytest
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_inr_compressor'))
from siren_v5_dct import DCTCompressor, _quality_to_q_scale, _HAS_SCIPY


pytestmark = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy required for DCT mode")


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


# ==================== QUALITY MAPPING TESTS ====================

def test_quality_to_q_scale_monotonic():
    """Higher quality should give lower q_scale (less quantization)."""
    q1 = _quality_to_q_scale(1.0)
    q05 = _quality_to_q_scale(0.5)
    q01 = _quality_to_q_scale(0.1)
    assert q1 < q05 < q01, f"q_scales should be monotonic: {q1}, {q05}, {q01}"


def test_quality_to_q_scale_range():
    """q_scale should be in reasonable range."""
    assert 0.1 <= _quality_to_q_scale(1.0) <= 1.0
    assert 1.0 <= _quality_to_q_scale(0.5) <= 30.0
    assert 10.0 <= _quality_to_q_scale(0.1) <= 30.0


# ==================== COMPRESSOR TESTS ====================

def test_dct_compress_decompress_shape():
    """DCT compressor must produce same shape output."""
    img = _make_smooth_image(64, seed=42)
    comp = DCTCompressor(quality=0.9)
    res = comp.compress(img, verbose=False)
    rec, meta = DCTCompressor.decompress(res['recipe_bytes'])
    assert rec.shape == img.shape
    assert rec.dtype == np.uint8
    assert meta['lossy']


def test_dct_high_quality_psnr():
    """High quality (q=1.0) must give PSNR > 30 dB."""
    img = _make_smooth_image(128, seed=42)
    comp = DCTCompressor(quality=1.0)
    res = comp.compress(img, verbose=False)
    rec, _ = DCTCompressor.decompress(res['recipe_bytes'])
    mse = np.mean((img.astype(float) - rec.astype(float))**2)
    psnr = 10*np.log10(255**2 / max(mse, 1e-10))
    assert psnr > 30, f"PSNR {psnr:.1f} < 30 dB"


def test_dct_lower_quality_smaller_size():
    """Lower quality should produce smaller file."""
    img = _make_smooth_image(128, seed=42)
    sizes = []
    for q in [1.0, 0.5, 0.25]:
        comp = DCTCompressor(quality=q)
        res = comp.compress(img, verbose=False)
        sizes.append(res['recipe_size'])
    assert sizes[0] >= sizes[1] >= sizes[2], f"Sizes should decrease with quality: {sizes}"


def test_dct_beats_zip():
    """DCT must beat ZIP on smooth content."""
    img = _make_smooth_image(128, seed=42)
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    comp = DCTCompressor(quality=0.5)
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] < zip_sz, "DCT %dB must beat ZIP %dB" % (res['recipe_size'], zip_sz)


def test_dct_beats_png_on_real_photo():
    """DCT must beat PNG significantly on real photos."""
    path = '/home/z/my-project/blackhole_repo/docs/assets/sample_photos/marble_128.png'
    if not os.path.exists(path):
        pytest.skip("Sample photo not available")
    img = np.array(Image.open(path).convert('RGB'))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    png_buf = io.BytesIO()
    Image.fromarray(img).save(png_buf, format='PNG', optimize=True)
    png_sz = png_buf.tell()
    comp = DCTCompressor(quality=0.9)
    res = comp.compress(img, verbose=False)
    # DCT should be at least 5x smaller than PNG
    assert res['recipe_size'] * 5 < png_sz, "DCT %dB should be 5x+ smaller than PNG %dB" % (res['recipe_size'], png_sz)


# ==================== FORMAT TESTS ====================

def test_dct_magic_bytes():
    """Recipe must start with BLKD magic."""
    img = _make_smooth_image(64, seed=1)
    comp = DCTCompressor(quality=0.9)
    res = comp.compress(img, verbose=False)
    assert res['recipe_bytes'][:4] == b'BLKD'


def test_dct_bad_magic_raises():
    """Decompress with bad magic must raise."""
    with pytest.raises(ValueError, match="bad magic"):
        DCTCompressor.decompress(b'BADM' + b'\x00' * 100)


def test_dct_recipe_smaller_than_raw():
    """Compressed recipe must be smaller than raw image bytes."""
    img = _make_smooth_image(128, seed=42)
    comp = DCTCompressor(quality=0.5)
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] < img.nbytes, "Recipe %dB must be smaller than raw %dB" % (res['recipe_size'], img.nbytes)


def test_dct_quality_preserved_in_recipe():
    """Quality setting should be preserved through compress/decompress cycle."""
    img = _make_smooth_image(64, seed=42)
    for q in [1.0, 0.5, 0.25]:
        comp = DCTCompressor(quality=q)
        res = comp.compress(img, verbose=False)
        _, meta = DCTCompressor.decompress(res['recipe_bytes'])
        assert abs(meta['quality'] - q) < 0.01, f"Quality {q} not preserved: got {meta['quality']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
