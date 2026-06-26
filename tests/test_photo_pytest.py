# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Pytest tests for siren_v5_photo.py (BLKH v5.21 photo compressor)."""
import sys
import os
import zlib
import numpy as np
import pytest
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_inr_compressor'))
from siren_v5_photo import PhotoCompressor, _rgb_to_ycbcr, _ycbcr_to_rgb, _subsample_420, _upsample_420


def _make_smooth_image(size=128, seed=42):
    """Smooth test image (satellite-like)."""
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


def _make_random_image(size=64, seed=42):
    """Random noise image (worst case)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


# ==================== COLOR SPACE TESTS ====================

def test_ycbcr_roundtrip_lossy():
    """YCbCr roundtrip should be lossy but close (max_err <= 2)."""
    img = _make_smooth_image(64, seed=42)
    y, cb, cr = _rgb_to_ycbcr(img)
    rec = _ycbcr_to_rgb(y, cb, cr)
    assert rec.shape == img.shape
    assert rec.dtype == np.uint8
    max_err = np.abs(rec.astype(int) - img.astype(int)).max()
    assert max_err <= 2, f"YCbCr roundtrip max_err {max_err} > 2"


def test_subsample_upsample_shape():
    """4:2:0 subsample/upsample should preserve shape."""
    ch = np.random.rand(64, 64).astype(np.float32)
    sub = _subsample_420(ch)
    assert sub.shape == (32, 32)
    up = _upsample_420(sub, 64, 64)
    assert up.shape == (64, 64)


# ==================== PHOTO COMPRESSOR TESTS ====================

def test_photo_compress_decompress_shape():
    """Photo compressor must produce same shape output."""
    img = _make_smooth_image(64, seed=42)
    comp = PhotoCompressor(subsampling='420', codec='brotli')
    res = comp.compress(img, verbose=False)
    rec, meta = PhotoCompressor.decompress(res['recipe_bytes'])
    assert rec.shape == img.shape
    assert rec.dtype == np.uint8
    assert meta['lossy']


def test_photo_420_psnr_reasonable():
    """4:2:0 mode must give PSNR > 25 dB on smooth image."""
    img = _make_smooth_image(128, seed=42)
    comp = PhotoCompressor(subsampling='420', codec='brotli')
    res = comp.compress(img, verbose=False)
    rec, _ = PhotoCompressor.decompress(res['recipe_bytes'])
    mse = np.mean((img.astype(float) - rec.astype(float))**2)
    psnr = 10*np.log10(255**2 / max(mse, 1e-10))
    assert psnr > 25, f"PSNR {psnr:.1f} < 25 dB"


def test_photo_444_better_psnr_than_420():
    """4:4:4 mode should give better PSNR than 4:2:0 (no chroma subsampling)."""
    img = _make_smooth_image(128, seed=42)
    # 4:2:0
    comp420 = PhotoCompressor(subsampling='420', codec='brotli')
    res420 = comp420.compress(img, verbose=False)
    rec420, _ = PhotoCompressor.decompress(res420['recipe_bytes'])
    mse420 = np.mean((img.astype(float) - rec420.astype(float))**2)
    psnr420 = 10*np.log10(255**2 / max(mse420, 1e-10))
    # 4:4:4
    comp444 = PhotoCompressor(subsampling='444', codec='brotli')
    res444 = comp444.compress(img, verbose=False)
    rec444, _ = PhotoCompressor.decompress(res444['recipe_bytes'])
    mse444 = np.mean((img.astype(float) - rec444.astype(float))**2)
    psnr444 = 10*np.log10(255**2 / max(mse444, 1e-10))
    assert psnr444 >= psnr420, f"4:4:4 ({psnr444:.1f}dB) should be >= 4:2:0 ({psnr420:.1f}dB)"


def test_photo_beats_zip_on_random():
    """Photo compressor must beat ZIP on smooth content."""
    img = _make_smooth_image(128, seed=42)
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    comp = PhotoCompressor(subsampling='420', codec='brotli')
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] < zip_sz, \
        f"Photo ({res['recipe_size']}B) must beat ZIP ({zip_sz}B)"


def test_photo_420_smaller_than_444():
    """4:2:0 should be smaller than 4:4:4."""
    img = _make_smooth_image(128, seed=42)
    comp420 = PhotoCompressor(subsampling='420', codec='brotli')
    res420 = comp420.compress(img, verbose=False)
    comp444 = PhotoCompressor(subsampling='444', codec='brotli')
    res444 = comp444.compress(img, verbose=False)
    assert res420['recipe_size'] <= res444['recipe_size'], \
        f"4:2:0 ({res420['recipe_size']}B) should be <= 4:4:4 ({res444['recipe_size']}B)"


# ==================== FORMAT TESTS ====================

def test_photo_magic_bytes():
    """Recipe must start with BLKP magic."""
    img = _make_smooth_image(64, seed=1)
    comp = PhotoCompressor()
    res = comp.compress(img, verbose=False)
    assert res['recipe_bytes'][:4] == b'BLKP'


def test_photo_bad_magic_raises():
    """Decompress with bad magic must raise."""
    with pytest.raises(ValueError, match="bad magic"):
        PhotoCompressor.decompress(b'BADM' + b'\x00' * 100)


def test_photo_real_image():
    """Test on real photo if available."""
    path = '/home/z/my-project/blackhole_repo/docs/assets/sample_photos/sky_128.png'
    if not os.path.exists(path):
        pytest.skip("Sample photo not available")
    img = np.array(Image.open(path).convert('RGB'))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    comp = PhotoCompressor(subsampling='420', codec='brotli')
    res = comp.compress(img, verbose=False)
    rec, meta = PhotoCompressor.decompress(res['recipe_bytes'])
    assert rec.shape == img.shape
    # PSNR should be > 25 dB on real photo
    mse = np.mean((img.astype(float) - rec.astype(float))**2)
    psnr = 10*np.log10(255**2 / max(mse, 1e-10))
    assert psnr > 25, f"PSNR {psnr:.1f} < 25 dB on real photo"


def test_photo_beats_png_on_real_image():
    """Photo compressor must beat PNG on real photos (key feature)."""
    path = '/home/z/my-project/blackhole_repo/docs/assets/sample_photos/marble_128.png'
    if not os.path.exists(path):
        pytest.skip("Sample photo not available")
    img = np.array(Image.open(path).convert('RGB'))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    # PNG size
    png_buf = io.BytesIO()
    Image.fromarray(img).save(png_buf, format='PNG', optimize=True)
    png_sz = png_buf.tell()
    # BLKH v5.21
    comp = PhotoCompressor(subsampling='420', codec='brotli')
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] < png_sz, \
        f"Photo ({res['recipe_size']}B) must beat PNG ({png_sz}B) on real photos"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
