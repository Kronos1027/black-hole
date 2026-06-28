# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Pytest tests for siren_v5_wavelet_v2.py (BLKH v5.19 enhanced wavelet)."""
import sys
import os
import zlib
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_inr_compressor'))
from siren_v5_wavelet_v2 import WaveletINRCompressorV2


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


def _make_gradient_image(size=128):
    """Simple linear gradient."""
    xs = np.linspace(0, 255, size, dtype=np.float32)
    ys = np.linspace(0, 255, size, dtype=np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = xs[None, :]
    img[:, :, 1] = ys[:, None]
    img[:, :, 2] = (xs[None, :] + ys[:, None]) / 2
    return img.astype(np.uint8)


# ==================== LOSSLESS TESTS ====================

def test_wavelet_v2_lossless_bitperfect_smooth():
    """Lossless mode must be SHA-256 verified bit-perfect."""
    img = _make_smooth_image(128, seed=42)
    comp = WaveletINRCompressorV2(wavelet='db6', level=3, lossless=True)
    res = comp.compress_bitperfect(img, verbose=False)
    rec, meta = WaveletINRCompressorV2.decompress(res['recipe_bytes'])
    assert meta['sha256_match'], "Lossless mode must be bit-perfect"
    assert np.array_equal(rec, img), "Reconstruction must match original"


def test_wavelet_v2_lossless_bitperfect_gradient():
    """Lossless mode on gradient."""
    img = _make_gradient_image(128)
    comp = WaveletINRCompressorV2(wavelet='db4', level=2, lossless=True)
    res = comp.compress_bitperfect(img, verbose=False)
    rec, meta = WaveletINRCompressorV2.decompress(res['recipe_bytes'])
    assert meta['sha256_match'], "Lossless mode must be bit-perfect on gradient"
    assert np.array_equal(rec, img)


def test_wavelet_v2_lossless_adaptive():
    """Adaptive wavelet selection must produce bit-perfect output."""
    img = _make_smooth_image(128, seed=123)
    comp = WaveletINRCompressorV2(wavelet='auto', level='auto', lossless=True)
    res = comp.compress_bitperfect(img, verbose=False)
    rec, meta = WaveletINRCompressorV2.decompress(res['recipe_bytes'])
    assert meta['sha256_match']
    assert np.array_equal(rec, img)


def test_wavelet_v2_lossless_multiple_wavelets():
    """Test that multiple wavelets work in lossless mode."""
    img = _make_smooth_image(64, seed=7)
    for wl in ['db4', 'db6', 'bior4.4', 'haar', 'sym4']:
        comp = WaveletINRCompressorV2(wavelet=wl, level=2, lossless=True)
        res = comp.compress_bitperfect(img, verbose=False)
        rec, meta = WaveletINRCompressorV2.decompress(res['recipe_bytes'])
        assert meta['sha256_match'], f"Wavelet {wl} must be bit-perfect"


def test_wavelet_v2_lossless_multiple_levels():
    """Test that multiple decomposition levels work in lossless mode."""
    img = _make_smooth_image(128, seed=99)
    for level in [2, 3, 4]:
        comp = WaveletINRCompressorV2(wavelet='db6', level=level, lossless=True)
        res = comp.compress_bitperfect(img, verbose=False)
        rec, meta = WaveletINRCompressorV2.decompress(res['recipe_bytes'])
        assert meta['sha256_match'], f"Level {level} must be bit-perfect"


# ==================== LOSSY TESTS ====================

def test_wavelet_v2_lossy_reconstructs():
    """Lossy mode must produce same shape output."""
    img = _make_smooth_image(128, seed=42)
    comp = WaveletINRCompressorV2(wavelet='db6', level=3, lossless=False, quality=1.0)
    res = comp.compress_bitperfect(img, verbose=False)
    rec, meta = WaveletINRCompressorV2.decompress(res['recipe_bytes'])
    assert rec.shape == img.shape
    assert rec.dtype == np.uint8
    assert not meta['lossless']


def test_wavelet_v2_lossy_high_quality_psnr():
    """Lossy mode at quality=1.0 must give PSNR > 50 dB on smooth images."""
    img = _make_smooth_image(128, seed=42)
    comp = WaveletINRCompressorV2(wavelet='db6', level=3, lossless=False, quality=1.0)
    res = comp.compress_bitperfect(img, verbose=False)
    rec, meta = WaveletINRCompressorV2.decompress(res['recipe_bytes'])
    mse = np.mean((img.astype(float) - rec.astype(float))**2)
    psnr = 10*np.log10(255**2 / max(mse, 1e-10))
    assert psnr > 50, f"PSNR {psnr:.1f} dB < 50 dB"


def test_wavelet_v2_lossy_compression_better_than_zip():
    """Lossy mode at moderate quality must beat ZIP on smooth images."""
    img = _make_smooth_image(256, seed=42)
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    comp = WaveletINRCompressorV2(wavelet='db6', level=3, lossless=False,
                                    quality=0.5, threshold=5.0)
    res = comp.compress_bitperfect(img, verbose=False)
    assert res['recipe_size'] < zip_sz, \
        f"Lossy mode ({res['recipe_size']}B) must beat ZIP ({zip_sz}B)"


def test_wavelet_v2_lossy_threshold_improves_compression():
    """Higher threshold must produce smaller files."""
    img = _make_smooth_image(128, seed=42)
    sizes = []
    for th in [0, 3, 7]:
        comp = WaveletINRCompressorV2(wavelet='db6', level=3, lossless=False,
                                        quality=0.5, threshold=th)
        res = comp.compress_bitperfect(img, verbose=False)
        sizes.append(res['recipe_size'])
    assert sizes[1] <= sizes[0], "Threshold>0 must compress better"
    assert sizes[2] <= sizes[1], "Higher threshold must compress better"


# ==================== ADAPTIVE SELECTION TESTS ====================

def test_wavelet_v2_adaptive_picks_valid_wavelet():
    """Adaptive mode must pick a valid wavelet from the candidate list."""
    img = _make_smooth_image(128, seed=42)
    comp = WaveletINRCompressorV2(wavelet='auto', level='auto', lossless=True)
    res = comp.compress_bitperfect(img, verbose=False)
    assert res['wavelet'] in {w for w, _ in [
        ('bior4.4', 3), ('db4', 3), ('db6', 3), ('sym4', 3), ('coif2', 3),
        ('bior4.4', 2), ('db4', 2), ('db6', 2), ('bior4.4', 4), ('db4', 4),
        ('haar', 3),
    ]}
    assert res['level'] in [2, 3, 4]


def test_wavelet_v2_adaptive_picks_smallest():
    """Adaptive mode must pick a wavelet that's no worse than db6/L3."""
    img = _make_smooth_image(128, seed=42)
    # Adaptive
    comp_a = WaveletINRCompressorV2(wavelet='auto', level='auto', lossless=True)
    res_a = comp_a.compress_bitperfect(img, verbose=False)
    # Fixed db6 L3
    comp_f = WaveletINRCompressorV2(wavelet='db6', level=3, lossless=True)
    res_f = comp_f.compress_bitperfect(img, verbose=False)
    assert res_a['recipe_size'] <= res_f['recipe_size'] + 50, \
        f"Adaptive ({res_a['recipe_size']}B) should be <= db6/L3 ({res_f['recipe_size']}B)"


# ==================== FORMAT TESTS ====================

def test_wavelet_v2_magic_bytes():
    """Recipe must start with BLK2 magic."""
    img = _make_smooth_image(64, seed=1)
    comp = WaveletINRCompressorV2(wavelet='db6', level=2, lossless=True)
    res = comp.compress_bitperfect(img, verbose=False)
    assert res['recipe_bytes'][:4] == b'BLK2'


def test_wavelet_v2_bad_magic_raises():
    """Decompress with bad magic must raise."""
    with pytest.raises(ValueError, match="bad magic"):
        WaveletINRCompressorV2.decompress(b'BADM' + b'\x00' * 100)


def test_wavelet_v2_recipe_smaller_than_raw():
    """Compressed recipe must be smaller than raw image bytes (lossy mode)."""
    img = _make_smooth_image(128, seed=42)
    comp = WaveletINRCompressorV2(wavelet='db6', level=3, lossless=False,
                                    quality=0.5, threshold=5.0)
    res = comp.compress_bitperfect(img, verbose=False)
    assert res['recipe_size'] < img.nbytes, \
        f"Recipe ({res['recipe_size']}B) must be smaller than raw ({img.nbytes}B)"


# ==================== REGRESSION TEST ====================

def test_wavelet_v2_beats_v5_18_psnr():
    """v5.19 must produce better PSNR than v5.18 (which was broken)."""
    img = _make_smooth_image(128, seed=42)
    # v5.19 lossy at q=1.0
    comp = WaveletINRCompressorV2(wavelet='db6', level=3, lossless=False, quality=1.0)
    res = comp.compress_bitperfect(img, verbose=False)
    rec, _ = WaveletINRCompressorV2.decompress(res['recipe_bytes'])
    mse = np.mean((img.astype(float) - rec.astype(float))**2)
    psnr_v519 = 10*np.log10(255**2 / max(mse, 1e-10))
    # v5.19 must produce at least 30 dB (v5.18 was 4-12 dB)
    assert psnr_v519 > 30, f"v5.19 PSNR {psnr_v519:.1f} dB must be > 30 dB (v5.18 was 4-12 dB)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
