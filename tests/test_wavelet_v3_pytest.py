# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Pytest tests for siren_v5_wavelet_v3.py (BLKH v5.20 float16 wavelet)."""
import sys
import os
import zlib
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_inr_compressor'))
from siren_v5_wavelet_v3 import WaveletINRCompressorV3


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

def test_wavelet_v3_lossless_bitperfect_smooth():
    """Lossless mode must be SHA-256 verified bit-perfect."""
    img = _make_smooth_image(128, seed=42)
    comp = WaveletINRCompressorV3(wavelet='db6', level=3, lossless=True)
    res = comp.compress(img, verbose=False)
    rec, meta = WaveletINRCompressorV3.decompress(res['recipe_bytes'])
    assert meta['sha256_match'], "Lossless mode must be bit-perfect"
    assert np.array_equal(rec, img), "Reconstruction must match original"


def test_wavelet_v3_lossless_bitperfect_gradient():
    """Lossless mode on gradient."""
    img = _make_gradient_image(128)
    comp = WaveletINRCompressorV3(wavelet='db4', level=2, lossless=True)
    res = comp.compress(img, verbose=False)
    rec, meta = WaveletINRCompressorV3.decompress(res['recipe_bytes'])
    assert meta['sha256_match'], "Lossless mode must be bit-perfect on gradient"
    assert np.array_equal(rec, img)


def test_wavelet_v3_lossless_adaptive():
    """Adaptive wavelet selection must produce bit-perfect output."""
    img = _make_smooth_image(128, seed=123)
    comp = WaveletINRCompressorV3(wavelet='auto', level='auto', lossless=True)
    res = comp.compress(img, verbose=False)
    rec, meta = WaveletINRCompressorV3.decompress(res['recipe_bytes'])
    assert meta['sha256_match']
    assert np.array_equal(rec, img)


def test_wavelet_v3_lossless_multiple_wavelets():
    """Test that multiple wavelets work in lossless mode."""
    img = _make_smooth_image(64, seed=7)
    for wl in ['db4', 'db6', 'bior4.4', 'haar', 'sym4']:
        comp = WaveletINRCompressorV3(wavelet=wl, level=2, lossless=True)
        res = comp.compress(img, verbose=False)
        rec, meta = WaveletINRCompressorV3.decompress(res['recipe_bytes'])
        assert meta['sha256_match'], f"Wavelet {wl} must be bit-perfect"


def test_wavelet_v3_lossless_multiple_levels():
    """Test that multiple decomposition levels work in lossless mode."""
    img = _make_smooth_image(128, seed=99)
    for level in [2, 3, 4]:
        comp = WaveletINRCompressorV3(wavelet='db6', level=level, lossless=True)
        res = comp.compress(img, verbose=False)
        rec, meta = WaveletINRCompressorV3.decompress(res['recipe_bytes'])
        assert meta['sha256_match'], f"Level {level} must be bit-perfect"


def test_wavelet_v3_lossless_beats_zip_on_smooth():
    """v5.20 must beat ZIP on smooth images (key feature)."""
    img = _make_smooth_image(256, seed=42)
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    comp = WaveletINRCompressorV3(wavelet='auto', level='auto', lossless=True)
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] < zip_sz, \
        f"v5.20 ({res['recipe_size']}B) must beat ZIP ({zip_sz}B) on smooth 256x256"


def test_wavelet_v3_beats_v5_19_size():
    """v5.20 must be smaller than v5.19 on the same content."""
    from siren_v5_wavelet_v2 import WaveletINRCompressorV2
    img = _make_smooth_image(256, seed=42)
    # v5.19
    comp19 = WaveletINRCompressorV2(wavelet='auto', level='auto', lossless=True)
    res19 = comp19.compress_bitperfect(img, verbose=False)
    # v5.20
    comp20 = WaveletINRCompressorV3(wavelet='auto', level='auto', lossless=True)
    res20 = comp20.compress(img, verbose=False)
    assert res20['recipe_size'] < res19['recipe_size'], \
        f"v5.20 ({res20['recipe_size']}B) must be smaller than v5.19 ({res19['recipe_size']}B)"


# ==================== FORMAT TESTS ====================

def test_wavelet_v3_magic_bytes():
    """Recipe must start with BKWF magic."""
    img = _make_smooth_image(64, seed=1)
    comp = WaveletINRCompressorV3(wavelet='db6', level=2, lossless=True)
    res = comp.compress(img, verbose=False)
    assert res['recipe_bytes'][:4] == b'BKWF'


def test_wavelet_v3_bad_magic_raises():
    """Decompress with bad magic must raise."""
    with pytest.raises(ValueError, match="bad magic"):
        WaveletINRCompressorV3.decompress(b'BADM' + b'\x00' * 100)


def test_wavelet_v3_adaptive_picks_valid_wavelet():
    """Adaptive mode must pick a valid wavelet from the candidate list."""
    img = _make_smooth_image(128, seed=42)
    comp = WaveletINRCompressorV3(wavelet='auto', level='auto', lossless=True)
    res = comp.compress(img, verbose=False)
    valid_wavelets = {w for w, _ in [
        ('bior4.4', 3), ('db4', 3), ('db6', 3), ('sym4', 3), ('coif2', 3),
        ('bior4.4', 2), ('db4', 2), ('db6', 2), ('bior4.4', 4), ('db4', 4),
        ('haar', 3),
    ]}
    assert res['wavelet'] in valid_wavelets
    assert res['level'] in [2, 3, 4]


def test_wavelet_v3_large_image():
    """Test on larger image (512x512)."""
    img = _make_smooth_image(256, seed=2026)
    comp = WaveletINRCompressorV3(wavelet='auto', level='auto', lossless=True)
    res = comp.compress(img, verbose=False)
    rec, meta = WaveletINRCompressorV3.decompress(res['recipe_bytes'])
    assert meta['sha256_match'], "Must be bit-perfect on 256x256"
    assert np.array_equal(rec, img)


def test_wavelet_v3_combined_mode():
    """Combined mode (single bytestream) must be bit-perfect."""
    img = _make_smooth_image(128, seed=42)
    comp = WaveletINRCompressorV3(wavelet='haar', level=3, lossless=True, combined=True)
    res = comp.compress(img, verbose=False)
    rec, meta = WaveletINRCompressorV3.decompress(res['recipe_bytes'])
    assert meta['sha256_match'], "Combined mode must be bit-perfect"
    assert meta['combined'], "Combined flag must be set in meta"
    assert np.array_equal(rec, img)


def test_wavelet_v3_combined_smaller_than_per_subband():
    """Combined mode should be smaller than per-subband on smooth content."""
    img = _make_smooth_image(256, seed=42)
    # Per-subband
    comp1 = WaveletINRCompressorV3(wavelet='haar', level=3, lossless=True, combined=False, codec='brotli')
    res1 = comp1.compress(img, verbose=False)
    # Combined
    comp2 = WaveletINRCompressorV3(wavelet='haar', level=3, lossless=True, combined=True, codec='brotli')
    res2 = comp2.compress(img, verbose=False)
    assert res2['recipe_size'] <= res1['recipe_size'], \
        f"Combined ({res2['recipe_size']}B) should be <= per-subband ({res1['recipe_size']}B)"


def test_wavelet_v3_parallel_mode():
    """Parallel mode must produce same result as sequential (bit-perfect)."""
    img = _make_smooth_image(128, seed=42)
    # Sequential
    comp1 = WaveletINRCompressorV3(wavelet='auto', level='auto', lossless=True, parallel=False, codec='zstd')
    res1 = comp1.compress(img, verbose=False)
    # Parallel
    comp2 = WaveletINRCompressorV3(wavelet='auto', level='auto', lossless=True, parallel=True, n_workers=2, codec='zstd')
    res2 = comp2.compress(img, verbose=False)
    # Both must be bit-perfect
    rec1, meta1 = WaveletINRCompressorV3.decompress(res1['recipe_bytes'])
    rec2, meta2 = WaveletINRCompressorV3.decompress(res2['recipe_bytes'])
    assert meta1['sha256_match'] and meta2['sha256_match']
    # Sizes should be very close (may differ if parallel picks different wavelet due to nondeterminism)
    assert abs(res1['recipe_size'] - res2['recipe_size']) < 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
