# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Pytest tests for siren_v5_rle.py (BLKH v5.28 DCT+RLE)."""
import sys
import os
import zlib
import numpy as np
import pytest
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_inr_compressor'))
from siren_v5_rle import RLEDCTCompressor, _rle_encode_zigzag, _rle_decode_zigzag, ZIGZAG, ZIGZAG_INV
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


def test_zigzag_inverse():
    """Zigzag and inverse zigzag should be perfect inverses."""
    arr = np.arange(64, dtype=np.int32)
    zigzagged = arr[ZIGZAG]
    recovered = zigzagged[ZIGZAG_INV]
    assert np.array_equal(arr, recovered)


def test_rle_roundtrip():
    """RLE encode/decode should be lossless."""
    # Create test data with many zeros (typical after DCT quantization)
    blocks = np.zeros((4, 4, 8, 8), dtype=np.int16)
    blocks[0, 0, 0, 0] = 100  # DC
    blocks[0, 0, 1, 0] = 50   # low-freq
    blocks[1, 1, 0, 0] = 80
    blocks[1, 1, 3, 4] = 20   # mid-freq
    # Rest are zeros

    rle = _rle_encode_zigzag(blocks)
    decoded = _rle_decode_zigzag(rle, 4, 4)
    assert np.array_equal(blocks, decoded), "RLE roundtrip must be lossless"


def test_rle_compress_decompress_shape():
    img = _make_smooth_image(64, seed=42)
    comp = RLEDCTCompressor(quality=0.9, speed='balanced')
    res = comp.compress(img, verbose=False)
    rec, meta = RLEDCTCompressor.decompress(res['recipe_bytes'])
    assert rec.shape == img.shape
    assert rec.dtype == np.uint8
    assert meta['lossy']


def test_rle_beats_zip():
    img = _make_smooth_image(128, seed=42)
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    comp = RLEDCTCompressor(quality=0.5, speed='fast')
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] < zip_sz


def test_rle_quality_preserved():
    img = _make_smooth_image(64, seed=42)
    for q in [0.9, 0.5]:
        comp = RLEDCTCompressor(quality=q, speed='fast')
        res = comp.compress(img, verbose=False)
        _, meta = RLEDCTCompressor.decompress(res['recipe_bytes'])
        assert abs(meta['quality'] - q) < 0.01


def test_rle_magic_bytes():
    img = _make_smooth_image(64, seed=1)
    comp = RLEDCTCompressor(quality=0.9)
    res = comp.compress(img, verbose=False)
    assert res['recipe_bytes'][:4] == b'BLKR'


def test_rle_bad_magic_raises():
    with pytest.raises(ValueError, match="bad magic"):
        RLEDCTCompressor.decompress(b'BADM' + b'\x00' * 100)


def test_rle_high_quality_psnr():
    img = _make_smooth_image(128, seed=42)
    comp = RLEDCTCompressor(quality=1.0, speed='balanced')
    res = comp.compress(img, verbose=False)
    rec, _ = RLEDCTCompressor.decompress(res['recipe_bytes'])
    mse = np.mean((img.astype(float) - rec.astype(float))**2)
    psnr = 10*np.log10(255**2 / max(mse, 1e-10))
    assert psnr > 25


def test_rle_smaller_or_equal_to_v522():
    """v5.28 RLE should be smaller or equal to v5.22 on most images."""
    from siren_v5_dct import DCTCompressor
    img = _make_smooth_image(128, seed=42)
    comp22 = DCTCompressor(quality=0.9, codec='brotli')
    res22 = comp22.compress(img, verbose=False)
    comp28 = RLEDCTCompressor(quality=0.9, speed='best')
    res28 = comp28.compress(img, verbose=False)
    # RLE should not be significantly larger (allow 5% overhead for noisy images)
    assert res28['recipe_size'] <= res22['recipe_size'] * 1.05


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
