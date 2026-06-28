# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Pytest tests for siren_v5_palette.py (BLKH v5.29 palette compressor)."""
import sys
import os
import zlib
import numpy as np
import pytest
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_inr_compressor'))
from siren_v5_palette import PaletteCompressor, MAX_PALETTE_COLORS


def _make_palette_image(size=128, n_colors=16, seed=42):
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, (n_colors, 3), dtype=np.uint8)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    y, x = np.mgrid[0:size, 0:size]
    region = (x // 16 + y // 16) % n_colors
    for c in range(3):
        img[:, :, c] = palette[region, c]
    return img


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


def test_palette_compress_decompress_shape():
    img = _make_palette_image(64, 16)
    comp = PaletteCompressor(max_colors=256, speed='balanced')
    res = comp.compress(img, verbose=False)
    rec, meta = PaletteCompressor.decompress(res['recipe_bytes'])
    assert rec.shape == img.shape
    assert rec.dtype == np.uint8


def test_palette_bit_perfect():
    """Palette mode must be TRUE bit-perfect (lossless)."""
    img = _make_palette_image(128, 16)
    comp = PaletteCompressor(max_colors=256, speed='balanced')
    res = comp.compress(img, verbose=False)
    rec, meta = PaletteCompressor.decompress(res['recipe_bytes'])
    assert meta['sha256_match'], "Palette must be bit-perfect"
    assert meta['lossless'], "Palette is lossless"
    assert np.array_equal(rec, img)


def test_palette_beats_zip():
    img = _make_palette_image(128, 16)
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    comp = PaletteCompressor(max_colors=256, speed='balanced')
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] < zip_sz


def test_palette_should_use_palette_true():
    img = _make_palette_image(128, 16)
    assert PaletteCompressor.should_use_palette(img, threshold=256)


def test_palette_should_use_palette_false():
    """Natural photos have too many colors for palette."""
    img = _make_smooth_image(128)
    # Smooth synthetic might have many unique colors
    assert not PaletteCompressor.should_use_palette(img, threshold=16)


def test_palette_magic_bytes():
    img = _make_palette_image(64, 8)
    comp = PaletteCompressor()
    res = comp.compress(img, verbose=False)
    assert res['recipe_bytes'][:4] == b'BLKQ'


def test_palette_bad_magic_raises():
    with pytest.raises(ValueError, match="bad magic"):
        PaletteCompressor.decompress(b'BADM' + b'\x00' * 100)


def test_palette_too_many_colors_raises():
    """Image with > max_colors should raise."""
    img = _make_smooth_image(128)  # many colors
    comp = PaletteCompressor(max_colors=8)
    with pytest.raises(ValueError, match="unique colors"):
        comp.compress(img, verbose=False)


def test_palette_multiple_sizes():
    for size in [64, 128, 256]:
        img = _make_palette_image(size, 16)
        comp = PaletteCompressor(max_colors=256)
        res = comp.compress(img, verbose=False)
        rec, meta = PaletteCompressor.decompress(res['recipe_bytes'])
        assert meta['sha256_match']
        assert np.array_equal(rec, img)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
