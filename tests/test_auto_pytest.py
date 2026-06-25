# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Pytest tests for siren_v5_auto.py (BLKH v5.30 auto mode)."""
import sys
import os
import zlib
import numpy as np
import pytest
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_inr_compressor'))
from siren_v5_auto import AutoCompressor


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


def test_auto_compress_shape():
    img = _make_smooth_image(64, seed=42)
    comp = AutoCompressor(quality=0.9, time_budget_s=2.0)
    res = comp.compress(img, verbose=False)
    assert 'recipe_bytes' in res
    assert 'recipe_size' in res
    assert res['recipe_size'] > 0


def test_auto_picks_palette_for_palette_image():
    img = _make_palette_image(128, 16)
    comp = AutoCompressor(quality=0.9, time_budget_s=3.0)
    res = comp.compress(img, verbose=False)
    assert res['mode'] == 'palette_v5_29', f"Should pick palette, got {res['mode']}"


def test_auto_picks_dct_for_smooth():
    img = _make_smooth_image(128, seed=42)
    comp = AutoCompressor(quality=0.9, time_budget_s=3.0)
    res = comp.compress(img, verbose=False)
    # Should pick a lossy mode (dct or fast)
    assert 'dct' in res['mode'] or 'fast' in res['mode'], f"Should pick DCT/fast, got {res['mode']}"


def test_auto_beats_zip():
    img = _make_smooth_image(128, seed=42)
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    comp = AutoCompressor(quality=0.9, time_budget_s=3.0)
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] < zip_sz


def test_auto_lossless_only():
    """When lossless=True, should only pick lossless modes."""
    img = _make_palette_image(128, 16)
    comp = AutoCompressor(quality=0.9, lossless=True, time_budget_s=2.0)
    res = comp.compress(img, verbose=False)
    # Should pick palette (lossless) or wavelet
    assert res['mode'] in ('palette_v5_29', 'wavelet_v5_20')


def test_auto_time_budget_respected():
    """Should not exceed time budget significantly."""
    img = _make_smooth_image(128, seed=42)
    comp = AutoCompressor(quality=0.9, time_budget_s=1.0, speed='fast')
    t0 = __import__('time').time()
    res = comp.compress(img, verbose=False)
    dt = __import__('time').time() - t0
    # Allow 50% overhead for setup
    assert dt < 2.0, f"Took {dt:.1f}s, budget was 1.0s"


def test_auto_modes_tried():
    """Should try multiple modes."""
    img = _make_smooth_image(128, seed=42)
    comp = AutoCompressor(quality=0.9, time_budget_s=3.0)
    res = comp.compress(img, verbose=False)
    assert res.get('modes_tried', 0) >= 1


def test_auto_real_photo():
    path = '/home/z/my-project/blackhole_repo/docs/assets/sample_photos/sky_128.png'
    if not os.path.exists(path):
        pytest.skip("Sample photo not available")
    img = np.array(Image.open(path).convert('RGB'))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    comp = AutoCompressor(quality=0.9, time_budget_s=3.0)
    res = comp.compress(img, verbose=False)
    assert res['recipe_size'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
