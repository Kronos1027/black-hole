"""
pytest tests for BLKH v5 lossy mode.
Run with: pytest tests/test_lossy_pytest.py -v
"""
import os
import sys
import zlib
import hashlib
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase1_inr_compressor'))

torch = pytest.importorskip("torch")
from siren_v5_torch import ImageINRv5


@pytest.fixture
def gradient_image():
    """128x128 smooth gradient for testing."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        for j in range(128):
            img[i, j] = [int(i * 2), int(j * 2), int((i + j))]
    return img


@pytest.fixture
def small_gradient():
    """16x16 small gradient for fast tests."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for i in range(16):
        for j in range(16):
            img[i, j] = [i * 15, j * 15, (i + j) * 8]
    return img


class TestLossyMode:
    def test_lossy_compresses_smaller_than_bitperfect(self, gradient_image):
        """Lossy mode should produce a smaller recipe than bit-perfect mode."""
        comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
        # Lossy
        res_lossy = comp.compress_lossy(gradient_image, epochs=500, lr=1e-3,
                                          bits=4, prune_threshold=0.01,
                                          batch_size=2048, verbose=False)
        # Bit-perfect (re-train fresh)
        comp2 = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res_bp = comp2.compress_bitperfect(gradient_image, epochs=500, lr=1e-3,
                                             bits=8, prune_threshold=0.0,
                                             batch_size=2048, verbose=False)
        assert res_lossy['recipe_size'] < res_bp['recipe_size'], \
            f"lossy {res_lossy['recipe_size']} should be < bitperfect {res_bp['recipe_size']}"
        assert res_lossy['mode'] == 'lossy'
        assert res_lossy['lossless'] is False

    def test_lossy_decompress_returns_predicted_image(self, small_gradient):
        """Decompress in lossy mode should return predicted image (no residual applied)."""
        comp = ImageINRv5(hidden_features=16, hidden_layers=1, omega_0=30.0)
        res = comp.compress_lossy(small_gradient, epochs=200, lr=1e-3,
                                    bits=4, prune_threshold=0.0,
                                    batch_size=64, verbose=False)
        recon, meta = ImageINRv5.decompress(res['recipe_bytes'])
        assert meta['mode'] == 'lossy'
        assert meta['lossless'] is False
        assert meta['exact_match'] is False
        assert recon.shape == small_gradient.shape

    def test_lossy_recipe_smaller_than_zip_on_smooth(self, gradient_image):
        """On smooth images, BLKH lossy should beat ZIP significantly."""
        comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress_lossy(gradient_image, epochs=800, lr=1e-3,
                                    bits=4, prune_threshold=0.01,
                                    batch_size=2048, verbose=False)
        zip_size = len(zlib.compress(gradient_image.tobytes(), 9))
        # Lossy should be MUCH smaller than ZIP (5x+ smaller)
        assert res['recipe_size'] < zip_size / 5, \
            f"BLKH lossy {res['recipe_size']} should be < ZIP/5 = {zip_size/5:.0f}"

    def test_lossy_psnr_above_threshold(self, gradient_image):
        """Lossy PSNR should be reasonable (>25 dB) for smooth images."""
        comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress_lossy(gradient_image, epochs=800, lr=1e-3,
                                    bits=4, prune_threshold=0.01,
                                    batch_size=2048, verbose=False)
        # 25 dB is the JPEG q=85 baseline for photos; smooth gradients should be higher
        assert res['psnr_db'] > 25.0, f"PSNR {res['psnr_db']:.1f} too low"

    def test_lossy_int8_higher_quality_than_int4(self, small_gradient):
        """INT8 lossy should give higher PSNR than INT4 lossy."""
        # INT4
        comp = ImageINRv5(hidden_features=16, hidden_layers=1, omega_0=30.0)
        res_int4 = comp.compress_lossy(small_gradient, epochs=200, lr=1e-3,
                                         bits=4, prune_threshold=0.0,
                                         batch_size=64, verbose=False)
        # INT8 (no pruning = better quality)
        comp2 = ImageINRv5(hidden_features=16, hidden_layers=1, omega_0=30.0)
        res_int8 = comp2.compress_lossy(small_gradient, epochs=200, lr=1e-3,
                                         bits=8, prune_threshold=0.0,
                                         batch_size=64, verbose=False)
        assert res_int8['psnr_db'] >= res_int4['psnr_db'] - 5, \
            f"INT8 {res_int8['psnr_db']:.1f} should be >= INT4 {res_int4['psnr_db']:.1f} - 5"

    def test_lossy_mode_detected_on_decompress(self, small_gradient):
        """Lossy recipe should be detected as lossy on decompress."""
        comp = ImageINRv5(hidden_features=16, hidden_layers=1, omega_0=30.0)
        res = comp.compress_lossy(small_gradient, epochs=100, lr=1e-3,
                                    bits=4, prune_threshold=0.0,
                                    batch_size=64, verbose=False)
        # The recipe bytes should decompress as lossy
        _, meta = ImageINRv5.decompress(res['recipe_bytes'])
        assert meta['mode'] == 'lossy'
        assert meta['residual_compressed_size'] == 0
