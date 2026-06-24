"""
pytest tests for BLKH v5.8 Hybrid mode.
Run with: pytest tests/test_hybrid_pytest.py -v
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
PIL = pytest.importorskip("PIL")  # hybrid needs Pillow
from siren_v5_hybrid import HybridCompressor


@pytest.fixture
def gradient_image():
    """64x64 smooth gradient."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            img[i, j] = [int(i * 4), int(j * 4), int((i + j) * 2)]
    return img


@pytest.fixture
def small_gradient():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for i in range(16):
        for j in range(16):
            img[i, j] = [i * 15, j * 15, (i + j) * 8]
    return img


class TestHybridMode:
    def test_webp_residual_roundtrip(self, small_gradient):
        """WebP residual must roundtrip bit-perfect."""
        comp = HybridCompressor(hidden_features=16, hidden_layers=1,
                                 omega_0=30.0, residual_codec='webp')
        res = comp.compress_bitperfect(small_gradient, epochs=200, lr=1e-3,
                                          bits=8, batch_size=64, verbose=False)
        recon, meta = HybridCompressor.decompress(res['recipe_bytes'])
        assert meta['exact_match'] is True
        assert meta['residual_codec'] == 'webp'
        assert np.array_equal(recon, small_gradient)

    def test_png_residual_roundtrip(self, small_gradient):
        """PNG residual must roundtrip bit-perfect."""
        comp = HybridCompressor(hidden_features=16, hidden_layers=1,
                                 omega_0=30.0, residual_codec='png')
        res = comp.compress_bitperfect(small_gradient, epochs=200, lr=1e-3,
                                          bits=8, batch_size=64, verbose=False)
        recon, meta = HybridCompressor.decompress(res['recipe_bytes'])
        assert meta['exact_match'] is True
        assert meta['residual_codec'] == 'png'

    def test_zlib_residual_roundtrip(self, small_gradient):
        """zlib residual (fallback) must roundtrip bit-perfect."""
        comp = HybridCompressor(hidden_features=16, hidden_layers=1,
                                 omega_0=30.0, residual_codec='zlib')
        res = comp.compress_bitperfect(small_gradient, epochs=200, lr=1e-3,
                                          bits=8, batch_size=64, verbose=False)
        recon, meta = HybridCompressor.decompress(res['recipe_bytes'])
        assert meta['exact_match'] is True
        assert meta['residual_codec'] == 'zlib'

    def test_hybrid_smaller_than_v5_on_smooth(self, gradient_image):
        """Hybrid mode should produce smaller recipe than v5 (XOR+zlib) on smooth images."""
        from siren_v5_torch import ImageINRv5
        # v5 (XOR + zlib)
        comp_v5 = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res_v5 = comp_v5.compress_bitperfect(gradient_image, epochs=500, lr=1e-3,
                                                bits=8, batch_size=2048, verbose=False)
        # Hybrid (WebP residual)
        comp_hyb = HybridCompressor(hidden_features=32, hidden_layers=2,
                                      omega_0=30.0, residual_codec='webp')
        res_hyb = comp_hyb.compress_bitperfect(gradient_image, epochs=500, lr=1e-3,
                                                  bits=8, batch_size=2048, verbose=False)
        assert res_hyb['recipe_size'] <= res_v5['recipe_size'], \
            f"hybrid {res_hyb['recipe_size']} should be <= v5 {res_v5['recipe_size']}"

    def test_hybrid_beats_zip_on_gradient(self, gradient_image):
        """On smooth gradients, hybrid should beat ZIP significantly."""
        comp = HybridCompressor(hidden_features=32, hidden_layers=2,
                                 omega_0=30.0, residual_codec='webp')
        res = comp.compress_bitperfect(gradient_image, epochs=800, lr=1e-3,
                                          bits=8, batch_size=2048, verbose=False)
        zip_size = len(zlib.compress(gradient_image.tobytes(), 9))
        assert res['recipe_size'] < zip_size, \
            f"hybrid {res['recipe_size']} should be < ZIP {zip_size}"

    def test_all_codecs_produce_valid_recipes(self, small_gradient):
        """All 3 codecs should produce valid bit-perfect recipes."""
        for codec in ['zlib', 'png', 'webp']:
            comp = HybridCompressor(hidden_features=16, hidden_layers=1,
                                     omega_0=30.0, residual_codec=codec)
            res = comp.compress_bitperfect(small_gradient, epochs=150, lr=1e-3,
                                              bits=8, batch_size=64, verbose=False)
            recon, meta = HybridCompressor.decompress(res['recipe_bytes'])
            assert meta['exact_match'], f"{codec} failed SHA-256"
            assert np.array_equal(recon, small_gradient), f"{codec} roundtrip failed"
