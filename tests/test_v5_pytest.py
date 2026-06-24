"""
pytest tests for Black Hole v5 (PyTorch backend).

Run with: pytest tests/test_v5_pytest.py -v
"""
import os
import sys
import zlib
import hashlib
import numpy as np
import pytest

# Add paths
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase1_inr_compressor'))

# Try to import torch; if unavailable, skip these tests
torch = pytest.importorskip("torch")
from siren_v5_torch import ImageINRv5, SIREN, quantize_int8, dequantize_int8


# ---------- fixtures ----------
@pytest.fixture
def small_gradient():
    """16x16 RGB smooth gradient — fastest possible test."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for i in range(16):
        for j in range(16):
            img[i, j] = [i * 15, j * 15, (i + j) * 8]
    return img


@pytest.fixture
def medium_gradient():
    """64x64 RGB smooth gradient."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            img[i, j] = [int(i * 4), int(j * 4), int((i + j) * 2)]
    return img


# ---------- SIREN unit tests ----------
class TestSIRENConstruction:
    def test_default_construction(self):
        m = SIREN()
        assert m.in_features == 1
        assert m.hidden_features == 64
        assert m.hidden_layers == 3
        assert m.out_features == 1
        assert m.omega_0 == 30.0

    def test_custom_construction(self):
        m = SIREN(in_features=2, hidden_features=32, hidden_layers=2,
                  out_features=3, omega_0=15.0)
        assert m.in_features == 2
        assert m.hidden_features == 32
        assert m.omega_0 == 15.0

    def test_num_parameters(self):
        m = SIREN(in_features=2, hidden_features=32, hidden_layers=2, out_features=3)
        # 2*32 + 32 + 32*32 + 32 + 32*32 + 32 + 32*3 + 3 = 64+32+1024+32+1024+32+96+3 = 2307
        assert m.num_parameters() == 2307

    def test_forward_shape(self):
        m = SIREN(in_features=2, hidden_features=16, hidden_layers=1, out_features=3)
        x = torch.randn(100, 2)
        y = m(x)
        assert y.shape == (100, 3)


# ---------- quantization tests ----------
class TestQuantization:
    def test_int8_roundtrip(self):
        W = {
            'layer1.weight': np.random.randn(32, 16).astype(np.float32) * 0.01,
            'layer1.bias': np.random.randn(32).astype(np.float32) * 0.001,
        }
        packed, meta = quantize_int8(W)
        W2 = dequantize_int8(packed, meta)
        # Order should be preserved
        assert list(W2.keys()) == list(W.keys())
        # Quantization error should be small
        for k in W:
            err = np.max(np.abs(W[k] - W2[k]))
            # INT8 has ~1/127 of range error
            assert err < np.max(np.abs(W[k])) * 0.02, f"{k}: err={err}"

    def test_int8_preserves_order(self):
        """Critical: order must match named_parameters() iteration."""
        m = SIREN(in_features=2, hidden_features=16, hidden_layers=1, out_features=3)
        W = m.state_to_numpy()
        packed, meta = quantize_int8(W)
        names_in_meta = [entry[-1] for entry in meta]
        names_in_model = [name for name, _ in m.named_parameters()]
        assert names_in_meta == names_in_model, \
            f"order mismatch: meta={names_in_meta} model={names_in_model}"


# ---------- bit-perfect roundtrip tests ----------
class TestBitPerfect:
    def test_small_gradient_roundtrip(self, small_gradient):
        """16x16 image — must roundtrip bit-perfect."""
        comp = ImageINRv5(hidden_features=16, hidden_layers=1, omega_0=30.0)
        res = comp.compress_bitperfect(small_gradient,
                                         epochs=200, lr=1e-3,
                                         bits=8, batch_size=64,
                                         verbose=False)
        assert res['recipe_size'] > 0
        assert res['model_bit_accuracy'] > 60.0  # at least decent prediction

        recon, meta = ImageINRv5.decompress(res['recipe_bytes'])
        assert meta['exact_match'] is True
        assert recon.shape == small_gradient.shape
        assert np.array_equal(recon, small_gradient)

    def test_sha256_verified(self, small_gradient):
        """SHA-256 must match."""
        comp = ImageINRv5(hidden_features=16, hidden_layers=1, omega_0=30.0)
        res = comp.compress_bitperfect(small_gradient,
                                         epochs=200, lr=1e-3,
                                         bits=8, batch_size=64,
                                         verbose=False)
        orig_sha = hashlib.sha256(small_gradient.tobytes()).hexdigest()
        assert res['sha256'] == orig_sha

        recon, meta = ImageINRv5.decompress(res['recipe_bytes'])
        assert meta['sha256_match'] is True
        assert meta['sha256_recovered'] == orig_sha

    def test_int4_bitperfect(self, small_gradient):
        """INT4 quantization must also roundtrip bit-perfect."""
        comp = ImageINRv5(hidden_features=16, hidden_layers=1, omega_0=30.0)
        res = comp.compress_bitperfect(small_gradient,
                                         epochs=200, lr=1e-3,
                                         bits=4, batch_size=64,
                                         verbose=False)
        recon, meta = ImageINRv5.decompress(res['recipe_bytes'])
        assert meta['exact_match'] is True
        # INT4 recipe should be smaller than INT8
        assert res['weights_packed_size'] < 200

    def test_bigger_image_roundtrip(self, medium_gradient):
        """64x64 image — must roundtrip bit-perfect."""
        comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress_bitperfect(medium_gradient,
                                         epochs=500, lr=1e-3,
                                         bits=8, batch_size=2048,
                                         verbose=False)
        recon, meta = ImageINRv5.decompress(res['recipe_bytes'])
        assert meta['exact_match'] is True
        assert np.array_equal(recon, medium_gradient)


# ---------- performance test ----------
class TestPerformance:
    def test_v5_beats_zip_on_smooth(self, medium_gradient):
        """On smooth 64x64, BLKH should beat ZIP."""
        comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress_bitperfect(medium_gradient,
                                         epochs=1000, lr=1e-3,
                                         bits=8, batch_size=2048,
                                         verbose=False)
        zip_size = len(zlib.compress(medium_gradient.tobytes(), 9))
        # This should pass — smooth signals are SIREN's strength
        assert res['recipe_size'] < zip_size, \
            f"BLKH {res['recipe_size']} should be < ZIP {zip_size}"
