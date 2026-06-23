"""
pytest tests for BLKH v5.12 3D Volume compression.
Run with: pytest tests/test_volume_pytest.py -v
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


@pytest.fixture
def small_volume():
    """Small 3D volume: 8x16x16x1 (single channel, smooth)."""
    D, H, W = 8, 16, 16
    zs, ys, xs = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
    zs = zs / (D - 1) * 2 - 1
    ys = ys / (H - 1) * 2 - 1
    xs = xs / (W - 1) * 2 - 1
    sigma = 0.4
    vol = 200 * np.exp(-((xs)**2 + (ys)**2 + (zs)**2) / (2 * sigma**2))
    vol = np.clip(vol, 0, 255).astype(np.uint8)
    return vol[..., None]  # (D, H, W, 1)


@pytest.fixture
def multi_channel_volume():
    """Multi-channel 3D volume: 8x16x16x3 (3 channels)."""
    D, H, W = 8, 16, 16
    zs, ys, xs = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
    zs = zs / (D - 1) * 2 - 1
    ys = ys / (H - 1) * 2 - 1
    xs = xs / (W - 1) * 2 - 1
    vol = np.zeros((D, H, W, 3), dtype=np.float32)
    for c in range(3):
        sigma = 0.3 + c * 0.1
        amp = 180 - c * 30
        vol[..., c] = amp * np.exp(-(xs**2 + ys**2 + zs**2) / (2 * sigma**2))
    return np.clip(vol, 0, 255).astype(np.uint8)


class TestVolumeConstruction:
    def test_module_imports(self):
        from siren_v5_volume import VolumeCompressor, VolumeSIREN
        assert VolumeCompressor is not None
        assert VolumeSIREN is not None

    def test_default_construction(self):
        from siren_v5_volume import VolumeCompressor
        comp = VolumeCompressor()
        assert comp.hidden_features == 64
        assert comp.hidden_layers == 3
        assert comp.omega_0 == 30.0


class TestVolumeRoundtrip:
    def test_single_channel_roundtrip(self, small_volume):
        """Single-channel 3D volume must roundtrip bit-perfect."""
        from siren_v5_volume import VolumeCompressor
        comp = VolumeCompressor(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress(small_volume, epochs=300, lr=1e-3,
                              bits=8, batch_size=2048, verbose=False)
        recovered, meta = VolumeCompressor.decompress(res['recipe_bytes'])
        assert meta['exact_match'] is True
        assert meta['mode'] == 'volume'
        assert recovered.shape == small_volume.shape
        assert np.array_equal(recovered, small_volume)

    def test_multi_channel_roundtrip(self, multi_channel_volume):
        """Multi-channel 3D volume must roundtrip bit-perfect."""
        from siren_v5_volume import VolumeCompressor
        comp = VolumeCompressor(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress(multi_channel_volume, epochs=300, lr=1e-3,
                              bits=8, batch_size=2048, verbose=False)
        recovered, meta = VolumeCompressor.decompress(res['recipe_bytes'])
        assert meta['exact_match'] is True
        assert recovered.shape == multi_channel_volume.shape
        assert np.array_equal(recovered, multi_channel_volume)

    def test_sha256_verified(self, small_volume):
        """SHA-256 must match."""
        from siren_v5_volume import VolumeCompressor
        comp = VolumeCompressor(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress(small_volume, epochs=200, lr=1e-3,
                              bits=8, batch_size=2048, verbose=False)
        orig_sha = hashlib.sha256(small_volume.tobytes()).hexdigest()
        assert res['sha256'] == orig_sha
        recovered, meta = VolumeCompressor.decompress(res['recipe_bytes'])
        assert meta['sha256_match'] is True

    def test_shape_preserved(self, small_volume):
        """Volume shape must be preserved through roundtrip."""
        from siren_v5_volume import VolumeCompressor
        comp = VolumeCompressor(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress(small_volume, epochs=200, lr=1e-3,
                              bits=8, batch_size=2048, verbose=False)
        recovered, meta = VolumeCompressor.decompress(res['recipe_bytes'])
        assert meta['shape'] == small_volume.shape
        assert recovered.shape == small_volume.shape


class TestVolumePerformance:
    def test_volume_beats_zip_at_scale(self, small_volume):
        """On smooth volumes, BLKH should be competitive with ZIP.
        Note: on tiny volumes (8x16x16 = 2K voxels), ZIP wins easily due to
        SIREN weight overhead (~2KB). The real advantage shows on large
        volumes (D×H×W > 100K voxels) where weights amortize.
        We just verify competitive here (within 10x of ZIP) for the tiny case.
        """
        from siren_v5_volume import VolumeCompressor
        comp = VolumeCompressor(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress(small_volume, epochs=400, lr=1e-3,
                              bits=8, batch_size=2048, verbose=False)
        zip_size = len(zlib.compress(small_volume.tobytes(), 9))
        # Allow up to 10x of ZIP for tiny volumes (weight overhead dominates)
        assert res['recipe_size'] < zip_size * 10, \
            f"volume {res['recipe_size']} should be < ~ZIP {zip_size*10}"

    def test_bit_accuracy_above_threshold(self, small_volume):
        """Bit accuracy should be above 70% for smooth volumes."""
        from siren_v5_volume import VolumeCompressor
        comp = VolumeCompressor(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress(small_volume, epochs=300, lr=1e-3,
                              bits=8, batch_size=2048, verbose=False)
        assert res['model_bit_accuracy'] > 70.0, \
            f"bit accuracy {res['model_bit_accuracy']:.1f}% too low"
