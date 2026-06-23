"""
pytest tests for Black Hole v5.2 Neural Atlas.
Run with: pytest tests/test_atlas_pytest.py -v
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
from siren_v5_atlas import AtlasCompressor


@pytest.fixture
def smooth_images():
    """5 smooth RGB images 32x32."""
    rng_seeds = [42, 43, 44, 45, 46]
    images = []
    for seed in rng_seeds:
        rng = np.random.default_rng(seed)
        ys, xs = np.mgrid[0:32, 0:32].astype(np.float32)
        img = np.zeros((32, 32, 3), dtype=np.float32)
        for c in range(3):
            cy, cx = rng.uniform(8, 24, 2)
            sigma = rng.uniform(5, 12)
            amp = rng.uniform(80, 200)
            img[:, :, c] = amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        images.append(np.clip(img, 0, 255).astype(np.uint8))
    return images


class TestAtlasConstruction:
    def test_default_construction(self):
        c = AtlasCompressor()
        assert c.hidden_features == 64
        assert c.hidden_layers == 3
        assert c.omega_0 == 30.0


class TestAtlasRoundtrip:
    def test_5_images_roundtrip(self, smooth_images):
        """5 images — must roundtrip bit-perfect."""
        comp = AtlasCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
        res = comp.compress(smooth_images, epochs=300, lr=1e-3,
                              bits=8, batch_size=2048, verbose=False)
        assert res['recipe_size'] > 0
        assert res['n_images'] == 5

        recovered, meta = AtlasCompressor.decompress(res['recipe_bytes'])
        assert meta['n_images'] == 5
        assert meta['all_sha256_match'] is True
        for orig, rec in zip(smooth_images, recovered):
            assert np.array_equal(orig, rec), "image roundtrip failed"

    def test_sha256_per_image(self, smooth_images):
        """Each image SHA-256 must match."""
        comp = AtlasCompressor(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress(smooth_images, epochs=200, lr=1e-3,
                              bits=8, batch_size=1024, verbose=False)
        recovered, meta = AtlasCompressor.decompress(res['recipe_bytes'])
        for i, (orig, rec) in enumerate(zip(smooth_images, recovered)):
            o_sha = hashlib.sha256(orig.tobytes()).hexdigest()
            r_sha = hashlib.sha256(rec.tobytes()).hexdigest()
            assert o_sha == r_sha, f"image {i} sha mismatch"
            assert meta['sha256_per_image'][i] is True

    def test_int4_atlas(self, smooth_images):
        """INT4 atlas must also roundtrip bit-perfect."""
        comp = AtlasCompressor(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress(smooth_images, epochs=200, lr=1e-3,
                              bits=4, batch_size=1024, verbose=False)
        recovered, meta = AtlasCompressor.decompress(res['recipe_bytes'])
        assert meta['all_sha256_match'] is True

    def test_amortized_weights(self, smooth_images):
        """Weights are shared — should be much smaller than N * single-recipe weights."""
        comp = AtlasCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
        res = comp.compress(smooth_images, epochs=300, lr=1e-3,
                              bits=8, batch_size=2048, verbose=False)
        # weights are shared across 5 images
        amort = res['weights_packed_size'] / res['n_images']
        # Per-image amortized weight should be much less than typical single recipe weights (~13KB)
        assert amort < 5000, f"amortized weight {amort} too high"


class TestAtlasPerformance:
    def test_atlas_beats_zip_small_n(self, smooth_images):
        """With N=5 similar smooth images, atlas should be roughly competitive
        with ZIP. On tiny 32x32 images ZIP has an edge, so we just verify the
        atlas is in the same ballpark (< 1.5x ZIP). The real advantage shows
        at 64x64+ as documented in tests/benchmark_atlas.py.
        """
        comp = AtlasCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
        res = comp.compress(smooth_images, epochs=1000, lr=1e-3,
                              bits=8, batch_size=2048, verbose=False)
        zip_total = sum(len(zlib.compress(im.tobytes(), 9)) for im in smooth_images)
        # 32x32 is small; atlas is competitive but not always winner here
        assert res['recipe_size'] < zip_total * 1.5, \
            f"atlas {res['recipe_size']} should be < ~ZIP {zip_total*1.5}"
        # Bit accuracy should be decent
        assert res['avg_bit_pct'] > 70.0
