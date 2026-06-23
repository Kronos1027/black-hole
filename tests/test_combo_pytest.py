"""
pytest tests for BLKH v5.9 Combo mode.
Run with: pytest tests/test_combo_pytest.py -v
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
PIL = pytest.importorskip("PIL")  # combo uses WebP residual
from siren_v5_combo import ComboCompressor


@pytest.fixture
def smooth_images():
    """5 smooth RGB images 32x32."""
    images = []
    for seed in [42, 43, 44, 45, 46]:
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


class TestComboConstruction:
    def test_default_construction(self):
        c = ComboCompressor()
        assert c.latent_dim == 16
        assert c.hidden_features == 16
        assert c.hidden_layers == 1
        assert c.omega_0 == 30.0
        assert c.residual_codec == 'webp'

    def test_custom_codec(self):
        c = ComboCompressor(residual_codec='png')
        assert c.residual_codec == 'png'
        c2 = ComboCompressor(residual_codec='zlib')
        assert c2.residual_codec == 'zlib'


class TestComboRoundtrip:
    def test_5_images_roundtrip(self, smooth_images):
        """5 images — must roundtrip bit-perfect with WebP residual."""
        comp = ComboCompressor(latent_dim=16, hidden_features=16, hidden_layers=1,
                                omega_0=30.0, residual_codec='webp')
        comp.train_base(smooth_images, epochs=400, lr=1e-3,
                         batch_size=2048, verbose=False)
        res = comp.compress_many(smooth_images, epochs=300, lr=3e-3,
                                   bits=8, batch_size=2048, verbose=False)
        assert res['recipe_size'] > 0
        assert res['n_images'] == 5

        recovered, meta = ComboCompressor.decompress(res['recipe_bytes'])
        assert meta['n_images'] == 5
        assert meta['all_sha256_match'] is True
        assert meta['mode'] == 'combo'
        for orig, rec in zip(smooth_images, recovered):
            assert np.array_equal(orig, rec), "image roundtrip failed"

    def test_png_residual_roundtrip(self, smooth_images):
        """PNG residual must roundtrip bit-perfect."""
        comp = ComboCompressor(latent_dim=16, hidden_features=16, hidden_layers=1,
                                omega_0=30.0, residual_codec='png')
        comp.train_base(smooth_images, epochs=400, lr=1e-3,
                         batch_size=2048, verbose=False)
        res = comp.compress_many(smooth_images, epochs=300, lr=3e-3,
                                   bits=8, batch_size=2048, verbose=False)
        recovered, meta = ComboCompressor.decompress(res['recipe_bytes'])
        assert meta['all_sha256_match'] is True
        assert meta['residual_codec'] == 'png'

    def test_sha256_per_image(self, smooth_images):
        """Each image SHA-256 must match."""
        comp = ComboCompressor(latent_dim=16, hidden_features=16, hidden_layers=1,
                                omega_0=30.0, residual_codec='webp')
        comp.train_base(smooth_images, epochs=300, lr=1e-3,
                         batch_size=2048, verbose=False)
        res = comp.compress_many(smooth_images, epochs=200, lr=3e-3,
                                   bits=8, batch_size=2048, verbose=False)
        recovered, meta = ComboCompressor.decompress(res['recipe_bytes'])
        for i, (orig, rec) in enumerate(zip(smooth_images, recovered)):
            o_sha = hashlib.sha256(orig.tobytes()).hexdigest()
            r_sha = hashlib.sha256(rec.tobytes()).hexdigest()
            assert o_sha == r_sha, f"image {i} sha mismatch"
            assert meta['sha256_per_image'][i] is True


class TestComboPerformance:
    def test_combo_beats_zip_on_smooth(self, smooth_images):
        """Combo should beat ZIP per-file on 5 smooth similar images."""
        comp = ComboCompressor(latent_dim=16, hidden_features=16, hidden_layers=1,
                                omega_0=30.0, residual_codec='webp')
        comp.train_base(smooth_images, epochs=600, lr=1e-3,
                         batch_size=2048, verbose=False)
        res = comp.compress_many(smooth_images, epochs=400, lr=3e-3,
                                   bits=8, batch_size=2048, verbose=False)
        zip_total = sum(len(zlib.compress(im.tobytes(), 9)) for im in smooth_images)
        # Combo should beat ZIP (was 2.4x in self-test on 64x64; on 32x32 smaller margin)
        assert res['recipe_size'] < zip_total, \
            f"combo {res['recipe_size']} should be < ZIP {zip_total}"

    def test_combo_amortized_weights(self, smooth_images):
        """Hypernetwork weights are shared — should be much smaller than N * single weights."""
        comp = ComboCompressor(latent_dim=16, hidden_features=16, hidden_layers=1,
                                omega_0=30.0, residual_codec='webp')
        comp.train_base(smooth_images, epochs=300, lr=1e-3,
                         batch_size=2048, verbose=False)
        res = comp.compress_many(smooth_images, epochs=200, lr=3e-3,
                                   bits=8, batch_size=2048, verbose=False)
        # Per-image amortized weight should be < 2KB
        amort = res['hyper_size'] / res['n_images']
        assert amort < 2000, f"amortized weight {amort} too high"

    def test_combo_beats_hyper_xor_zlib(self, smooth_images):
        """Combo (WebP residual) should beat Hyper alone (XOR+zlib residual)."""
        from siren_v5_hyper import HyperCompressor
        # Hyper (XOR+zlib)
        comp_h = HyperCompressor(latent_dim=16, hidden_features=16, hidden_layers=1, omega_0=30.0)
        comp_h.train_base(smooth_images, epochs=500, lr=1e-3, batch_size=2048, verbose=False)
        res_h = comp_h.compress_many(smooth_images, epochs=300, lr=3e-3,
                                       bits=8, batch_size=2048, verbose=False)
        # Combo (WebP)
        comp_c = ComboCompressor(latent_dim=16, hidden_features=16, hidden_layers=1,
                                   omega_0=30.0, residual_codec='webp')
        comp_c.train_base(smooth_images, epochs=500, lr=1e-3, batch_size=2048, verbose=False)
        res_c = comp_c.compress_many(smooth_images, epochs=300, lr=3e-3,
                                       bits=8, batch_size=2048, verbose=False)
        assert res_c['recipe_size'] <= res_h['recipe_size'], \
            f"combo {res_c['recipe_size']} should be <= hyper {res_h['recipe_size']}"
