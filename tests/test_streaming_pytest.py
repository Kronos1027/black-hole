"""
pytest tests for BLKH v5.13 Streaming atlas.
Run with: pytest tests/test_streaming_pytest.py -v
"""
import os
import sys
import tempfile
import hashlib
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase1_inr_compressor'))

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")


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


class TestStreamingAtlas:
    def test_module_imports(self):
        from siren_v5_streaming import StreamingAtlas, StreamingWriter, StreamingReader
        assert StreamingAtlas is not None
        assert StreamingWriter is not None
        assert StreamingReader is not None

    def test_construction(self):
        from siren_v5_streaming import StreamingAtlas
        atlas = StreamingAtlas(latent_dim=16, hidden_features=16, hidden_layers=1)
        assert atlas.latent_dim == 16
        assert atlas.hidden_features == 16

    def test_train_base_and_stream(self, smooth_images):
        """Train base, then stream images to a file."""
        from siren_v5_streaming import StreamingAtlas
        atlas = StreamingAtlas(latent_dim=16, hidden_features=16, hidden_layers=1)
        atlas.train_base(smooth_images, epochs=300, lr=1e-3,
                          batch_size=2048, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.blks', delete=False) as tmp:
            path = tmp.name
        try:
            with atlas.open_stream(path, smooth_images[0].shape) as stream:
                for img in smooth_images:
                    stream.append(img, epochs=200)
                assert stream.n_images == 5
            # File should exist and have content
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_random_access_read(self, smooth_images):
        """Read individual images by index (random access)."""
        from siren_v5_streaming import StreamingAtlas
        atlas = StreamingAtlas(latent_dim=16, hidden_features=16, hidden_layers=1)
        atlas.train_base(smooth_images, epochs=300, lr=1e-3,
                          batch_size=2048, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.blks', delete=False) as tmp:
            path = tmp.name
        try:
            with atlas.open_stream(path, smooth_images[0].shape) as stream:
                for img in smooth_images:
                    stream.append(img, epochs=200)

            # Read back individual images
            with StreamingAtlas.open_read(path) as reader:
                assert reader.n_images == 5
                for i in [0, 2, 4]:
                    recovered = reader.read(i)
                    assert recovered.shape == smooth_images[i].shape
                    assert np.array_equal(recovered, smooth_images[i]), \
                        f"image {i} mismatch"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_sha256_per_image(self, smooth_images):
        """Each image's SHA-256 must match on random access read."""
        from siren_v5_streaming import StreamingAtlas
        atlas = StreamingAtlas(latent_dim=16, hidden_features=16, hidden_layers=1)
        atlas.train_base(smooth_images, epochs=300, lr=1e-3,
                          batch_size=2048, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.blks', delete=False) as tmp:
            path = tmp.name
        try:
            with atlas.open_stream(path, smooth_images[0].shape) as stream:
                for img in smooth_images:
                    stream.append(img, epochs=200)

            with StreamingAtlas.open_read(path) as reader:
                for i in range(5):
                    recovered = reader.read(i)
                    o_sha = hashlib.sha256(smooth_images[i].tobytes()).hexdigest()
                    r_sha = hashlib.sha256(recovered.tobytes()).hexdigest()
                    assert o_sha == r_sha, f"image {i} sha mismatch"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_append_many(self, smooth_images):
        """append_many should work the same as multiple append calls."""
        from siren_v5_streaming import StreamingAtlas
        atlas = StreamingAtlas(latent_dim=16, hidden_features=16, hidden_layers=1)
        atlas.train_base(smooth_images, epochs=200, lr=1e-3,
                          batch_size=2048, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.blks', delete=False) as tmp:
            path = tmp.name
        try:
            with atlas.open_stream(path, smooth_images[0].shape) as stream:
                indices = stream.append_many(smooth_images, epochs=150)
                assert indices == [0, 1, 2, 3, 4]
                assert stream.n_images == 5

            with StreamingAtlas.open_read(path) as reader:
                assert reader.n_images == 5
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_index_out_of_range(self, smooth_images):
        """Reading out of range should raise IndexError."""
        from siren_v5_streaming import StreamingAtlas
        atlas = StreamingAtlas(latent_dim=16, hidden_features=16, hidden_layers=1)
        atlas.train_base(smooth_images, epochs=200, lr=1e-3,
                          batch_size=2048, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.blks', delete=False) as tmp:
            path = tmp.name
        try:
            with atlas.open_stream(path, smooth_images[0].shape) as stream:
                stream.append(smooth_images[0], epochs=100)

            with StreamingAtlas.open_read(path) as reader:
                with pytest.raises(IndexError):
                    reader.read(5)
                with pytest.raises(IndexError):
                    reader.read(-1)
        finally:
            if os.path.exists(path):
                os.unlink(path)
