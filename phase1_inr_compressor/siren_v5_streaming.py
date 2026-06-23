"""
siren_v5_streaming.py — v5.13 Streaming atlas for datacenter use
==================================================================
Solves the datacenter problem: when you have 1000+ images in a corpus,
you don't want to load the ENTIRE atlas recipe to decompress ONE image.

This module provides a STREAMING index format that allows:
  - Compress images one at a time, append to atlas
  - Decompress any single image without loading the whole atlas
  - The shared hypernetwork is loaded ONCE, then per-image data is indexed

Use case: A hospital PACS system storing 100,000 MRI slices.
  - Train hypernetwork once (one-time cost)
  - Compress each new MRI slice as it arrives (append to atlas)
  - Retrieve any slice by index in O(1) without loading all 100K slices

Format (.blks — Black-hole Streaming):
  [header: magic, version, hypernetwork, index_offset]
  [hypernetwork weights]
  [per-image records, each: offset, length, latent, residual, sha]
  [index at end: list of (image_id, offset, length)]

This is a simplified append-only format. A production version would use
a proper database (SQLite) or chunked binary format.
"""
from __future__ import annotations
import os
import sys
import io
import time
import zlib
import struct
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siren_v5_torch import SineLayer, quantize_int8, dequantize_int8
from siren_v5_hyper import HyperNetwork, SIRENFunction
from siren_v5_hybrid import (
    encode_residual_webp, decode_residual_webp,
    encode_residual_png, decode_residual_png,
)


MAGIC_STREAMING = b'BLKS'
VERSION_STREAMING = 1


class StreamingAtlas:
    """
    Streaming atlas: compress/decompress individual images from a shared
    hypernetwork without loading the entire atlas.

    Usage:
        # Phase 1: train base (one-time)
        atlas = StreamingAtlas(latent_dim=16, hidden_features=16, hidden_layers=1)
        atlas.train_base(corpus_images, epochs=2000)

        # Phase 2: open a streaming file and append images
        with atlas.open_stream('atlas.blks') as stream:
            for img in new_images:
                stream.append(img)  # appends one image
            # Or batch:
            stream.append_many(images)

        # Phase 3: read individual images by index
        with StreamingAtlas.open_read('atlas.blks') as reader:
            img_42 = reader.read(42)  # load ONLY image 42
            img_0 = reader.read(0)
    """

    def __init__(self, latent_dim=16, hidden_features=16, hidden_layers=1,
                 omega_0=30.0, residual_codec: str = 'webp',
                 device: str | None = None):
        self.latent_dim = latent_dim
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = float(omega_0)
        self.residual_codec = residual_codec
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.siren_fn = SIRENFunction(
            in_features=2, hidden_features=hidden_features,
            hidden_layers=hidden_layers, out_features=3, omega_0=omega_0
        ).to(self.device)
        self.hyper = HyperNetwork(
            latent_dim=latent_dim,
            target_total_params=self.siren_fn.total_params,
        ).to(self.device)
        self._base_trained = False
        self._cached_hyper_state: dict | None = None

    def _make_coords(self, H: int, W: int) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij',
        )
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    def train_base(self, images: list[np.ndarray],
                   epochs: int = 3000, lr: float = 1e-3,
                   batch_size: int = 2048,
                   verbose: bool = False) -> dict:
        """Train the shared HyperNetwork on a corpus."""
        assert all(im.shape == images[0].shape for im in images)
        H, W, C = images[0].shape
        assert C == 3
        N = len(images)

        target_tensors = []
        for im in images:
            t = torch.from_numpy(
                (im.astype(np.float32) / 127.5 - 1.0).reshape(-1, 3)
            ).to(self.device)
            target_tensors.append(t)

        coords = self._make_coords(H, W)
        latents = [torch.zeros(self.latent_dim, device=self.device, requires_grad=True)
                   for _ in range(N)]

        opt = torch.optim.Adam(list(self.hyper.parameters()) + latents, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        warmup = max(1, epochs // 20)

        history = []
        t0 = time.time()
        for epoch in range(epochs):
            if epoch < warmup:
                for g in opt.param_groups:
                    g['lr'] = lr * (epoch + 1) / warmup
            idx = epoch % N
            target = target_tensors[idx]
            z = latents[idx]
            n = coords.shape[0]
            if batch_size < n:
                bidx = torch.randint(0, n, (batch_size,), device=self.device)
                xb = coords[bidx]; yb = target[bidx]
            else:
                xb, yb = coords, target
            flat_w = self.hyper(z.unsqueeze(0)).squeeze(0)
            pred = self.siren_fn(xb, flat_w)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            if epoch >= warmup:
                sched.step()
            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  streaming base epoch {epoch}/{epochs}  loss={loss.item():.6e}")
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(float(loss.item()))

        self._base_trained = True
        self._cached_hyper_state = {
            name: p.detach().cpu().numpy().astype(np.float32)
            for name, p in self.hyper.named_parameters()
        }
        return {'train_time_s': time.time() - t0, 'final_loss': history[-1] if history else float('nan')}

    def _compress_one(self, img: np.ndarray, epochs: int = 800, lr: float = 3e-3,
                       batch_size: int = 2048) -> tuple[bytes, float, bytes]:
        """Compress one image. Returns (latent_q_bytes, latent_scale, residual_compressed)."""
        H, W, C = img.shape
        target = torch.from_numpy(
            (img.astype(np.float32) / 127.5 - 1.0).reshape(-1, 3)
        ).to(self.device)
        coords = self._make_coords(H, W)
        n = coords.shape[0]
        bs = min(batch_size, n)

        z = torch.zeros(self.latent_dim, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        warmup = max(1, epochs // 20)
        for epoch in range(epochs):
            if epoch < warmup:
                for g in opt.param_groups:
                    g['lr'] = lr * (epoch + 1) / warmup
            if bs < n:
                bidx = torch.randint(0, n, (bs,), device=self.device)
                xb = coords[bidx]; yb = target[bidx]
            else:
                xb, yb = coords, target
            flat_w = self.hyper(z.unsqueeze(0)).squeeze(0)
            pred = self.siren_fn(xb, flat_w)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            if epoch >= warmup:
                sched.step()

        z_np = z.detach().cpu().numpy().astype(np.float32)
        z_max_abs = float(np.max(np.abs(z_np))) if z_np.size > 0 else 0.0
        z_scale = z_max_abs / 127.0 if z_max_abs > 0 else 1.0
        z_q = np.clip(np.round(z_np / z_scale), -127, 127).astype(np.int8)
        z_q_tensor = torch.from_numpy((z_q.astype(np.float32) * z_scale)).to(self.device)

        with torch.no_grad():
            flat_w = self.hyper(z_q_tensor.unsqueeze(0)).squeeze(0)
            pred = self.siren_fn(coords, flat_w).cpu().numpy()
        predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(H, W, C)

        residual_img = ((img.astype(np.int16) - predicted.astype(np.int16)) % 256).astype(np.uint8)
        if self.residual_codec == 'webp':
            residual_compressed = encode_residual_webp(residual_img)
        elif self.residual_codec == 'png':
            residual_compressed = encode_residual_png(residual_img)
        else:
            residual_compressed = zlib.compress(residual_img.tobytes(), 9)

        return z_q.tobytes(), z_scale, residual_compressed

    def open_stream(self, path: str, img_shape: tuple) -> 'StreamingWriter':
        """Open a streaming atlas file for writing (appending images).
        img_shape: (H, W, C) — all images must have this shape.
        """
        if not self._base_trained or self._cached_hyper_state is None:
            raise RuntimeError("Must call train_base() first")
        return StreamingWriter(self, path, img_shape)

    @staticmethod
    def open_read(path: str, device: str | None = None) -> 'StreamingReader':
        """Open a streaming atlas file for reading (random access by index)."""
        return StreamingReader(path, device)


class StreamingWriter:
    """Write-only streaming atlas. Append images one at a time."""

    def __init__(self, atlas: StreamingAtlas, path: str, img_shape: tuple):
        self.atlas = atlas
        self.path = path
        self.H, self.W, self.C = img_shape
        self.n_images = 0
        self.index = []  # list of (offset, length)

        # Quantize and write hypernetwork
        self.packed_hyper, self.hyper_meta = quantize_int8(atlas._cached_hyper_state)
        # Reload quantized hypernetwork for compression
        hyper_q = dequantize_int8(self.packed_hyper, self.hyper_meta)
        with torch.no_grad():
            for name, p in atlas.hyper.named_parameters():
                p.copy_(torch.from_numpy(hyper_q[name]))
        atlas.hyper.eval()

        # Write header (will be updated at close)
        self.f = open(path, 'wb')
        self._write_header()

    def _write_header(self):
        """Write fixed-size header."""
        self.f.seek(0)
        self.f.write(MAGIC_STREAMING)
        self.f.write(struct.pack('<B', VERSION_STREAMING))
        # codec
        codec_id = {'zlib': 0, 'png': 1, 'webp': 2}[self.atlas.residual_codec]
        self.f.write(struct.pack('<B', codec_id))
        # arch
        self.f.write(struct.pack('<H', self.atlas.latent_dim))
        self.f.write(struct.pack('<H', self.atlas.hidden_features))
        self.f.write(struct.pack('<B', self.atlas.hidden_layers))
        self.f.write(struct.pack('<f', self.atlas.omega_0))
        # shape
        self.f.write(struct.pack('<H', self.H))
        self.f.write(struct.pack('<H', self.W))
        self.f.write(struct.pack('<B', self.C))
        # n_images (placeholder, updated at close)
        self.f.write(struct.pack('<I', 0))
        # hypernetwork
        self.f.write(struct.pack('<I', len(self.packed_hyper)))
        self.f.write(self.packed_hyper)
        self.f.write(struct.pack('<H', len(self.hyper_meta)))
        for entry in self.hyper_meta:
            n_bytes, shape, scale, name = entry
            name_b = name.encode('utf-8')
            self.f.write(struct.pack('<B', len(name_b))); self.f.write(name_b)
            self.f.write(struct.pack('<I', n_bytes))
            self.f.write(struct.pack('<B', len(shape)))
            for d in shape: self.f.write(struct.pack('<i', int(d)))
            self.f.write(struct.pack('<d', float(scale)))
            self.f.write(struct.pack('<I', 0))
        self.data_start = self.f.tell()

    def append(self, img: np.ndarray, epochs: int = 800) -> int:
        """Append one image. Returns its index."""
        assert img.shape == (self.H, self.W, self.C), \
            f"shape {img.shape} != expected {(self.H, self.W, self.C)}"
        idx = self.n_images
        offset = self.f.tell()

        latent_q, latent_scale, residual = self.atlas._compress_one(img, epochs=epochs)

        # Write record: latent + residual + sha
        sha = hashlib.sha256(img.tobytes()).digest()
        self.f.write(struct.pack('<I', len(latent_q)))
        self.f.write(latent_q)
        self.f.write(struct.pack('<d', float(latent_scale)))
        self.f.write(struct.pack('<Q', len(residual)))
        self.f.write(residual)
        self.f.write(sha)

        length = self.f.tell() - offset
        self.index.append((offset, length))
        self.n_images += 1
        return idx

    def append_many(self, images: list[np.ndarray], epochs: int = 800) -> list[int]:
        """Append multiple images. Returns list of indices."""
        return [self.append(img, epochs=epochs) for img in images]

    def close(self):
        """Finalize the atlas: write index and update n_images in header."""
        # Write index at the end
        index_offset = self.f.tell()
        self.f.write(struct.pack('<I', len(self.index)))
        for offset, length in self.index:
            self.f.write(struct.pack('<QI', offset, length))
        # Update n_images in header
        self.f.seek(4 + 1 + 1 + 2 + 2 + 1 + 4 + 2 + 2 + 1)  # position of n_images
        self.f.write(struct.pack('<I', self.n_images))
        # Write index_offset at the very end
        self.f.seek(0, 2)  # end
        # Already wrote index; nothing more to do
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class StreamingReader:
    """Random-access reader for streaming atlas. Read any image by index."""

    def __init__(self, path: str, device: str | None = None):
        self.path = path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.f = open(path, 'rb')

        # Read header
        magic = self.f.read(4)
        if magic != MAGIC_STREAMING:
            raise ValueError(f"bad magic: {magic!r}")
        self.version = struct.unpack('<B', self.f.read(1))[0]
        assert self.version == VERSION_STREAMING
        codec_id = struct.unpack('<B', self.f.read(1))[0]
        self.codec_name = {0: 'zlib', 1: 'png', 2: 'webp'}[codec_id]
        self.latent_dim = struct.unpack('<H', self.f.read(2))[0]
        self.hidden_features = struct.unpack('<H', self.f.read(2))[0]
        self.hidden_layers = struct.unpack('<B', self.f.read(1))[0]
        self.omega_0 = struct.unpack('<f', self.f.read(4))[0]
        self.H = struct.unpack('<H', self.f.read(2))[0]
        self.W = struct.unpack('<H', self.f.read(2))[0]
        self.C = struct.unpack('<B', self.f.read(1))[0]
        self.n_images = struct.unpack('<I', self.f.read(4))[0]

        # Read hypernetwork
        hyper_size = struct.unpack('<I', self.f.read(4))[0]
        packed_hyper = self.f.read(hyper_size)
        n_meta = struct.unpack('<H', self.f.read(2))[0]
        hyper_meta = []
        for _ in range(n_meta):
            name_len = struct.unpack('<B', self.f.read(1))[0]
            name = self.f.read(name_len).decode('utf-8')
            n_bytes = struct.unpack('<I', self.f.read(4))[0]
            ndim = struct.unpack('<B', self.f.read(1))[0]
            shape = tuple(struct.unpack('<' + 'i'*ndim, self.f.read(4*ndim)))
            scale = struct.unpack('<d', self.f.read(8))[0]
            _ = struct.unpack('<I', self.f.read(4))[0]
            hyper_meta.append((n_bytes, shape, scale, name))

        self.data_start = self.f.tell()

        # Build model
        hyper_dict = dequantize_int8(packed_hyper, hyper_meta)
        self.siren_fn = SIRENFunction(
            in_features=2, hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers, out_features=3, omega_0=self.omega_0,
        ).to(self.device)
        self.hyper = HyperNetwork(
            latent_dim=self.latent_dim,
            target_total_params=self.siren_fn.total_params,
        ).to(self.device)
        with torch.no_grad():
            for name, p in self.hyper.named_parameters():
                p.copy_(torch.from_numpy(hyper_dict[name]))
        self.hyper.eval()

        # Build coords
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, self.H, device=self.device),
            torch.linspace(-1, 1, self.W, device=self.device),
            indexing='ij',
        )
        self.coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

        # Read index (at end of file)
        self.f.seek(0, 2)
        file_size = self.f.tell()
        # Index is at: file_size - 4 - n_images * 12
        # But we know n_images from header
        index_size = 4 + self.n_images * 12
        self.f.seek(file_size - index_size)
        n_idx = struct.unpack('<I', self.f.read(4))[0]
        self.index = []
        for _ in range(n_idx):
            offset = struct.unpack('<Q', self.f.read(8))[0]
            length = struct.unpack('<I', self.f.read(4))[0]
            self.index.append((offset, length))

    def read(self, idx: int) -> np.ndarray:
        """Read a single image by index (0-based). Loads ONLY that image."""
        if idx < 0 or idx >= self.n_images:
            raise IndexError(f"index {idx} out of range [0, {self.n_images})")
        offset, length = self.index[idx]
        self.f.seek(offset)
        # Read record
        latent_size = struct.unpack('<I', self.f.read(4))[0]
        latent_q = self.f.read(latent_size)
        latent_scale = struct.unpack('<d', self.f.read(8))[0]
        resid_size = struct.unpack('<Q', self.f.read(8))[0]
        residual_compressed = self.f.read(resid_size)
        sha_expected = self.f.read(32)

        # Dequantize latent
        z_q = np.frombuffer(latent_q, dtype=np.int8).astype(np.float32)
        z = torch.from_numpy(z_q * latent_scale).to(self.device)

        # Inference
        with torch.no_grad():
            flat_w = self.hyper(z.unsqueeze(0)).squeeze(0)
            pred = self.siren_fn(self.coords, flat_w).cpu().numpy()
        predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(self.H, self.W, self.C)

        # Decode residual
        if self.codec_name == 'webp':
            residual_img = decode_residual_webp(residual_compressed)
        elif self.codec_name == 'png':
            residual_img = decode_residual_png(residual_compressed)
        else:
            from siren_v5_hybrid import decode_residual_zlib
            residual_img = decode_residual_zlib(residual_compressed, (self.H, self.W, self.C))

        # Recover
        recovered = ((predicted.astype(np.int16) + residual_img.astype(np.int16)) % 256).astype(np.uint8)
        return recovered

    def read_many(self, indices: list[int]) -> list[np.ndarray]:
        """Read multiple images by index."""
        return [self.read(i) for i in indices]

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
#  Self-test
# ============================================================
def _self_test():
    import tempfile
    print(f"[streaming] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Generate 10 smooth images
    N = 10
    SIZE = 32
    images = []
    for i in range(N):
        rng = np.random.default_rng(seed=42 + i)
        ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
        img = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
        for c in range(3):
            cy, cx = rng.uniform(SIZE * 0.2, SIZE * 0.8, 2)
            sigma = rng.uniform(SIZE * 0.1, SIZE * 0.25)
            amp = rng.uniform(80, 200)
            img[:, :, c] = amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        images.append(np.clip(img, 0, 255).astype(np.uint8))

    print(f"[streaming] {N} images x {SIZE}x{SIZE}x3")

    # Phase 1: train base
    atlas = StreamingAtlas(latent_dim=16, hidden_features=16, hidden_layers=1, omega_0=30.0)
    print(f"[streaming] Phase 1: training base...")
    t0 = time.time()
    atlas.train_base(images, epochs=1500, lr=1e-3, batch_size=2048, verbose=False)
    print(f"  base trained in {time.time()-t0:.1f}s")

    # Phase 2: open stream and append images
    with tempfile.NamedTemporaryFile(suffix='.blks', delete=False) as tmp:
        path = tmp.name
    print(f"\n[streaming] Phase 2: appending {N} images to {path}...")
    t0 = time.time()
    with atlas.open_stream(path, images[0].shape) as stream:
        for img in images:
            stream.append(img, epochs=400)
    file_size = os.path.getsize(path)
    dt = time.time() - t0
    print(f"  wrote {file_size:,}B in {dt:.1f}s ({file_size/N:.0f}B/image)")

    # Phase 3: random access
    print(f"\n[streaming] Phase 3: random access (reading images 0, 5, 9)...")
    t0 = time.time()
    with StreamingAtlas.open_read(path) as reader:
        print(f"  atlas has {reader.n_images} images, shape ({reader.H}, {reader.W}, {reader.C})")
        for idx in [0, 5, 9]:
            t_read = time.time()
            img = reader.read(idx)
            dt_read = time.time() - t_read
            # Verify
            o_sha = hashlib.sha256(images[idx].tobytes()).hexdigest()[:16]
            r_sha = hashlib.sha256(img.tobytes()).hexdigest()[:16]
            match = np.array_equal(images[idx], img)
            print(f"  image {idx}: read in {dt_read*1000:.1f}ms, sha match: {match}")

    os.unlink(path)
    print(f"\n[streaming] Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    _self_test()
