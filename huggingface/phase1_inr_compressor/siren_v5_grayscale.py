# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

"""
siren_v5_grayscale.py — v5.16 Native grayscale support
=======================================================
MRI, CT, seismic, and many scientific images are single-channel.
Forcing them to 3 channels (RGB) wastes 3x space on the residual.

This module adds native grayscale (1-channel) support:
  - SIREN f(x,y) → 1 value (instead of 3)
  - Residual is 1-channel image (1/3 the size)
  - SIREN weights slightly smaller (out=1 vs out=3)

Expected improvement for grayscale images:
  - Recipe size: ~3x smaller residual
  - Same SIREN quality
  - Same encoding speed (fewer output params = slightly faster)
"""
from __future__ import annotations
import os
import sys
import io
import time
import zlib
import struct
import hashlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siren_v5_torch import SineLayer, quantize_int8, dequantize_int8
from siren_v5_hybrid import (
    encode_residual_webp, decode_residual_webp,
    encode_residual_png, decode_residual_png,
)


MAGIC_GRAYSCALE = b'BLKG'  # BLK Grayscale
VERSION_GRAYSCALE = 1


def encode_residual_png_gray(residual_img: np.ndarray) -> bytes:
    """Encode 1-channel residual as PNG."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(residual_img, mode='L').save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def decode_residual_png_gray(data: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    return np.array(img, dtype=np.uint8)


def encode_residual_webp_gray(residual_img: np.ndarray) -> bytes:
    """Encode 1-channel residual as WebP lossless."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(residual_img, mode='L').save(buf, format='WebP', lossless=True)
    return buf.getvalue()


def decode_residual_webp_gray(data: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    return np.array(img, dtype=np.uint8)


class GrayscaleCompressor:
    """
    BLKH for native grayscale (1-channel) images.
    SIREN f(x,y) → 1 value. Residual is 1-channel PNG/WebP.
    3x smaller residual than RGB approach for the same image.
    """

    def __init__(self, hidden_features=32, hidden_layers=2, omega_0=30.0,
                 residual_codec: str = 'png',
                 device: str | None = None):
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = float(omega_0)
        self.residual_codec = residual_codec
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _make_model(self):
        from siren_v5_torch import SIREN
        return SIREN(in_features=2, hidden_features=self.hidden_features,
                     hidden_layers=self.hidden_layers, out_features=1,
                     omega_0=self.omega_0).to(self.device)

    def _make_coords(self, H, W):
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij',
        )
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    def compress_bitperfect(self, image: np.ndarray,
                             epochs=200, lr=4e-3, bits=8,
                             batch_size=16384, use_amp=True,
                             patience=3, verbose=False) -> dict:
        """Compress grayscale image (H, W) uint8. Bit-perfect."""
        assert image.dtype == np.uint8 and image.ndim == 2, "Expected (H, W) uint8"
        H, W = image.shape
        original_bytes = image.tobytes()

        coords = self._make_coords(H, W)
        values = torch.from_numpy(
            (image.astype(np.float32) / 127.5 - 1.0).reshape(-1, 1)
        ).to(self.device)

        N = coords.shape[0]
        model = self._make_model()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        warmup = max(1, epochs // 20)

        amp_dtype = torch.bfloat16 if (use_amp and self.device.type == 'cpu') else (
            torch.float16 if use_amp else None
        )

        model.train()
        history = []
        best_loss = float('inf')
        patience_counter = 0
        t0 = time.time()
        for epoch in range(epochs):
            if epoch < warmup:
                for g in opt.param_groups:
                    g['lr'] = lr * (epoch + 1) / warmup

            if batch_size < N:
                idx = torch.randint(0, N, (batch_size,), device=self.device)
                xb = coords[idx]; yb = values[idx]
            else:
                xb, yb = coords, values

            if amp_dtype is not None:
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                    pred = model(xb)
                    loss = torch.nn.functional.mse_loss(pred, yb)
            else:
                pred = model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            if epoch >= warmup:
                sched.step()

            cur_loss = float(loss.item())
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(cur_loss)
                if patience > 0 and epoch >= warmup:
                    if cur_loss < best_loss - 1e-6:
                        best_loss = cur_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break

        train_time = time.time() - t0

        # Quantize
        weights_np = model.state_to_numpy()
        packed, packed_meta = quantize_int8(weights_np)
        q_weights = dequantize_int8(packed, packed_meta)
        model.load_from_numpy(q_weights)
        model.eval()

        # Inference
        with torch.inference_mode():
            pred = model(coords).cpu().numpy()
        predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(H, W)

        # Residual (1-channel!)
        residual_img = ((image.astype(np.int16) - predicted.astype(np.int16)) % 256).astype(np.uint8)
        assert np.array_equal(
            ((predicted.astype(np.int16) + residual_img.astype(np.int16)) % 256).astype(np.uint8),
            image
        ), "residual math failed"

        # Encode residual (1-channel = 1/3 the size of RGB!)
        if self.residual_codec == 'webp':
            residual_compressed = encode_residual_webp_gray(residual_img)
        elif self.residual_codec == 'png':
            residual_compressed = encode_residual_png_gray(residual_img)
        else:
            residual_compressed = zlib.compress(residual_img.tobytes(), 9)

        sha = hashlib.sha256(original_bytes).digest()

        # Pack recipe
        recipe = self._pack_recipe(bits, packed, packed_meta, H, W, residual_compressed, sha)

        # Stats
        a = np.frombuffer(original_bytes, dtype=np.uint8)
        b = np.frombuffer(predicted.tobytes(), dtype=np.uint8)
        bit_acc = float(np.mean(np.unpackbits(a) == np.unpackbits(b))) * 100

        return {
            'recipe_bytes': recipe,
            'original_size': len(original_bytes),
            'recipe_size': len(recipe),
            'weights_packed_size': len(packed),
            'residual_compressed_size': len(residual_compressed),
            'model_bit_accuracy': bit_acc,
            'train_time_s': train_time,
            'sha256': sha.hex(),
            'mode': 'grayscale',
        }

    def _pack_recipe(self, bits, packed, packed_meta, H, W, residual_compressed, sha):
        out = bytearray()
        out += MAGIC_GRAYSCALE
        out += struct.pack('<B', VERSION_GRAYSCALE)
        out += struct.pack('<B', bits)
        codec_id = {'zlib': 0, 'png': 1, 'webp': 2}[self.residual_codec]
        out += struct.pack('<B', codec_id)
        out += struct.pack('<H', self.hidden_features)
        out += struct.pack('<B', self.hidden_layers)
        out += struct.pack('<f', self.omega_0)
        out += struct.pack('<H', H)
        out += struct.pack('<H', W)
        # No C field (always 1 channel)
        out += struct.pack('<I', len(packed))
        out += packed
        out += struct.pack('<H', len(packed_meta))
        for entry in packed_meta:
            n_bytes, shape, scale, name = entry
            name_b = name.encode('utf-8')
            out += struct.pack('<B', len(name_b)); out += name_b
            out += struct.pack('<I', n_bytes)
            out += struct.pack('<B', len(shape))
            for d in shape: out += struct.pack('<i', int(d))
            out += struct.pack('<d', float(scale))
            out += struct.pack('<I', 0)
        out += struct.pack('<Q', len(residual_compressed))
        out += residual_compressed
        out += sha
        return bytes(out)

    @staticmethod
    def decompress(recipe_bytes: bytes, device: str | None = None) -> tuple[np.ndarray, dict]:
        from siren_v5_torch import SIREN
        buf = recipe_bytes
        off = 0
        if buf[:4] != MAGIC_GRAYSCALE:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_GRAYSCALE
        bits = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        codec_id = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        codec_name = {0: 'zlib', 1: 'png', 2: 'webp'}[codec_id]
        hidden = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden_l = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        omega = struct.unpack('<f', buf[off:off+4])[0]; off += 4
        H = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        W = struct.unpack('<H', buf[off:off+2])[0]; off += 2

        packed_size = struct.unpack('<I', buf[off:off+4])[0]; off += 4
        packed = buf[off:off+packed_size]; off += packed_size
        n_meta = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        meta = []
        for _ in range(n_meta):
            name_len = struct.unpack('<B', buf[off:off+1])[0]; off += 1
            name = buf[off:off+name_len].decode('utf-8'); off += name_len
            n_bytes = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            ndim = struct.unpack('<B', buf[off:off+1])[0]; off += 1
            shape = tuple(struct.unpack('<' + 'i'*ndim, buf[off:off+4*ndim])); off += 4*ndim
            scale = struct.unpack('<d', buf[off:off+8])[0]; off += 8
            _ = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            meta.append((n_bytes, shape, scale, name))

        resid_size = struct.unpack('<Q', buf[off:off+8])[0]; off += 8
        residual_compressed = buf[off:off+resid_size]; off += resid_size
        sha_expected = buf[off:off+32]; off += 32

        weights = dequantize_int8(packed, meta)
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = SIREN(in_features=2, hidden_features=hidden,
                       hidden_layers=hidden_l, out_features=1,
                       omega_0=omega).to(dev)
        model.load_from_numpy(weights)
        model.eval()

        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing='ij',
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

        with torch.inference_mode():
            pred = model(coords).cpu().numpy()
        predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(H, W)

        if codec_id == 2:
            residual_img = decode_residual_webp_gray(residual_compressed)
        elif codec_id == 1:
            residual_img = decode_residual_png_gray(residual_compressed)
        else:
            raw = zlib.decompress(residual_compressed)
            residual_img = np.frombuffer(raw, dtype=np.uint8).reshape(H, W)

        recovered = ((predicted.astype(np.int16) + residual_img.astype(np.int16)) % 256).astype(np.uint8)
        sha_got = hashlib.sha256(recovered.tobytes()).digest()

        return recovered, {
            'H': H, 'W': W,
            'bits': bits,
            'residual_codec': codec_name,
            'sha256_match': sha_got == sha_expected,
            'exact_match': sha_got == sha_expected,
            'mode': 'grayscale',
        }


def _self_test():
    import zlib as _z
    print(f"[gray] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Generate grayscale MRI-like image
    SIZE = 256
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    gray = np.zeros((SIZE, SIZE), dtype=np.float32)
    for _ in range(3):
        kx, ky = rng.integers(1, 5, 2)
        amp = rng.uniform(40, 80)
        phase = rng.uniform(0, 2*np.pi)
        gray += amp * np.sin(2*np.pi*kx*xs + phase) * np.cos(2*np.pi*ky*ys)
    gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)

    orig = gray.nbytes
    zip_sz = len(_z.compress(gray.tobytes(), 9))
    print(f"[gray] Image: {gray.shape} = {orig:,}B (1 channel)")
    print(f"[gray] ZIP: {zip_sz:,}B ({orig/zip_sz:.2f}x)")

    # Grayscale BLKH
    comp = GrayscaleCompressor(hidden_features=32, hidden_layers=2,
                                omega_0=30.0, residual_codec='png')
    t0 = time.time()
    res = comp.compress_bitperfect(gray, epochs=100, lr=4e-3, bits=8,
                                     batch_size=16384, use_amp=True,
                                     patience=3, verbose=False)
    dt = time.time() - t0
    rec, meta = GrayscaleCompressor.decompress(res['recipe_bytes'])
    print(f"\n[gray] BLKH Grayscale: {res['recipe_size']:,}B  ({orig/res['recipe_size']:.2f}x)")
    print(f"  weights: {res['weights_packed_size']:,}B  residual: {res['residual_compressed_size']:,}B")
    print(f"  bit%: {res['model_bit_accuracy']:.1f}  SHA: {meta['exact_match']}  time: {dt:.2f}s")
    print(f"  vs ZIP: {zip_sz/res['recipe_size']:.2f}x")

    # Compare with RGB approach (3x larger)
    from siren_v5_hybrid import HybridCompressor
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    comp_rgb = HybridCompressor(auto_tune=True, residual_codec='png')
    t0 = time.time()
    res_rgb = comp_rgb.compress_bitperfect(gray_rgb, epochs=100, lr=4e-3, bits=8,
                                             batch_size=16384, use_amp=True,
                                             patience=3, verbose=False)
    dt_rgb = time.time() - t0
    print(f"\n[gray] BLKH RGB (3ch): {res_rgb['recipe_size']:,}B  time: {dt_rgb:.2f}s")
    print(f"[gray] Grayscale savings: {res_rgb['recipe_size'] - res['recipe_size']:,}B "
          f"({(1 - res['recipe_size']/res_rgb['recipe_size'])*100:.0f}% smaller)")


if __name__ == '__main__':
    _self_test()
