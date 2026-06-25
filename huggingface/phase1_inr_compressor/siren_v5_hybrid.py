# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

# BLKH-AUTH-DPS-2025-Kronos1027-darlan1027pc@gmail.com
"""
siren_v5_hybrid.py — v5.8 Hybrid mode: SIREN + image-codec residual
====================================================================
Instead of using XOR + zlib for the residual (which treats it as random
bytes), we encode the residual as an IMAGE using PNG or WebP lossless.
The residual = (original - predicted) mod 256 is a 3-channel image that
image codecs are designed to compress efficiently.

Pipeline:
  1. Train SIREN (same as bit-perfect mode)
  2. Quantize weights, reload, inference -> predicted
  3. residual_image = (original - predicted) mod 256  (uint8, 3 channels)
  4. Encode residual_image with PNG or WebP lossless
  5. Pack: SIREN weights + encoded residual + SHA-256

Decompress:
  1. Unpack, dequantize SIREN weights
  2. Inference -> predicted
  3. Decode residual_image from PNG/WebP
  4. recovered = (predicted + residual_image) mod 256
  5. Verify SHA-256

Expected improvement: PNG/WebP lossless on the residual should be 2-3x
smaller than zlib on XOR bytes, because image codecs use 2D prediction
filters that capture the structure of the residual (which is NOT random —
it's the SIREN's prediction error, often smooth).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siren_v5_torch import (
    ImageINRv5, SIREN, quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4, prune_weights,
    MAGIC_V5, VERSION_V5,
)


MAGIC_HYBRID = b'BLK8'
VERSION_HYBRID = 1


def encode_residual_png(residual_img: np.ndarray) -> bytes:
    """Encode residual as PNG (lossless image codec)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(residual_img).save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def decode_residual_png(data: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    return np.array(img, dtype=np.uint8)


def encode_residual_webp(residual_img: np.ndarray) -> bytes:
    """Encode residual as WebP lossless (often smaller than PNG)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(residual_img).save(buf, format='WebP', lossless=True)
    return buf.getvalue()


def decode_residual_webp(data: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    return np.array(img, dtype=np.uint8)


def encode_residual_zlib(residual_img: np.ndarray) -> bytes:
    """Fallback: zlib on raw bytes (original method)."""
    return zlib.compress(residual_img.tobytes(), 9)


def decode_residual_zlib(data: bytes, shape: tuple) -> np.ndarray:
    raw = zlib.decompress(data)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


class HybridCompressor:
    """
    SIREN + image-codec residual (PNG/WebP lossless).
    Bit-perfect roundtrip, smaller than XOR+zlib on natural images.
    """

    def __init__(self, hidden_features=32, hidden_layers=2, omega_0=30.0,
                 residual_codec: str = 'webp',
                 auto_tune: bool = False,
                 device: str | None = None):
        """
        Args:
            residual_codec: 'webp' (best), 'png' (fallback), 'zlib' (original)
            auto_tune: If True, automatically pick hidden_features and hidden_layers
                       based on image size (from scaling benchmarks).
        """
        self._auto_tune = auto_tune
        self._user_hidden = hidden_features
        self._user_layers = hidden_layers
        self.inner = ImageINRv5(
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            omega_0=omega_0,
            device=device,
        )
        self.residual_codec = residual_codec

    @staticmethod
    def _auto_tune_config(H: int, W: int) -> tuple[int, int]:
        """Pick optimal SIREN size based on image dimensions.
        Derived from scaling benchmarks:
          < 256px:   h=32, l=2 (fast, sufficient)
          256-512px: h=64, l=3 (sweet spot, 4.5x vs ZIP)
          > 512px:   h=128, l=3 (more capacity for large images)
        """
        max_dim = max(H, W)
        if max_dim <= 256:
            return 32, 2
        elif max_dim <= 512:
            return 64, 3
        else:
            return 128, 3

    def compress_bitperfect(self, image_array: np.ndarray,
                            epochs: int = 1500, lr: float = 1e-3,
                            bits: int = 8, prune_threshold: float = 0.0,
                            batch_size: int | None = None,
                            use_amp: bool = False,
                            patience: int = 0,
                            verbose: bool = False) -> dict:
        """Compress with image-codec residual instead of XOR+zlib."""
        assert image_array.dtype == np.uint8 and image_array.ndim == 3
        H, W, C = image_array.shape
        original_bytes = image_array.tobytes()

        # Auto-tune SIREN size if enabled
        if self._auto_tune:
            tuned_h, tuned_l = self._auto_tune_config(H, W)
            if tuned_h != self.inner.hidden_features or tuned_l != self.inner.hidden_layers:
                if verbose:
                    print(f"  [auto-tune] h={tuned_h}, l={tuned_l} for {H}x{W} image")
                self.inner = ImageINRv5(
                    hidden_features=tuned_h,
                    hidden_layers=tuned_l,
                    omega_0=self.inner.omega_0,
                    device=str(self.inner.device),
                )

        # 1. Train SIREN
        meta = self.inner.compress(image_array, epochs=epochs, lr=lr,
                                     batch_size=batch_size, use_amp=use_amp,
                                     patience=patience, verbose=verbose)

        # 2. Quantize weights (reload to match decompress)
        weights_np = self.inner._model.state_to_numpy()
        if prune_threshold > 0:
            weights_np = prune_weights(weights_np, prune_threshold)
        if bits == 8:
            packed, packed_meta = quantize_int8(weights_np)
            q_weights = dequantize_int8(packed, packed_meta)
        else:
            packed, packed_meta = quantize_int4(weights_np)
            q_weights = dequantize_int4(packed, packed_meta)
        self.inner._model.load_from_numpy(q_weights)
        self.inner._model.eval()

        # 3. Inference
        t0 = time.time()
        predicted = self.inner.reconstruct()
        predict_time = time.time() - t0

        # 4. Compute residual as IMAGE (mod 256 difference)
        # This is the key: residual_img is a valid uint8 image
        residual_img = ((image_array.astype(np.int16) - predicted.astype(np.int16)) % 256).astype(np.uint8)
        # Sanity: adding back should recover original
        recovered_check = ((predicted.astype(np.int16) + residual_img.astype(np.int16)) % 256).astype(np.uint8)
        assert np.array_equal(recovered_check, image_array), "residual math failed"

        # 5. Encode residual with chosen codec
        if self.residual_codec == 'webp':
            residual_compressed = encode_residual_webp(residual_img)
        elif self.residual_codec == 'png':
            residual_compressed = encode_residual_png(residual_img)
        else:
            residual_compressed = encode_residual_zlib(residual_img)

        # 6. SHA-256
        sha = hashlib.sha256(original_bytes).digest()

        # 7. Pack recipe (reuse BLK5 format but with codec tag)
        recipe = self._pack_recipe(bits, packed, packed_meta,
                                    residual_compressed, sha)

        # 8. Stats
        a = np.frombuffer(original_bytes, dtype=np.uint8)
        b = np.frombuffer(predicted.tobytes(), dtype=np.uint8)
        bit_acc = float(np.mean(np.unpackbits(a) == np.unpackbits(b))) * 100

        return {
            'recipe_bytes': recipe,
            'original_size': len(original_bytes),
            'recipe_size': len(recipe),
            'weights_packed_size': len(packed),
            'residual_compressed_size': len(residual_compressed),
            'residual_raw_size': residual_img.nbytes,
            'residual_codec': self.residual_codec,
            'model_bit_accuracy': bit_acc,
            'train_time_s': meta['train_time_s'],
            'predict_time_s': predict_time,
            'psnr_db': meta['psnr'],
            'sha256': sha.hex(),
        }

    def _pack_recipe(self, bits: int, packed: bytes, packed_meta: list,
                     residual_compressed: bytes, sha: bytes) -> bytes:
        out = bytearray()
        out += MAGIC_HYBRID
        out += struct.pack('<B', VERSION_HYBRID)
        out += struct.pack('<B', bits)
        # codec tag: 0=zlib, 1=png, 2=webp
        codec_id = {'zlib': 0, 'png': 1, 'webp': 2}[self.residual_codec]
        out += struct.pack('<B', codec_id)
        out += struct.pack('<B', self.inner.in_features)
        out += struct.pack('<H', self.inner.hidden_features)
        out += struct.pack('<B', self.inner.hidden_layers)
        out += struct.pack('<B', self.inner.out_features)
        out += struct.pack('<f', self.inner.omega_0)
        out += struct.pack('<H', self.inner.H)
        out += struct.pack('<H', self.inner.W)
        out += struct.pack('<B', self.inner.C)
        # weights
        out += struct.pack('<I', len(packed))
        out += packed
        out += struct.pack('<H', len(packed_meta))
        for entry in packed_meta:
            if bits == 8:
                n_bytes, shape, scale, name = entry
                n_elem = 0
            else:
                n_bytes, shape, scale, n_elem, name = entry
            name_b = name.encode('utf-8')
            out += struct.pack('<B', len(name_b)); out += name_b
            out += struct.pack('<I', n_bytes)
            out += struct.pack('<B', len(shape))
            for d in shape: out += struct.pack('<i', int(d))
            out += struct.pack('<d', float(scale))
            out += struct.pack('<I', n_elem)
        # residual + sha
        out += struct.pack('<Q', len(residual_compressed))
        out += residual_compressed
        out += sha
        return bytes(out)

    @staticmethod
    def decompress(recipe_bytes: bytes, device: str | None = None) -> tuple[np.ndarray, dict]:
        buf = recipe_bytes
        off = 0
        if buf[:4] != MAGIC_HYBRID:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_HYBRID
        bits = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        codec_id = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        codec_name = {0: 'zlib', 1: 'png', 2: 'webp'}[codec_id]
        in_features = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        hidden_features = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden_layers = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        out_features = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        omega_0 = struct.unpack('<f', buf[off:off+4])[0]; off += 4
        H = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        W = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        C = struct.unpack('<B', buf[off:off+1])[0]; off += 1

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
            n_elem = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            if bits == 8:
                meta.append((n_bytes, shape, scale, name))
            else:
                meta.append((n_bytes, shape, scale, n_elem, name))

        resid_size = struct.unpack('<Q', buf[off:off+8])[0]; off += 8
        residual_compressed = buf[off:off+resid_size]; off += resid_size
        sha_expected = buf[off:off+32]; off += 32

        # Dequantize weights
        if bits == 8:
            weights = dequantize_int8(packed, meta)
        else:
            weights = dequantize_int4(packed, meta)

        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = SIREN(in_features=in_features,
                      hidden_features=hidden_features,
                      hidden_layers=hidden_layers,
                      out_features=out_features,
                      omega_0=omega_0).to(dev)
        model.load_from_numpy(weights)
        model.eval()

        # Inference (use inference_mode for maximum speed — faster than no_grad)
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing="ij",
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
        with torch.inference_mode():
            pred = model(coords).cpu().numpy()
        # pred has shape (H*W, 3) — reshape to (H, W, 3)
        predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(H, W, C)

        # Decode residual
        if codec_id == 2:  # webp
            residual_img = decode_residual_webp(residual_compressed)
        elif codec_id == 1:  # png
            residual_img = decode_residual_png(residual_compressed)
        else:  # zlib
            residual_img = decode_residual_zlib(residual_compressed, (H, W, C))

        # Recover: (predicted + residual) mod 256
        recovered = ((predicted.astype(np.int16) + residual_img.astype(np.int16)) % 256).astype(np.uint8)
        sha_got = hashlib.sha256(recovered.tobytes()).digest()

        return recovered, {
            'H': H, 'W': W, 'C': C,
            'bits': bits,
            'residual_codec': codec_name,
            'residual_compressed_size': resid_size,
            'sha256_match': sha_got == sha_expected,
            'exact_match': sha_got == sha_expected,
            'mode': 'hybrid_bitperfect',
            'lossless': True,
        }


def _self_test():
    import zlib as _z
    print(f"[hybrid] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Smooth image
    SIZE = 128
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for i in range(SIZE):
        for j in range(SIZE):
            img[i, j] = [int(i * 2), int(j * 2), int((i + j))]

    orig = img.nbytes
    zip_size = len(_z.compress(img.tobytes(), 9))
    print(f"[hybrid] Image: {img.shape} = {orig:,}B")
    print(f"[hybrid] ZIP: {zip_size:,}B ({orig/zip_size:.2f}x)")

    # Test all 3 codecs
    for codec in ['zlib', 'png', 'webp']:
        print(f"\n--- codec={codec} ---")
        comp = HybridCompressor(hidden_features=32, hidden_layers=2,
                                 omega_0=30.0, residual_codec=codec)
        t0 = time.time()
        res = comp.compress_bitperfect(img, epochs=1000, lr=1e-3,
                                         bits=8, batch_size=2048, verbose=False)
        dt = time.time() - t0

        recon, meta = HybridCompressor.decompress(res['recipe_bytes'])
        assert meta['exact_match'], f"SHA-256 failed for {codec}"

        print(f"  recipe: {res['recipe_size']:,}B  weights: {res['weights_packed_size']:,}B  "
              f"residual: {res['residual_compressed_size']:,}B  bit%: {res['model_bit_accuracy']:.1f}  "
              f"SHA: {meta['exact_match']}  {dt:.1f}s")
        winner = "BLKH" if res['recipe_size'] < zip_size else "ZIP"
        print(f"  vs ZIP: {zip_size/res['recipe_size']:.3f}x  -> {winner}")


if __name__ == '__main__':
    _self_test()
