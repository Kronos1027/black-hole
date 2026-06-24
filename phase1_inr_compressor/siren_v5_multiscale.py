# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

"""
siren_v5_multiscale.py — v5.15 Multi-scale SIREN
==================================================
Uses multiple SIREN branches with different omega_0 frequencies in parallel,
then concatenates their outputs through a fusion layer. This allows the
network to represent both low-frequency (smooth gradients) and high-frequency
(detail, texture) content simultaneously.

Architecture:
  Branch 1: SIREN(omega_0=15)  -> captures smooth, large-scale structure
  Branch 2: SIREN(omega_0=30)  -> captures medium-scale structure
  Branch 3: SIREN(omega_0=60)  -> captures fine detail
  Fusion:   Linear(3*hidden, out) -> combines all scales

The fusion layer learns which scale to weight for each spatial location.
This is similar to multi-scale GANs and wavelet decompositions.

Expected improvement: 10-30% better bit accuracy on mixed-frequency content
(smooth + textured images), leading to smaller residuals.
"""
from __future__ import annotations
import os
import sys
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


MAGIC_MULTISCALE = b'BLKM'  # BLK Multi-scale
VERSION_MULTISCALE = 1


class MultiScaleSIREN(nn.Module):
    """Multi-scale SIREN with parallel branches at different omega_0."""

    def __init__(self, in_features=2, hidden_features=32, hidden_layers=2,
                 out_features=3, omegas=(15.0, 30.0, 60.0)):
        super().__init__()
        self.omegas = tuple(float(o) for o in omegas)
        self.n_scales = len(omegas)
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers

        # Create one SIREN branch per scale
        self.branches = nn.ModuleList()
        for omega in self.omegas:
            layers = [SineLayer(in_features, hidden_features,
                                is_first=True, omega_0=omega)]
            for _ in range(hidden_layers):
                layers.append(SineLayer(hidden_features, hidden_features,
                                         is_first=False, omega_0=omega))
            self.branches.append(nn.Sequential(*layers))

        # Fusion layer: concatenate all branch outputs -> final output
        fusion_input = hidden_features * self.n_scales
        self.fusion = nn.Linear(fusion_input, out_features)
        # Initialize fusion with small weights (start with balanced contribution)
        nn.init.xavier_uniform_(self.fusion.weight, gain=0.5)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, x):
        # Run all branches in parallel
        branch_outputs = [branch(x) for branch in self.branches]
        # Concatenate along feature dimension
        combined = torch.cat(branch_outputs, dim=-1)
        # Fusion layer produces final output (no activation — linear)
        return self.fusion(combined)

    def state_to_numpy(self):
        return {name: p.detach().cpu().numpy().astype(np.float32)
                for name, p in self.named_parameters()}

    def load_from_numpy(self, state):
        with torch.no_grad():
            for name, p in self.named_parameters():
                arr = np.asarray(state[name]).astype(np.float32)
                p.copy_(torch.from_numpy(arr))


class MultiScaleCompressor:
    """
    Multi-scale SIREN compressor with hybrid WebP residual.
    Bit-perfect roundtrip, potentially better quality than single-scale.
    """

    def __init__(self, hidden_features=32, hidden_layers=2,
                 omegas=(15.0, 30.0, 60.0),
                 omega_0=30.0,  # kept for compat, unused
                 residual_codec: str = 'webp',
                 device: str | None = None):
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omegas = tuple(float(o) for o in omegas)
        self.residual_codec = residual_codec
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: MultiScaleSIREN | None = None
        self.H = None
        self.W = None
        self.C = None

    def _make_model(self):
        return MultiScaleSIREN(
            in_features=2, hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers, out_features=3,
            omegas=self.omegas,
        ).to(self.device)

    def _make_coords(self, H, W):
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij',
        )
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    def compress_bitperfect(self, image_array, epochs=1500, lr=1e-3,
                             bits=8, batch_size=2048,
                             use_amp=False, verbose=False):
        assert image_array.dtype == np.uint8 and image_array.ndim == 3
        H, W, C = image_array.shape
        self.H, self.W, self.C = H, W, C
        original_bytes = image_array.tobytes()

        coords = self._make_coords(H, W)
        values = torch.from_numpy(
            (image_array.astype(np.float32) / 127.5 - 1.0).reshape(-1, C)
        ).to(self.device)

        N = coords.shape[0]
        if batch_size >= N:
            batch_size = N

        model = self._make_model()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        warmup = max(1, epochs // 20)

        amp_dtype = None
        if use_amp:
            amp_dtype = torch.bfloat16 if self.device.type == 'cpu' else torch.float16

        model.train()
        history = []
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

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  multiscale epoch {epoch}/{epochs}  loss={loss.item():.6e}")
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(float(loss.item()))

        train_time = time.time() - t0
        self._model = model

        # Quantize weights
        weights_np = model.state_to_numpy()
        packed, packed_meta = quantize_int8(weights_np)
        q_weights = dequantize_int8(packed, packed_meta)
        model.load_from_numpy(q_weights)
        model.eval()

        # Inference (always FP32 for consistency, even if trained with AMP)
        t0 = time.time()
        with torch.no_grad():
            pred = model(coords).cpu().numpy()
        predict_time = time.time() - t0
        predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(H, W, C)

        # Hybrid residual
        residual_img = ((image_array.astype(np.int16) - predicted.astype(np.int16)) % 256).astype(np.uint8)
        recovered_check = ((predicted.astype(np.int16) + residual_img.astype(np.int16)) % 256).astype(np.uint8)
        assert np.array_equal(recovered_check, image_array), "residual math failed"

        if self.residual_codec == 'webp':
            residual_compressed = encode_residual_webp(residual_img)
        elif self.residual_codec == 'png':
            residual_compressed = encode_residual_png(residual_img)
        else:
            import zlib as _z
            residual_compressed = _z.compress(residual_img.tobytes(), 9)

        sha = hashlib.sha256(original_bytes).digest()

        # Stats
        a = np.frombuffer(original_bytes, dtype=np.uint8)
        b = np.frombuffer(predicted.tobytes(), dtype=np.uint8)
        bit_acc = float(np.mean(np.unpackbits(a) == np.unpackbits(b))) * 100

        # Pack recipe
        recipe = self._pack_recipe(bits, packed, packed_meta,
                                     residual_compressed, sha)

        return {
            'recipe_bytes': recipe,
            'original_size': len(original_bytes),
            'recipe_size': len(recipe),
            'weights_packed_size': len(packed),
            'residual_compressed_size': len(residual_compressed),
            'model_bit_accuracy': bit_acc,
            'train_time_s': train_time,
            'predict_time_s': predict_time,
            'sha256': sha.hex(),
            'omegas': self.omegas,
            'mode': 'multiscale',
        }

    def _pack_recipe(self, bits, packed, packed_meta,
                     residual_compressed, sha):
        out = bytearray()
        out += MAGIC_MULTISCALE
        out += struct.pack('<B', VERSION_MULTISCALE)
        out += struct.pack('<B', bits)
        codec_id = {'zlib': 0, 'png': 1, 'webp': 2}[self.residual_codec]
        out += struct.pack('<B', codec_id)
        out += struct.pack('<H', self.hidden_features)
        out += struct.pack('<B', self.hidden_layers)
        out += struct.pack('<B', len(self.omegas))
        for o in self.omegas:
            out += struct.pack('<f', o)
        out += struct.pack('<H', self.H)
        out += struct.pack('<H', self.W)
        out += struct.pack('<B', self.C)
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
    def decompress(recipe_bytes, device=None):
        buf = recipe_bytes
        off = 0
        if buf[:4] != MAGIC_MULTISCALE:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_MULTISCALE
        bits = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        codec_id = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        codec_name = {0: 'zlib', 1: 'png', 2: 'webp'}[codec_id]
        hidden = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden_l = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        n_omegas = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        omegas = []
        for _ in range(n_omegas):
            omegas.append(struct.unpack('<f', buf[off:off+4])[0]); off += 4
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
            _ = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            meta.append((n_bytes, shape, scale, name))

        resid_size = struct.unpack('<Q', buf[off:off+8])[0]; off += 8
        residual_compressed = buf[off:off+resid_size]; off += resid_size
        sha_expected = buf[off:off+32]; off += 32

        weights = dequantize_int8(packed, meta)
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = MultiScaleSIREN(in_features=2, hidden_features=hidden,
                                  hidden_layers=hidden_l, out_features=C,
                                  omegas=omegas).to(dev)
        model.load_from_numpy(weights)
        model.eval()

        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing='ij',
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

        with torch.no_grad():
            pred = model(coords).cpu().numpy()
        predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(H, W, C)

        if codec_id == 2:
            residual_img = decode_residual_webp(residual_compressed)
        elif codec_id == 1:
            residual_img = decode_residual_png(residual_compressed)
        else:
            import zlib as _z
            raw = _z.decompress(residual_compressed)
            residual_img = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, C)

        recovered = ((predicted.astype(np.int16) + residual_img.astype(np.int16)) % 256).astype(np.uint8)
        sha_got = hashlib.sha256(recovered.tobytes()).digest()

        return recovered, {
            'H': H, 'W': W, 'C': C,
            'bits': bits,
            'omegas': omegas,
            'residual_codec': codec_name,
            'sha256_match': sha_got == sha_expected,
            'exact_match': sha_got == sha_expected,
            'mode': 'multiscale',
        }


def _self_test():
    import zlib as _z
    print(f"[multiscale] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"[multiscale] Mode: Multi-scale SIREN (omega=15,30,60) + WebP residual")

    # Test image: gradient + high-freq detail (mixed frequency content)
    SIZE = 128
    img = np.zeros((SIZE, SIZE, 3), dtype=np.int32)
    for i in range(SIZE):
        for j in range(SIZE):
            # Low-freq gradient + high-freq pattern (mixed)
            low = i * 2
            high = int(30 * np.sin(i * 0.5) * np.sin(j * 0.3))
            img[i, j] = [low + high, j * 2, i + j]
    img = np.clip(img, 0, 255).astype(np.uint8)

    orig = img.nbytes
    zip_sz = len(_z.compress(img.tobytes(), 9))
    print(f"[multiscale] Image: {img.shape} = {orig:,}B (mixed freq content)")
    print(f"[multiscale] ZIP: {zip_sz:,}B ({orig/zip_sz:.2f}x)")

    # Single-scale (baseline) — use HybridCompressor
    from siren_v5_hybrid import HybridCompressor
    comp_single = HybridCompressor(hidden_features=32, hidden_layers=2,
                                     residual_codec='webp')
    res_single = comp_single.compress_bitperfect(img, epochs=800, lr=1e-3,
                                                   bits=8, batch_size=2048,
                                                   use_amp=True, verbose=False)
    rec_single, meta_single = HybridCompressor.decompress(res_single['recipe_bytes'])

    # Multi-scale
    comp_multi = MultiScaleCompressor(hidden_features=32, hidden_layers=2,
                                       omegas=(15.0, 30.0, 60.0),
                                       residual_codec='webp')
    res_multi = comp_multi.compress_bitperfect(img, epochs=800, lr=1e-3,
                                                 bits=8, batch_size=2048,
                                                 use_amp=True, verbose=True)
    rec_multi, meta_multi = MultiScaleCompressor.decompress(res_multi['recipe_bytes'])

    print(f"\n[multiscale] Results:")
    print(f"  Single-scale:  {res_single['recipe_size']:>7,}B  bit%={res_single['model_bit_accuracy']:.1f}  "
          f"SHA={meta_single['exact_match']}  vs ZIP={zip_sz/res_single['recipe_size']:.2f}x")
    print(f"  Multi-scale:   {res_multi['recipe_size']:>7,}B  bit%={res_multi['model_bit_accuracy']:.1f}  "
          f"SHA={meta_multi['exact_match']}  vs ZIP={zip_sz/res_multi['recipe_size']:.2f}x")
    
    improvement = (1 - res_multi['recipe_size'] / res_single['recipe_size']) * 100
    print(f"  Improvement:   {improvement:+.1f}% {'(multi-scale wins)' if improvement > 0 else '(single-scale wins)'}")


if __name__ == '__main__':
    _self_test()
