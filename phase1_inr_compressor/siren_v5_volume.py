# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

# BLKH-AUTH-DPS-2025-Kronos1027-darlan1027pc@gmail.com
"""
siren_v5_volume.py — v5.12 3D Volume compression (MRI/CT/scientific)
=====================================================================
Compress 3D volumetric data (MRI, CT, seismic, microscopy) using a SIREN
with 3D spatial coordinate: f(x, y, z) -> value.

Key insight: 3D volumes have HUGE spatial redundancy along all 3 axes.
A SIREN with continuous (x, y, z) input can exploit this much better than
slice-by-slice ZIP. Medical imaging datasets (MRI, CT) are the canonical
use case — gigabytes of smooth volumetric data.

Two modes:
  1. Single-channel (grayscale): f(x, y, z) -> scalar (1 output)
     Typical: MRI T1/T2/PD, CT density, seismic amplitude
  2. Multi-channel (RGB or multi-modal): f(x, y, z) -> vector
     Typical: colored microscopy, multi-modal MRI

Recipe format (.blk3):
  [magic 'BLK3'][version][bits][codec_id]
  [hidden_features][hidden_layers][omega_0]
  [D][H][W][C]  (depth, height, width, channels)
  [siren_packed + meta]
  [residual_compressed][sha]

Usage:
    comp = VolumeCompressor(hidden_features=64, hidden_layers=3)
    res = comp.compress(volume, epochs=3000)  # volume = D×H×W×C uint8
    recovered = comp.decompress(res['recipe_bytes'])
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
    encode_residual_zlib, decode_residual_zlib,
)


MAGIC_VOLUME = b'BLK3'
VERSION_VOLUME = 1


class VolumeSIREN(nn.Module):
    """SIREN with 3D spatial input: (x, y, z) -> value(s)."""

    def __init__(self, hidden_features=64, hidden_layers=3, omega_0=30.0,
                 out_features=1):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers

        layers = [SineLayer(3, hidden_features, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                     is_first=False, omega_0=omega_0))
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = float(np.sqrt(6.0 / hidden_features)) / omega_0
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def state_to_numpy(self):
        return {name: p.detach().cpu().numpy().astype(np.float32)
                for name, p in self.named_parameters()}

    def load_from_numpy(self, state):
        with torch.no_grad():
            for name, p in self.named_parameters():
                arr = np.asarray(state[name]).astype(np.float32)
                p.copy_(torch.from_numpy(arr))


class VolumeCompressor:
    """
    Compress 3D volumetric data using SIREN f(x, y, z) -> value(s).

    Pipeline:
      1. Train ONE SIREN on the entire volume
      2. Quantize weights (INT8)
      3. Inference -> predicted volume
      4. Residual = (original - predicted) mod 256, compressed with zlib
      5. SHA-256 verification

    For volumes, we use zlib residual (not WebP) because the residual is
    3D data, not a 2D image. A future version could use per-slice PNG.
    """

    def __init__(self, hidden_features=64, hidden_layers=3, omega_0=30.0,
                 device: str | None = None):
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = float(omega_0)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _make_model(self, out_features: int) -> VolumeSIREN:
        return VolumeSIREN(
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            omega_0=self.omega_0,
            out_features=out_features,
        ).to(self.device)

    def _make_coords(self, D: int, H: int, W: int) -> torch.Tensor:
        """Build (D*H*W, 3) coords: (x, y, z) per voxel.
        All in [-1, 1].
        """
        zs, ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, D, device=self.device),
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij',
        )
        return torch.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], dim=-1)

    def compress(self, volume: np.ndarray,
                 epochs: int = 3000, lr: float = 1e-3,
                 bits: int = 8,
                 batch_size: int | None = None,
                 use_amp: bool = False,
                 verbose: bool = False) -> dict:
        """Compress 3D volume. Volume shape: (D, H, W, C) uint8."""
        assert volume.dtype == np.uint8 and volume.ndim == 4
        D, H, W, C = volume.shape
        t0 = time.time()

        target_np = (volume.astype(np.float32) / 127.5 - 1.0).reshape(-1, C)
        target = torch.from_numpy(target_np).to(self.device)

        coords = self._make_coords(D, H, W)
        n_total = coords.shape[0]
        if batch_size is None or batch_size >= n_total:
            batch_size = min(8192, n_total) if self.device.type == 'cpu' else min(32768, n_total)

        model = self._make_model(out_features=C)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        warmup = max(1, epochs // 20)

        amp_dtype = None
        if use_amp:
            amp_dtype = torch.bfloat16 if self.device.type == 'cpu' else torch.float16

        model.train()
        history = []
        for epoch in range(epochs):
            if epoch < warmup:
                for g in opt.param_groups:
                    g['lr'] = lr * (epoch + 1) / warmup

            if batch_size < n_total:
                idx = torch.randint(0, n_total, (batch_size,), device=self.device)
                xb = coords[idx]; yb = target[idx]
            else:
                xb, yb = coords, target

            if amp_dtype is not None:
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                    pred = model(xb)
                    loss = torch.nn.functional.mse_loss(pred, yb)
            else:
                pred = model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch >= warmup:
                sched.step()

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  volume epoch {epoch}/{epochs}  loss={loss.item():.6e}")
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(float(loss.item()))

        train_time = time.time() - t0

        # Quantize weights
        weights_np = model.state_to_numpy()
        packed, packed_meta = quantize_int8(weights_np)
        q_weights = dequantize_int8(packed, packed_meta)
        model.load_from_numpy(q_weights)
        model.eval()

        # Inference (chunked)
        CHUNK = 1 << 18
        predicted_list = []
        with torch.no_grad():
            for i in range(0, n_total, CHUNK):
                if amp_dtype is not None:
                    with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                        pred_chunk = model(coords[i:i + CHUNK]).cpu().numpy()
                else:
                    pred_chunk = model(coords[i:i + CHUNK]).cpu().numpy()
                predicted_list.append(pred_chunk)
        predicted = np.concatenate(predicted_list, axis=0)
        predicted = np.clip((predicted + 1.0) * 127.5, 0, 255).astype(np.uint8)
        predicted = predicted.reshape(D, H, W, C)

        # Residual (zlib for 3D data — WebP/PNG are 2D only)
        original_bytes = volume.tobytes()
        predicted_bytes = predicted.tobytes()
        a = np.frombuffer(original_bytes, dtype=np.uint8)
        b = np.frombuffer(predicted_bytes, dtype=np.uint8)
        residual = (a ^ b).tobytes()
        residual_compressed = zlib.compress(residual, 9)

        sha = hashlib.sha256(original_bytes).digest()
        bit_acc = float(np.mean(np.unpackbits(a) == np.unpackbits(b))) * 100

        recipe = self._pack_recipe(bits, packed, packed_meta,
                                     D, H, W, C, residual_compressed, sha)

        return {
            'recipe_bytes': recipe,
            'recipe_size': len(recipe),
            'original_size': len(original_bytes),
            'weights_packed_size': len(packed),
            'residual_compressed_size': len(residual_compressed),
            'model_bit_accuracy': bit_acc,
            'train_time_s': train_time,
            'final_loss': history[-1] if history else float('nan'),
            'shape': (D, H, W, C),
            'sha256': sha.hex(),
        }

    def _pack_recipe(self, bits: int, packed: bytes, packed_meta: list,
                     D: int, H: int, W: int, C: int,
                     residual_compressed: bytes, sha: bytes) -> bytes:
        out = bytearray()
        out += MAGIC_VOLUME
        out += struct.pack('<B', VERSION_VOLUME)
        out += struct.pack('<B', bits)
        out += struct.pack('<H', self.hidden_features)
        out += struct.pack('<B', self.hidden_layers)
        out += struct.pack('<f', self.omega_0)
        out += struct.pack('<H', D)
        out += struct.pack('<H', H)
        out += struct.pack('<H', W)
        out += struct.pack('<B', C)
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
        buf = recipe_bytes
        off = 0
        if buf[:4] != MAGIC_VOLUME:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_VOLUME
        bits = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        hidden = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden_l = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        omega = struct.unpack('<f', buf[off:off+4])[0]; off += 4
        D = struct.unpack('<H', buf[off:off+2])[0]; off += 2
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
        model = VolumeSIREN(hidden_features=hidden, hidden_layers=hidden_l,
                              omega_0=omega, out_features=C).to(dev)
        model.load_from_numpy(weights)
        model.eval()

        # Build coords
        zs, ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, D, device=dev),
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing='ij',
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], dim=-1)

        # Inference chunked
        n_total = D * H * W
        CHUNK = 1 << 18
        preds = []
        with torch.no_grad():
            for i in range(0, n_total, CHUNK):
                preds.append(model(coords[i:i+CHUNK]).cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(D, H, W, C)
        predicted_bytes = predicted.tobytes()

        # Apply residual
        residual = zlib.decompress(residual_compressed)
        a = np.frombuffer(predicted_bytes, dtype=np.uint8)
        b = np.frombuffer(residual, dtype=np.uint8)
        recovered = (a ^ b).tobytes()
        sha_got = hashlib.sha256(recovered).digest()
        recovered = np.frombuffer(recovered, dtype=np.uint8).reshape(D, H, W, C)

        return recovered, {
            'D': D, 'H': H, 'W': W, 'C': C,
            'bits': bits,
            'sha256_match': sha_got == sha_expected,
            'exact_match': sha_got == sha_expected,
            'mode': 'volume',
            'shape': (D, H, W, C),
        }


# ============================================================
#  Self-test
# ============================================================
def _self_test():
    import zlib as _z
    print(f"[volume] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"[volume] Mode: 3D SIREN f(x,y,z) -> value")

    # Generate a synthetic 3D volume: smooth gaussian blob in 3D
    D, H, W, C = 16, 32, 32, 1  # small for fast test
    zs, ys, xs = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
    # Normalize to [-1, 1]
    zs = zs / (D - 1) * 2 - 1
    ys = ys / (H - 1) * 2 - 1
    xs = xs / (W - 1) * 2 - 1
    # 3D gaussian
    cx, cy, cz = 0.0, 0.0, 0.0
    sigma = 0.3
    vol = 200 * np.exp(-((xs - cx)**2 + (ys - cy)**2 + (zs - cz)**2) / (2 * sigma**2))
    # Add subtle variation
    vol += 30 * np.sin(xs * 5) * np.cos(ys * 4)
    vol = np.clip(vol, 0, 255).astype(np.uint8)
    vol = vol[..., None]  # add channel dim -> (D, H, W, 1)

    total_orig = vol.nbytes
    zip_size = len(_z.compress(vol.tobytes(), 9))
    print(f"[volume] Volume: {vol.shape} = {total_orig:,}B")
    print(f"[volume] ZIP: {zip_size:,}B ({total_orig/zip_size:.2f}x)")

    comp = VolumeCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
    t0 = time.time()
    res = comp.compress(vol, epochs=1000, lr=1e-3,
                          bits=8, batch_size=4096, verbose=True)
    dt = time.time() - t0

    print(f"\n[volume] BLKH Volume: {res['recipe_size']:,}B  "
          f"(ratio {total_orig/res['recipe_size']:.2f}x)")
    print(f"  SIREN weights:        {res['weights_packed_size']:,}B")
    print(f"  residual:             {res['residual_compressed_size']:,}B")
    print(f"  bit acc:              {res['model_bit_accuracy']:.1f}%")
    print(f"  train time:           {res['train_time_s']:.1f}s  total: {dt:.1f}s")

    # Verify
    t0 = time.time()
    recovered, meta = VolumeCompressor.decompress(res['recipe_bytes'])
    print(f"\n[volume] Decompress: {time.time()-t0:.1f}s  "
          f"exact_match: {meta['exact_match']}")
    print(f"  shape: {meta['shape']}")

    winner = "BLKH" if res['recipe_size'] < zip_size else "ZIP"
    print(f"\n[volume] vs ZIP: {zip_size/res['recipe_size']:.3f}x  -> {winner}")


if __name__ == '__main__':
    _self_test()
