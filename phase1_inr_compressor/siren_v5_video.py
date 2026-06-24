# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

# BLKH-AUTH-DPS-2025-Kronos1027-darlan1027pc@gmail.com
"""
siren_v5_video.py — v5.11 Video compression with temporal SIREN
=================================================================
Compress a video (sequence of frames) using a SINGLE SIREN with temporal
coordinate: f(x, y, t) -> RGB.

Key insight: video frames have HUGE temporal redundancy. Adjacent frames
differ by small motion + small lighting changes. A SIREN with continuous
time coordinate t can exploit this:

  - SIREN learns the "video function" — given any (x, y, t), produce RGB
  - Adjacent frames share most of the SIREN's representational capacity
  - Per-frame: only a small WebP residual is needed to correct errors
  - Total recipe: shared SIREN weights + N small residuals

This is the NeRV (Neural Representations for Videos) approach, simplified
to use SIREN instead of convolutional decoder.

Recipe format (.blkv):
  [magic 'BLKV'][version][bits][codec_id]
  [hidden_features][hidden_layers][omega_0]
  [H][W][C][N_frames][fps]
  [siren_packed + meta]
  for each frame: [residual_compressed][sha]

Usage:
    comp = VideoCompressor(hidden_features=64, hidden_layers=3)
    res = comp.compress(frames, epochs=3000)  # frames = list of HxWx3 uint8
    recovered = comp.decompress(res['recipe_bytes'])  # list of HxWx3 uint8
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
    encode_residual_zlib, decode_residual_zlib,
)


MAGIC_VIDEO = b'BLKV'
VERSION_VIDEO = 1


# ============================================================
#  Temporal SIREN: f(x, y, t) -> RGB
# ============================================================
class TemporalSIREN(nn.Module):
    """SIREN with 3D input: (x, y, t) -> RGB.
    The t coordinate is scaled by a separate omega_t to control temporal
    frequency sensitivity.
    """

    def __init__(self, hidden_features=64, hidden_layers=3, omega_0=30.0,
                 omega_t=1.0):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.omega_t = float(omega_t)
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers

        # 3 input features: (x, y, t)
        layers = [SineLayer(3, hidden_features, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                     is_first=False, omega_0=omega_0))
        final = nn.Linear(hidden_features, 3)
        with torch.no_grad():
            bound = float(np.sqrt(6.0 / hidden_features)) / omega_0
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x has shape (N, 3) = (x, y, t)
        # Scale the t coordinate by omega_t before the first layer
        # The SineLayer(is_first=True) doesn't apply omega_0, so we apply omega_t here
        x_scaled = x.clone()
        x_scaled[:, 2] = x[:, 2] * self.omega_t
        return self.net(x_scaled)

    def state_to_numpy(self):
        return {name: p.detach().cpu().numpy().astype(np.float32)
                for name, p in self.named_parameters()}

    def load_from_numpy(self, state):
        with torch.no_grad():
            for name, p in self.named_parameters():
                arr = np.asarray(state[name]).astype(np.float32)
                p.copy_(torch.from_numpy(arr))


# ============================================================
#  Video compressor
# ============================================================
class VideoCompressor:
    """
    Compress N video frames into a single .blkv recipe.

    Pipeline:
      1. Train ONE TemporalSIREN on all frames: f(x, y, t) -> RGB
         where t in [-1, 1] maps linearly to frame index
      2. Quantize SIREN weights (shared across ALL frames)
      3. Per frame: inference + WebP residual + SHA-256

    The SIREN exploits temporal redundancy — adjacent frames share
    representational capacity, so per-frame cost is small.
    """

    def __init__(self, hidden_features=64, hidden_layers=3,
                 omega_0=30.0, omega_t=1.0,
                 residual_codec: str = 'webp',
                 device: str | None = None):
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = float(omega_0)
        self.omega_t = float(omega_t)
        self.residual_codec = residual_codec
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _make_model(self):
        return TemporalSIREN(
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            omega_0=self.omega_0,
            omega_t=self.omega_t,
        ).to(self.device)

    def _make_coords(self, H: int, W: int, N: int) -> torch.Tensor:
        """Build (N*H*W, 3) coords: (x, y, t) per pixel per frame.
        x, y in [-1, 1]; t in [-1, 1] linearly spaced across N frames.
        """
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij',
        )
        xy = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # (H*W, 2)
        xy_rep = xy.repeat(N, 1)  # (N*H*W, 2)
        if N > 1:
            t_vals = torch.linspace(-1, 1, N, device=self.device)
        else:
            t_vals = torch.tensor([0.0], device=self.device)
        t_rep = t_vals.repeat_interleave(H * W).reshape(-1, 1)  # (N*H*W, 1)
        return torch.cat([xy_rep, t_rep], dim=-1)  # (N*H*W, 3)

    def compress(self, frames: list[np.ndarray],
                 epochs: int = 3000, lr: float = 1e-3,
                 bits: int = 8,
                 batch_size: int | None = None,
                 use_amp: bool = False,
                 verbose: bool = False) -> dict:
        """Compress N video frames into a single .blkv recipe."""
        assert all(f.shape == frames[0].shape for f in frames), "all frames must have same shape"
        assert frames[0].dtype == np.uint8 and frames[0].ndim == 3
        H, W, C = frames[0].shape
        assert C == 3
        N = len(frames)
        t0 = time.time()

        # Stack all frames into one big target tensor
        stacked = np.stack(frames, axis=0)  # (N, H, W, 3)
        target_np = (stacked.astype(np.float32) / 127.5 - 1.0).reshape(-1, 3)
        target = torch.from_numpy(target_np).to(self.device)

        coords = self._make_coords(H, W, N)  # (N*H*W, 3)
        n_total = coords.shape[0]
        if batch_size is None or batch_size >= n_total:
            batch_size = n_total if self.device.type == 'cpu' else min(16384, n_total)

        model = self._make_model()
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
                print(f"  video epoch {epoch}/{epochs}  loss={loss.item():.6e}")
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(float(loss.item()))

        train_time = time.time() - t0

        # Quantize weights (shared across ALL frames)
        weights_np = model.state_to_numpy()
        packed, packed_meta = quantize_int8(weights_np)
        q_weights = dequantize_int8(packed, packed_meta)
        model.load_from_numpy(q_weights)
        model.eval()

        # Inference for ALL frames (chunked)
        CHUNK = 1 << 18
        predicted_list = []
        with torch.no_grad():
            for i in range(0, n_total, CHUNK):
                pred_chunk = model(coords[i:i + CHUNK]).cpu().numpy()
                predicted_list.append(pred_chunk)
        predicted = np.concatenate(predicted_list, axis=0)
        predicted = np.clip((predicted + 1.0) * 127.5, 0, 255).astype(np.uint8)
        predicted = predicted.reshape(N, H, W, 3)

        # Per-frame residual (hybrid: WebP)
        per_frame_data = []
        total_residual = 0
        bit_pcts = []
        for i in range(N):
            orig_bytes = frames[i].tobytes()
            pred_bytes = predicted[i].tobytes()
            # Hybrid residual: image difference
            residual_img = ((frames[i].astype(np.int16) - predicted[i].astype(np.int16)) % 256).astype(np.uint8)
            # Sanity
            recovered_check = ((predicted[i].astype(np.int16) + residual_img.astype(np.int16)) % 256).astype(np.uint8)
            assert np.array_equal(recovered_check, frames[i]), f"frame {i}: residual math failed"

            if self.residual_codec == 'webp':
                residual_compressed = encode_residual_webp(residual_img)
            elif self.residual_codec == 'png':
                residual_compressed = encode_residual_png(residual_img)
            else:
                residual_compressed = encode_residual_zlib(residual_img)

            sha = hashlib.sha256(orig_bytes).digest()
            a = np.frombuffer(orig_bytes, dtype=np.uint8)
            b = np.frombuffer(pred_bytes, dtype=np.uint8)
            bit_acc = float(np.mean(np.unpackbits(a) == np.unpackbits(b))) * 100

            per_frame_data.append({
                'residual_compressed': residual_compressed,
                'sha': sha,
            })
            total_residual += len(residual_compressed)
            bit_pcts.append(bit_acc)

        # Pack recipe
        recipe = self._pack_recipe(bits, packed, packed_meta,
                                     H, W, C, N, per_frame_data)

        total_orig = sum(f.nbytes for f in frames)
        return {
            'recipe_bytes': recipe,
            'recipe_size': len(recipe),
            'n_frames': N,
            'fps': 30,  # default; user can override
            'weights_packed_size': len(packed),
            'residual_total': total_residual,
            'residual_per_frame': total_residual / N,
            'avg_bit_pct': float(np.mean(bit_pcts)),
            'min_bit_pct': float(np.min(bit_pcts)),
            'max_bit_pct': float(np.max(bit_pcts)),
            'train_time_s': train_time,
            'total_orig': total_orig,
            'final_loss': history[-1] if history else float('nan'),
            'residual_codec': self.residual_codec,
        }

    def _pack_recipe(self, bits: int, packed: bytes, packed_meta: list,
                     H: int, W: int, C: int, N: int,
                     per_frame: list[dict]) -> bytes:
        out = bytearray()
        out += MAGIC_VIDEO
        out += struct.pack('<B', VERSION_VIDEO)
        out += struct.pack('<B', bits)
        codec_id = {'zlib': 0, 'png': 1, 'webp': 2}[self.residual_codec]
        out += struct.pack('<B', codec_id)
        out += struct.pack('<H', self.hidden_features)
        out += struct.pack('<B', self.hidden_layers)
        out += struct.pack('<f', self.omega_0)
        out += struct.pack('<f', self.omega_t)
        out += struct.pack('<H', H)
        out += struct.pack('<H', W)
        out += struct.pack('<B', C)
        out += struct.pack('<H', N)
        out += struct.pack('<H', 30)  # fps default
        # SIREN weights (shared across all frames)
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
        # Per-frame: residual + sha
        for d in per_frame:
            out += struct.pack('<Q', len(d['residual_compressed']))
            out += d['residual_compressed']
            out += d['sha']
        return bytes(out)

    @staticmethod
    def decompress(recipe_bytes: bytes, device: str | None = None) -> tuple[list[np.ndarray], dict]:
        buf = recipe_bytes
        off = 0
        if buf[:4] != MAGIC_VIDEO:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_VIDEO
        bits = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        codec_id = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        codec_name = {0: 'zlib', 1: 'png', 2: 'webp'}[codec_id]
        hidden = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden_l = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        omega_0 = struct.unpack('<f', buf[off:off+4])[0]; off += 4
        omega_t = struct.unpack('<f', buf[off:off+4])[0]; off += 4
        H = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        W = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        C = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        N = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        fps = struct.unpack('<H', buf[off:off+2])[0]; off += 2

        # SIREN weights
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

        weights = dequantize_int8(packed, meta)

        # Per-frame: residual + sha
        per_frame = []
        for _ in range(N):
            rsize = struct.unpack('<Q', buf[off:off+8])[0]; off += 8
            resid = buf[off:off+rsize]; off += rsize
            sha = buf[off:off+32]; off += 32
            per_frame.append((resid, sha))

        # Rebuild model
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = TemporalSIREN(hidden_features=hidden, hidden_layers=hidden_l,
                                omega_0=omega_0, omega_t=omega_t).to(dev)
        model.load_from_numpy(weights)
        model.eval()

        # Build coords for ALL frames
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing='ij',
        )
        xy = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
        xy_rep = xy.repeat(N, 1)
        if N > 1:
            t_vals = torch.linspace(-1, 1, N, device=dev)
        else:
            t_vals = torch.tensor([0.0], device=dev)
        t_rep = t_vals.repeat_interleave(H * W).reshape(-1, 1)
        coords = torch.cat([xy_rep, t_rep], dim=-1)

        # Inference chunked
        n_total = N * H * W
        CHUNK = 1 << 18
        preds = []
        with torch.no_grad():
            for i in range(0, n_total, CHUNK):
                preds.append(model(coords[i:i+CHUNK]).cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        pred = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(N, H, W, 3)

        # Apply residuals + verify
        frames = []
        all_match = True
        per_frame_sha = []
        for i in range(N):
            resid, sha_expected = per_frame[i]
            if codec_id == 2:
                residual_img = decode_residual_webp(resid)
            elif codec_id == 1:
                residual_img = decode_residual_png(resid)
            else:
                residual_img = decode_residual_zlib(resid, (H, W, C))
            recovered = ((pred[i].astype(np.int16) + residual_img.astype(np.int16)) % 256).astype(np.uint8)
            sha_got = hashlib.sha256(recovered.tobytes()).digest()
            match = (sha_got == sha_expected)
            if not match:
                all_match = False
            per_frame_sha.append(match)
            frames.append(recovered)

        return frames, {
            'n_frames': N,
            'fps': fps,
            'bits': bits,
            'residual_codec': codec_name,
            'all_sha256_match': all_match,
            'sha256_per_frame': per_frame_sha,
            'mode': 'video',
        }


# ============================================================
#  Self-test
# ============================================================
def _self_test():
    import zlib as _z
    print(f"[video] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"[video] Mode: Temporal SIREN f(x,y,t) -> RGB")

    # Generate a synthetic video: 8 frames of a moving gaussian blob
    N = 8
    SIZE = 64
    frames = []
    for i in range(N):
        rng = np.random.default_rng(seed=42)
        ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
        img = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
        # Moving blob: center moves linearly with frame
        t = i / (N - 1)  # 0 to 1
        cx = SIZE * (0.2 + 0.6 * t)
        cy = SIZE * (0.5 + 0.1 * np.sin(t * np.pi))
        for c in range(3):
            sigma = 8.0 + c
            amp = 200 - c * 30
            img[:, :, c] = amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        # Add subtle color shift over time
        img[:, :, 0] += 30 * t
        img[:, :, 2] += 30 * (1 - t)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))

    total_orig = sum(f.nbytes for f in frames)
    zip_total = sum(len(_z.compress(f.tobytes(), 9)) for f in frames)
    print(f"[video] {N} frames x {SIZE}x{SIZE}x3 = {total_orig:,}B")
    print(f"[video] ZIP per-frame total: {zip_total:,}B ({total_orig/zip_total:.2f}x)")

    comp = VideoCompressor(hidden_features=64, hidden_layers=3,
                            omega_0=30.0, omega_t=1.0,
                            residual_codec='webp')
    t0 = time.time()
    res = comp.compress(frames, epochs=1500, lr=1e-3,
                         bits=8, batch_size=4096, verbose=True)
    dt = time.time() - t0

    print(f"\n[video] BLKH Video: {res['recipe_size']:,}B  "
          f"(ratio {total_orig/res['recipe_size']:.2f}x)")
    print(f"  SIREN weights (shared): {res['weights_packed_size']:,}B "
          f"-> {res['weights_packed_size']/N:.0f}B/frame amortized")
    print(f"  residual per frame:     {res['residual_per_frame']:.0f}B ({res['residual_codec']})")
    print(f"  bit acc avg: {res['avg_bit_pct']:.1f}%")
    print(f"  train time:  {res['train_time_s']:.1f}s  total: {dt:.1f}s")

    # Verify
    t0 = time.time()
    recovered, dmeta = VideoCompressor.decompress(res['recipe_bytes'])
    print(f"\n[video] Decompress: {time.time()-t0:.1f}s  "
          f"all_sha256_match: {dmeta['all_sha256_match']}")
    print(f"  per-frame SHA match: {sum(dmeta['sha256_per_frame'])}/{dmeta['n_frames']}")

    winner = "BLKH" if res['recipe_size'] < zip_total else "ZIP"
    print(f"\n[video] vs ZIP: {zip_total/res['recipe_size']:.3f}x  -> {winner}")


if __name__ == '__main__':
    _self_test()
