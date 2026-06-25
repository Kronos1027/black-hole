# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

# BLKH-AUTH-DPS-2025-Kronos1027-darlan1027pc@gmail.com
"""
siren_v5_atlas.py — v5.2 Neural Atlas for datacenter-scale compression
=======================================================================
A single SIREN trained on N similar images simultaneously. Per-image, only a
small "slice index" is stored, plus the residual. The shared weights are
paid ONCE across the entire corpus — so per-file cost shrinks dramatically
as N grows.

This is the datacenter use case: if you have 1000 similar MRI slices,
1000 satellite tiles, 1000 game textures, the recipe size is dominated
by the per-image residual (~hundreds of bytes), not the weights (~13KB
shared).

Two strategies implemented:
  Strategy A — Tiled: stack images along a 3rd "image_id" coordinate.
              f(x, y, z) -> RGB, where z selects the image.
              Single SIREN, N images. Recipe: shared weights + per-image residual.

  Strategy B — Modulation: shared base weights + per-image modulation
              vectors (COIN++ style). Per-image cost is tiny (a few hundred
              floats quantized to INT8).

We implement Strategy A here because it is simpler, requires no
meta-training loop, and gives a fair baseline. Strategy B is left for v5.3.
"""
from __future__ import annotations
import os
import sys
import io
import time
import zlib
import struct
import hashlib
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siren_v5_torch import (
    SIREN, SineLayer, quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4, prune_weights,
)


MAGIC_ATLAS = b'BLA5'  # Black-hole Atlas v5
VERSION_ATLAS = 1


# ============================================================
#  Atlas SIREN — 3D coordinate (x, y, z) -> RGB
# ============================================================
class AtlasSIREN(nn.Module):
    """3D-input SIREN: (x, y, image_id) -> RGB."""
    def __init__(self, hidden_features=64, hidden_layers=3, omega_0=30.0):
        super().__init__()
        self.omega_0 = float(omega_0)
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
        return self.net(x)

    def state_to_numpy(self):
        return {name: p.detach().cpu().numpy().astype(np.float32)
                for name, p in self.named_parameters()}

    def load_from_numpy(self, state):
        with torch.no_grad():
            for name, p in self.named_parameters():
                arr = np.asarray(state[name]).astype(np.float32)
                p.copy_(torch.from_numpy(arr))


# ============================================================
#  Atlas compressor
# ============================================================
class AtlasCompressor:
    """
    Compress N similar images into a single .bla5 recipe.

    Recipe layout (.bla5):
        [4B  magic 'BLA5']
        [1B  version]
        [1B  bits]
        [2B  hidden_features]
        [1B  hidden_layers]
        [4B  omega_0_x1e6 (float)]
        [2B  H]
        [2B  W]
        [1B  C]
        [2B  N (num images)]
        [4B  weights_packed_size]
        [weights_packed bytes]
        [2B  n_meta_entries]
        [... per-tensor meta: name_len, name, n_bytes, ndim, shape, scale, n_elem]
        for each image:
            [4B  resid_compressed_size]
            [resid_compressed bytes]
            [32B sha256_original]
    """

    def __init__(self, hidden_features=64, hidden_layers=3, omega_0=30.0,
                 device: str | None = None):
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = float(omega_0)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _make_model(self):
        return AtlasSIREN(self.hidden_features, self.hidden_layers,
                          self.omega_0).to(self.device)

    def _build_coords(self, H: int, W: int, N: int) -> torch.Tensor:
        """
        Build (N * H * W, 3) coords: (x, y, z) per pixel per image.
        x, y in [-1, 1]; z = image_id mapped to [-1, 1] spaced by 2/(N-1).
        """
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij',
        )
        # shape (H, W)
        xy = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # (H*W, 2)
        # repeat for N images
        xy_rep = xy.repeat(N, 1)  # (N*H*W, 2)
        # z coords: one per image, in [-1, 1]
        if N > 1:
            z_vals = torch.linspace(-1, 1, N, device=self.device)
        else:
            z_vals = torch.tensor([0.0], device=self.device)
        z_rep = z_vals.repeat_interleave(H * W).reshape(-1, 1)  # (N*H*W, 1)
        coords = torch.cat([xy_rep, z_rep], dim=-1)  # (N*H*W, 3)
        return coords

    def compress(self, images: list[np.ndarray],
                 epochs: int = 2000, lr: float = 1e-3,
                 bits: int = 8, prune_threshold: float = 0.0,
                 batch_size: int | None = None,
                 verbose: bool = False) -> dict:
        """
        Compress N images into a single .bla5 recipe (bit-perfect).
        images: list of (H, W, C) uint8 arrays. Must all have same shape.
        """
        assert all(img.shape == images[0].shape for img in images), \
            "All images must have the same shape"
        assert images[0].dtype == np.uint8
        H, W, C = images[0].shape
        assert C == 3, "Atlas currently supports RGB only"
        N = len(images)
        t0 = time.time()

        # Pack all images into one big target tensor (N*H*W, 3) in [-1, 1]
        stacked = np.stack(images, axis=0)  # (N, H, W, 3)
        target_np = (stacked.astype(np.float32) / 127.5 - 1.0).reshape(-1, 3)
        target = torch.from_numpy(target_np).to(self.device)

        coords = self._build_coords(H, W, N)  # (N*H*W, 3)
        n_total = coords.shape[0]
        if batch_size is None or batch_size >= n_total:
            batch_size = n_total

        model = self._make_model()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        warmup = max(1, epochs // 20)

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
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            if epoch >= warmup:
                sched.step()
            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  atlas epoch {epoch}/{epochs}  loss={loss.item():.6e}")
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(float(loss.item()))

        train_time = time.time() - t0

        # Quantize weights
        W_np = model.state_to_numpy()
        if prune_threshold > 0:
            W_np = prune_weights(W_np, prune_threshold)
        if bits == 8:
            packed, packed_meta = quantize_int8(W_np)
        else:
            packed, packed_meta = quantize_int4(W_np)
        # Reload quantized
        if bits == 8:
            Wq = dequantize_int8(packed, packed_meta)
        else:
            Wq = dequantize_int4(packed, packed_meta)
        model.load_from_numpy(Wq)
        model.eval()

        # Inference for ALL images at once (chunked if needed)
        CHUNK = 1 << 18  # 256K pixels per chunk
        predicted_list = []
        with torch.no_grad():
            for i in range(0, n_total, CHUNK):
                pred_chunk = model(coords[i:i + CHUNK]).cpu().numpy()
                predicted_list.append(pred_chunk)
        predicted = np.concatenate(predicted_list, axis=0)  # (N*H*W, 3)
        predicted_img = np.clip((predicted + 1.0) * 127.5, 0, 255).astype(np.uint8)
        predicted_img = predicted_img.reshape(N, H, W, 3)

        # Per-image residual
        residuals = []
        shas = []
        bit_pcts = []
        for i in range(N):
            orig_bytes = images[i].tobytes()
            pred_bytes = predicted_img[i].tobytes()
            a = np.frombuffer(orig_bytes, dtype=np.uint8)
            b = np.frombuffer(pred_bytes, dtype=np.uint8)
            residual = (a ^ b).tobytes()
            residuals.append(zlib.compress(residual, 9))
            shas.append(hashlib.sha256(orig_bytes).digest())
            bit_pcts.append(float(np.mean(np.unpackbits(a) == np.unpackbits(b))) * 100)

        # Pack recipe
        recipe = self._pack_recipe(bits, packed, packed_meta,
                                    H, W, C, N, residuals, shas)

        # Stats
        total_residual = sum(len(r) for r in residuals)
        total_orig = sum(img.nbytes for img in images)
        return {
            'recipe_bytes': recipe,
            'recipe_size': len(recipe),
            'weights_packed_size': len(packed),
            'residual_total': total_residual,
            'residual_per_image': total_residual / N,
            'avg_bit_pct': float(np.mean(bit_pcts)),
            'min_bit_pct': float(np.min(bit_pcts)),
            'max_bit_pct': float(np.max(bit_pcts)),
            'train_time_s': train_time,
            'n_images': N,
            'total_orig': total_orig,
            'final_loss': history[-1] if history else float('nan'),
        }

    def _pack_recipe(self, bits: int, packed: bytes, packed_meta: list,
                     H: int, W: int, C: int, N: int,
                     residuals: list[bytes], shas: list[bytes]) -> bytes:
        out = bytearray()
        out += MAGIC_ATLAS
        out += struct.pack('<B', VERSION_ATLAS)
        out += struct.pack('<B', bits)
        out += struct.pack('<H', self.hidden_features)
        out += struct.pack('<B', self.hidden_layers)
        out += struct.pack('<f', self.omega_0)
        out += struct.pack('<H', H)
        out += struct.pack('<H', W)
        out += struct.pack('<B', C)
        out += struct.pack('<H', N)
        # weights
        out += struct.pack('<I', len(packed))
        out += packed
        # meta
        out += struct.pack('<H', len(packed_meta))
        for entry in packed_meta:
            if bits == 8:
                n_bytes, shape, scale, name = entry
                n_elem = 0
            else:
                n_bytes, shape, scale, n_elem, name = entry
            name_b = name.encode('utf-8')
            out += struct.pack('<B', len(name_b))
            out += name_b
            out += struct.pack('<I', n_bytes)
            out += struct.pack('<B', len(shape))
            for d in shape:
                out += struct.pack('<i', int(d))
            out += struct.pack('<d', float(scale))
            out += struct.pack('<I', n_elem)
        # per-image residuals
        for resid, sha in zip(residuals, shas):
            out += struct.pack('<I', len(resid))
            out += resid
            out += sha
        return bytes(out)

    @staticmethod
    def decompress(recipe_bytes: bytes, device: str | None = None) -> tuple[list[np.ndarray], dict]:
        buf = recipe_bytes
        off = 0
        if buf[:4] != MAGIC_ATLAS:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_ATLAS
        bits = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        hidden = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden_l = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        omega = struct.unpack('<f', buf[off:off+4])[0]; off += 4
        H = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        W = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        C = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        N = struct.unpack('<H', buf[off:off+2])[0]; off += 2

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

        per_image = []
        for _ in range(N):
            rsize = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            resid = buf[off:off+rsize]; off += rsize
            sha = buf[off:off+32]; off += 32
            per_image.append((resid, sha))

        # Dequantize
        if bits == 8:
            weights_dict = dequantize_int8(packed, meta)
        else:
            weights_dict = dequantize_int4(packed, meta)

        # Build model
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = AtlasSIREN(hidden, hidden_l, omega).to(dev)
        model.load_from_numpy(weights_dict)
        model.eval()

        # Build coords for all images
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing='ij',
        )
        xy = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # (H*W, 2)
        xy_rep = xy.repeat(N, 1)
        if N > 1:
            z_vals = torch.linspace(-1, 1, N, device=dev)
        else:
            z_vals = torch.tensor([0.0], device=dev)
        z_rep = z_vals.repeat_interleave(H * W).reshape(-1, 1)
        coords = torch.cat([xy_rep, z_rep], dim=-1)

        # Inference chunked
        n_total = N * H * W
        CHUNK = 1 << 18
        preds = []
        with torch.no_grad():
            for i in range(0, n_total, CHUNK):
                preds.append(model(coords[i:i+CHUNK]).cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        pred_img = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8)
        pred_img = pred_img.reshape(N, H, W, 3)

        # Apply residuals + verify
        recovered = []
        all_match = True
        shas_match = []
        for i in range(N):
            pred_bytes = pred_img[i].tobytes()
            resid, sha_expected = per_image[i]
            residual = zlib.decompress(resid)
            a = np.frombuffer(pred_bytes, dtype=np.uint8)
            b = np.frombuffer(residual, dtype=np.uint8)
            rec = (a ^ b).tobytes()
            sha_got = hashlib.sha256(rec).digest()
            match = (sha_got == sha_expected)
            if not match:
                all_match = False
            shas_match.append(match)
            recovered.append(np.frombuffer(rec, dtype=np.uint8).reshape(H, W, 3))

        return recovered, {
            'n_images': N,
            'all_sha256_match': all_match,
            'sha256_per_image': shas_match,
            'bits': bits,
            'hidden_features': hidden,
            'hidden_layers': hidden_l,
            'omega_0': omega,
        }


# ============================================================
#  Self-test
# ============================================================
def _self_test():
    print(f"[atlas] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")

    # 10 similar smooth images
    N = 10
    SIZE = 64
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

    total_orig = sum(im.nbytes for im in images)
    zip_total = sum(len(zlib.compress(im.tobytes(), 9)) for im in images)
    print(f"[atlas] {N} images x {SIZE}x{SIZE}x3 = {total_orig:,}B")
    print(f"[atlas] ZIP per-file total: {zip_total:,}B  (ratio {total_orig/zip_total:.2f}x)")

    comp = AtlasCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
    t0 = time.time()
    res = comp.compress(images, epochs=1500, lr=1e-3, bits=8,
                         batch_size=8192, verbose=True)
    dt = time.time() - t0
    print(f"\n[atlas] BLKH Atlas: {res['recipe_size']:,}B  "
          f"(ratio {total_orig/res['recipe_size']:.2f}x)")
    print(f"  weights (shared): {res['weights_packed_size']:,}B  "
          f"-> {res['weights_packed_size']/N:.0f}B/image amortized")
    print(f"  residual total:   {res['residual_total']:,}B  "
          f"-> {res['residual_per_image']:.0f}B/image avg")
    print(f"  bit acc avg: {res['avg_bit_pct']:.1f}%")
    print(f"  train time: {res['train_time_s']:.1f}s  total {dt:.1f}s")

    # Verify
    t0 = time.time()
    rec, meta = AtlasCompressor.decompress(res['recipe_bytes'])
    print(f"\n[atlas] Decompress: {time.time()-t0:.1f}s  "
          f"all_sha256_match: {meta['all_sha256_match']}")
    print(f"  per-image SHA match: {sum(meta['sha256_per_image'])}/{meta['n_images']}")

    winner = "BLKH" if res['recipe_size'] < zip_total else "ZIP"
    print(f"\n[atlas] vs ZIP: {zip_total/res['recipe_size']:.3f}x  -> {winner}")


if __name__ == '__main__':
    _self_test()
