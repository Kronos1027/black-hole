# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

# BLKH-AUTH-DPS-2025-Kronos1027-darlan1027pc@gmail.com
"""
siren_v5_meta.py — v5.3 Meta-Learning with Per-Image Modulations (COIN++ style)
================================================================================
Solves the v5.2 atlas limitation: a single shared SIREN degrades as N grows
beyond ~10 because it can't represent all the image diversity with one set
of weights. Meta-learning fixes this by:

    1. Training a SHARED BASE network once on a corpus of similar images.
    2. For each new image, train ONLY a small "modulation" vector (a few
       hundred floats per image).
    3. Per-image cost drops to ~hundreds of bytes (modulation quantized to
       INT8) + a small residual.

The base weights are paid ONCE for the entire corpus. Per-image recipe is:
    [quantized modulation (INT8)] + [XOR residual (zlib)]

Reference:
    Dupont et al., 2021 — COIN++: Data Agnostic Neural Compression
    (we implement a simplified version: multiplicative modulation on
     pre-activations, no hypernetwork)

API:
    MetaCompressor(hidden_features=64, hidden_layers=3)
        .train_base(images, epochs=2000)        # train shared base
        .compress(image, epochs=500)             # compress ONE new image
        .compress_many(images, epochs=500)       # compress N images (parallel recipe)
        .decompress(recipe) -> image

Recipe format .blkm (Black-hole Meta):
    [magic 'BLKM'][version][base_meta][N x [modulation + residual + sha]]
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
    SineLayer, quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4,
)


MAGIC_META = b'BLKM'
VERSION_META = 1


# ============================================================
#  Meta-SIREN: shared base weights + per-image modulation
# ============================================================
class MetaSIRENLayer(nn.Module):
    """SIREN layer with FiLM-style modulation: y = gamma * z + beta.

    gamma and beta are per-image learned parameters (one scalar per output unit).
    gamma is initialized to 1.0 (identity), beta to 0.0 (no shift).
    This gives the modulation 2x more capacity than purely multiplicative.
    """

    def __init__(self, in_features: int, out_features: int,
                 is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_features, out_features)
        # FiLM: gamma (scale) and beta (shift)
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / max(self.in_features, 1)
            else:
                bound = float(np.sqrt(6.0 / max(self.in_features, 1))) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        # FiLM modulation
        z = self.gamma * z + self.beta
        if self.is_first:
            return torch.sin(z)
        return torch.sin(self.omega_0 * z)


class MetaSIREN(nn.Module):
    """SIREN with per-layer multiplicative modulation (COIN++ style)."""

    def __init__(self, in_features=2, hidden_features=64, hidden_layers=3,
                 out_features=3, omega_0=30.0):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features

        layers = [MetaSIRENLayer(in_features, hidden_features,
                                  is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers):
            layers.append(MetaSIRENLayer(hidden_features, hidden_features,
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

    # ---- base weights (shared across all images) ----
    def base_parameters(self):
        """Iterator over shared base params (linear weights + biases, NOT gamma/beta)."""
        for name, p in self.named_parameters():
            if 'gamma' not in name and 'beta' not in name:
                yield name, p

    def modulation_parameters(self):
        """Iterator over per-image FiLM params (gamma + beta)."""
        for name, p in self.named_parameters():
            if 'gamma' in name or 'beta' in name:
                yield name, p

    def freeze_base(self, freeze: bool = True):
        for name, p in self.named_parameters():
            if 'gamma' not in name and 'beta' not in name:
                p.requires_grad = not freeze
        # gamma/beta are always trainable per-image
        for name, p in self.named_parameters():
            if 'gamma' in name or 'beta' in name:
                p.requires_grad = True

    def reset_modulations(self):
        """Reset FiLM params to identity (gamma=1, beta=0) — call before each new image."""
        with torch.no_grad():
            for name, p in self.named_parameters():
                if 'gamma' in name:
                    p.fill_(1.0)
                elif 'beta' in name:
                    p.fill_(0.0)

    def base_state_to_numpy(self) -> dict:
        return {name: p.detach().cpu().numpy().astype(np.float32)
                for name, p in self.base_parameters()}

    def modulation_state_to_numpy(self) -> dict:
        return {name: p.detach().cpu().numpy().astype(np.float32)
                for name, p in self.modulation_parameters()}

    def load_base_from_numpy(self, state: dict):
        with torch.no_grad():
            for name, p in self.base_parameters():
                if name in state:
                    arr = np.asarray(state[name]).astype(np.float32)
                    p.copy_(torch.from_numpy(arr))

    def load_modulation_from_numpy(self, state: dict):
        with torch.no_grad():
            for name, p in self.modulation_parameters():
                if name in state:
                    arr = np.asarray(state[name]).astype(np.float32)
                    p.copy_(torch.from_numpy(arr))


# ============================================================
#  Meta compressor
# ============================================================
class MetaCompressor:
    """
    Train a shared base SIREN on a corpus, then compress new images by
    training ONLY their modulation vectors. Per-image cost = quantized
    modulation + XOR residual.

    Use case: datacenter with 100+ similar images (MRI slices, satellite
    tiles, game textures). Train base once, compress new images in
    ~0.5s each with a few hundred bytes per image.
    """

    def __init__(self, hidden_features=64, hidden_layers=3, omega_0=30.0,
                 device: str | None = None):
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = float(omega_0)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: MetaSIREN | None = None
        self._base_trained = False
        # cached base state (numpy) so we can rebuild fresh models for new images
        self._cached_base: dict | None = None

    def _make_model(self) -> MetaSIREN:
        return MetaSIREN(in_features=2,
                          hidden_features=self.hidden_features,
                          hidden_layers=self.hidden_layers,
                          out_features=3,
                          omega_0=self.omega_0).to(self.device)

    def _make_coords(self, H: int, W: int) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij',
        )
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    # ============================================================
    #  Phase 1: Train shared base on a corpus
    # ============================================================
    def train_base(self, images: list[np.ndarray],
                   epochs: int = 2000, lr: float = 1e-3,
                   batch_size: int = 2048,
                   verbose: bool = False) -> dict:
        """Train the shared base SIREN on a corpus of similar images.

        Strategy: each epoch, sample one random image. Train ONLY the base
        weights (modulations are frozen at 1.0 during base training). The
        base learns to be a generic "smooth image prior". Modulations are
        trained per-image in compress() / compress_many().
        """
        assert all(im.shape == images[0].shape for im in images)
        H, W, C = images[0].shape
        assert C == 3
        N = len(images)

        # Pre-encode all images to tensors
        target_tensors = []
        for im in images:
            t = torch.from_numpy(
                (im.astype(np.float32) / 127.5 - 1.0).reshape(-1, 3)
            ).to(self.device)
            target_tensors.append(t)

        coords = self._make_coords(H, W)  # (H*W, 2)

        model = self._make_model()
        # Freeze FiLM modulations during base training — they stay at identity
        model.freeze_base(False)  # unfreeze base
        for name, p in model.named_parameters():
            if 'gamma' in name or 'beta' in name:
                p.requires_grad = False
        # Optimizer only over base
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        warmup = max(1, epochs // 20)

        model.train()
        history = []
        t0 = time.time()
        for epoch in range(epochs):
            if epoch < warmup:
                for g in opt.param_groups:
                    g['lr'] = lr * (epoch + 1) / warmup

            # Cycle through images so base sees them all
            idx = epoch % N
            target = target_tensors[idx]

            n = coords.shape[0]
            if batch_size < n:
                bidx = torch.randint(0, n, (batch_size,), device=self.device)
                xb = coords[bidx]
                yb = target[bidx]
            else:
                xb, yb = coords, target

            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch >= warmup:
                sched.step()

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  base epoch {epoch}/{epochs}  img={idx}  loss={loss.item():.6e}")
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(float(loss.item()))

        train_time = time.time() - t0
        self._model = model
        self._base_trained = True
        self._cached_base = model.base_state_to_numpy()
        return {
            'train_time_s': train_time,
            'n_images': N,
            'final_loss': history[-1] if history else float('nan'),
            'history': history,
        }

    # ============================================================
    #  Phase 2: Compress ONE new image (train only its modulation)
    # ============================================================
    def compress(self, image: np.ndarray,
                 epochs: int = 500, lr: float = 1e-3,
                 bits: int = 8,
                 batch_size: int | None = None,
                 zlib_level: int = 9,
                 verbose: bool = False) -> dict:
        """Compress one image using the trained base. Returns recipe bytes."""
        if not self._base_trained or self._cached_base is None:
            raise RuntimeError("Must call train_base() first")
        assert image.dtype == np.uint8 and image.ndim == 3
        H, W, C = image.shape
        original_bytes = image.tobytes()

        # Build a fresh model with cached base, freeze base
        model = self._make_model()
        model.load_base_from_numpy(self._cached_base)
        model.reset_modulations()
        model.freeze_base(True)  # only modulations are trainable
        model.train()

        coords = self._make_coords(H, W)
        target = torch.from_numpy(
            (image.astype(np.float32) / 127.5 - 1.0).reshape(-1, 3)
        ).to(self.device)
        n = coords.shape[0]
        if batch_size is None or batch_size >= n:
            batch_size = n

        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        warmup = max(1, epochs // 20)
        history = []
        t0 = time.time()
        for epoch in range(epochs):
            if epoch < warmup:
                for g in opt.param_groups:
                    g['lr'] = lr * (epoch + 1) / warmup
            if batch_size < n:
                bidx = torch.randint(0, n, (batch_size,), device=self.device)
                xb = coords[bidx]; yb = target[bidx]
            else:
                xb, yb = coords, target
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            if epoch >= warmup:
                sched.step()
            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  mod epoch {epoch}/{epochs}  loss={loss.item():.6e}")
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(float(loss.item()))

        train_time = time.time() - t0

        # Quantize modulation (small vector — INT8 is fine)
        mod_state = model.modulation_state_to_numpy()
        packed_mod, mod_meta = quantize_int8(mod_state)
        # Reload quantized modulation into model so residual matches decompress
        mod_q = dequantize_int8(packed_mod, mod_meta)
        model.load_modulation_from_numpy(mod_q)
        model.eval()

        # Inference
        with torch.no_grad():
            pred = model(coords).cpu().numpy()
        predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8)
        predicted_bytes = predicted.tobytes()

        # Residual XOR
        a = np.frombuffer(original_bytes, dtype=np.uint8)
        b = np.frombuffer(predicted_bytes, dtype=np.uint8)
        residual = (a ^ b).tobytes()
        residual_compressed = zlib.compress(residual, zlib_level)

        sha = hashlib.sha256(original_bytes).digest()
        bit_acc = float(np.mean(np.unpackbits(a) == np.unpackbits(b))) * 100

        # Pack single-image recipe
        recipe = self._pack_recipe_single(bits, packed_mod, mod_meta,
                                            H, W, C, residual_compressed, sha)
        return {
            'recipe_bytes': recipe,
            'recipe_size': len(recipe),
            'modulation_size': len(packed_mod),
            'residual_compressed_size': len(residual_compressed),
            'residual_raw_size': len(residual),
            'model_bit_accuracy': bit_acc,
            'train_time_s': train_time,
            'sha256': sha.hex(),
            'final_loss': history[-1] if history else float('nan'),
        }

    # ============================================================
    #  Phase 2b: Compress MANY images (returns aggregate recipe with shared base)
    # ============================================================
    def compress_many(self, images: list[np.ndarray],
                      epochs: int = 500, lr: float = 1e-3,
                      bits: int = 8,
                      batch_size: int | None = None,
                      zlib_level: int = 9,
                      verbose: bool = False) -> dict:
        """Compress N images using the trained base. Returns ONE recipe
        containing: shared quantized base + per-image (modulation + residual + sha).
        This is the datacenter-scale path: base paid once, per-image cost is tiny.
        """
        if not self._base_trained or self._cached_base is None:
            raise RuntimeError("Must call train_base() first")
        assert all(im.shape == images[0].shape for im in images)
        H, W, C = images[0].shape
        N = len(images)

        # Quantize the base ONCE for the whole corpus
        base_state = self._cached_base
        packed_base, base_meta = quantize_int8(base_state)

        per_image_data = []
        total_residual = 0
        bit_pcts = []
        t0 = time.time()
        for i, img in enumerate(images):
            if verbose:
                print(f"  compressing image {i+1}/{N}...")
            # Build fresh model with QUANTIZED base, train only modulation
            model = self._make_model()
            base_q = dequantize_int8(packed_base, base_meta)
            model.load_base_from_numpy(base_q)
            model.reset_modulations()
            model.freeze_base(True)
            model.train()

            coords = self._make_coords(H, W)
            target = torch.from_numpy(
                (img.astype(np.float32) / 127.5 - 1.0).reshape(-1, 3)
            ).to(self.device)
            n = coords.shape[0]
            if batch_size is None or batch_size >= n:
                batch_size_eff = n
            else:
                batch_size_eff = batch_size

            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            warmup = max(1, epochs // 20)
            for epoch in range(epochs):
                if epoch < warmup:
                    for g in opt.param_groups:
                        g['lr'] = lr * (epoch + 1) / warmup
                if batch_size_eff < n:
                    bidx = torch.randint(0, n, (batch_size_eff,), device=self.device)
                    xb = coords[bidx]; yb = target[bidx]
                else:
                    xb, yb = coords, target
                pred = model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                if epoch >= warmup:
                    sched.step()

            # Quantize modulation
            mod_state = model.modulation_state_to_numpy()
            packed_mod, mod_meta = quantize_int8(mod_state)
            mod_q = dequantize_int8(packed_mod, mod_meta)
            model.load_modulation_from_numpy(mod_q)
            model.eval()

            # Inference
            with torch.no_grad():
                pred = model(coords).cpu().numpy()
            predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8)
            predicted_bytes = predicted.tobytes()

            original_bytes = img.tobytes()
            a = np.frombuffer(original_bytes, dtype=np.uint8)
            b = np.frombuffer(predicted_bytes, dtype=np.uint8)
            residual = (a ^ b).tobytes()
            residual_compressed = zlib.compress(residual, zlib_level)
            sha = hashlib.sha256(original_bytes).digest()
            bit_acc = float(np.mean(np.unpackbits(a) == np.unpackbits(b))) * 100

            per_image_data.append({
                'packed_mod': packed_mod,
                'mod_meta': mod_meta,
                'residual_compressed': residual_compressed,
                'sha': sha,
            })
            total_residual += len(residual_compressed)
            bit_pcts.append(bit_acc)

        total_time = time.time() - t0

        # Pack aggregate recipe
        recipe = self._pack_recipe_aggregate(bits, packed_base, base_meta,
                                                H, W, C, per_image_data)
        return {
            'recipe_bytes': recipe,
            'recipe_size': len(recipe),
            'base_size': len(packed_base),
            'modulation_per_image': len(per_image_data[0]['packed_mod']),
            'residual_total': total_residual,
            'residual_per_image': total_residual / N,
            'avg_bit_pct': float(np.mean(bit_pcts)),
            'min_bit_pct': float(np.min(bit_pcts)),
            'max_bit_pct': float(np.max(bit_pcts)),
            'n_images': N,
            'total_time_s': total_time,
            'per_image_time_s': total_time / N,
        }

    # ============================================================
    #  Recipe packing (single-image .blkm)
    # ============================================================
    def _pack_recipe_single(self, bits: int, packed_mod: bytes, mod_meta: list,
                             H: int, W: int, C: int,
                             residual_compressed: bytes, sha: bytes) -> bytes:
        out = bytearray()
        out += MAGIC_META
        out += struct.pack('<B', VERSION_META)
        out += struct.pack('<B', 0)  # mode 0 = single-image (base embedded)
        # For single-image mode, we embed the base in the recipe
        # so decompress is self-contained.
        if self._cached_base is None:
            raise RuntimeError("base not trained")
        packed_base, base_meta = quantize_int8(self._cached_base)
        out += struct.pack('<B', bits)
        out += struct.pack('<H', self.hidden_features)
        out += struct.pack('<B', self.hidden_layers)
        out += struct.pack('<f', self.omega_0)
        out += struct.pack('<H', H)
        out += struct.pack('<H', W)
        out += struct.pack('<B', C)
        # base weights
        out += struct.pack('<I', len(packed_base))
        out += packed_base
        out += struct.pack('<H', len(base_meta))
        for entry in base_meta:
            n_bytes, shape, scale, name = entry
            name_b = name.encode('utf-8')
            out += struct.pack('<B', len(name_b)); out += name_b
            out += struct.pack('<I', n_bytes)
            out += struct.pack('<B', len(shape))
            for d in shape: out += struct.pack('<i', int(d))
            out += struct.pack('<d', float(scale))
            out += struct.pack('<I', 0)  # n_elem placeholder for int8
        # modulation
        out += struct.pack('<I', len(packed_mod))
        out += packed_mod
        out += struct.pack('<H', len(mod_meta))
        for entry in mod_meta:
            n_bytes, shape, scale, name = entry
            name_b = name.encode('utf-8')
            out += struct.pack('<B', len(name_b)); out += name_b
            out += struct.pack('<I', n_bytes)
            out += struct.pack('<B', len(shape))
            for d in shape: out += struct.pack('<i', int(d))
            out += struct.pack('<d', float(scale))
            out += struct.pack('<I', 0)
        # residual + sha
        out += struct.pack('<Q', len(residual_compressed))
        out += residual_compressed
        out += sha
        return bytes(out)

    def _pack_recipe_aggregate(self, bits: int, packed_base: bytes, base_meta: list,
                                H: int, W: int, C: int,
                                per_image: list[dict]) -> bytes:
        out = bytearray()
        out += MAGIC_META
        out += struct.pack('<B', VERSION_META)
        out += struct.pack('<B', 1)  # mode 1 = aggregate (N images, shared base)
        out += struct.pack('<B', bits)
        out += struct.pack('<H', self.hidden_features)
        out += struct.pack('<B', self.hidden_layers)
        out += struct.pack('<f', self.omega_0)
        out += struct.pack('<H', H)
        out += struct.pack('<H', W)
        out += struct.pack('<B', C)
        out += struct.pack('<H', len(per_image))
        # base weights (shared, paid once)
        out += struct.pack('<I', len(packed_base))
        out += packed_base
        out += struct.pack('<H', len(base_meta))
        for entry in base_meta:
            n_bytes, shape, scale, name = entry
            name_b = name.encode('utf-8')
            out += struct.pack('<B', len(name_b)); out += name_b
            out += struct.pack('<I', n_bytes)
            out += struct.pack('<B', len(shape))
            for d in shape: out += struct.pack('<i', int(d))
            out += struct.pack('<d', float(scale))
            out += struct.pack('<I', 0)
        # per-image data
        for d in per_image:
            # modulation
            out += struct.pack('<I', len(d['packed_mod']))
            out += d['packed_mod']
            out += struct.pack('<H', len(d['mod_meta']))
            for entry in d['mod_meta']:
                n_bytes, shape, scale, name = entry
                name_b = name.encode('utf-8')
                out += struct.pack('<B', len(name_b)); out += name_b
                out += struct.pack('<I', n_bytes)
                out += struct.pack('<B', len(shape))
                for dd in shape: out += struct.pack('<i', int(dd))
                out += struct.pack('<d', float(scale))
                out += struct.pack('<I', 0)
            # residual + sha
            out += struct.pack('<Q', len(d['residual_compressed']))
            out += d['residual_compressed']
            out += d['sha']
        return bytes(out)

    # ============================================================
    #  Decompress
    # ============================================================
    @staticmethod
    def decompress(recipe_bytes: bytes, device: str | None = None) -> tuple[list[np.ndarray], dict]:
        """Decompress recipe. Returns (list_of_images, meta)."""
        buf = recipe_bytes
        off = 0
        if buf[:4] != MAGIC_META:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_META
        mode = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        bits = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        hidden = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden_l = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        omega = struct.unpack('<f', buf[off:off+4])[0]; off += 4
        H = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        W = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        C = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        if mode == 1:
            N = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        else:
            N = 1

        # Read base weights
        base_size = struct.unpack('<I', buf[off:off+4])[0]; off += 4
        packed_base = buf[off:off+base_size]; off += base_size
        n_base_meta = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        base_meta = []
        for _ in range(n_base_meta):
            name_len = struct.unpack('<B', buf[off:off+1])[0]; off += 1
            name = buf[off:off+name_len].decode('utf-8'); off += name_len
            n_bytes = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            ndim = struct.unpack('<B', buf[off:off+1])[0]; off += 1
            shape = tuple(struct.unpack('<' + 'i'*ndim, buf[off:off+4*ndim])); off += 4*ndim
            scale = struct.unpack('<d', buf[off:off+8])[0]; off += 8
            _ = struct.unpack('<I', buf[off:off+4])[0]; off += 4  # n_elem unused for int8
            base_meta.append((n_bytes, shape, scale, name))

        base_dict = dequantize_int8(packed_base, base_meta)

        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Build fresh model with quantized base
        model = MetaSIREN(in_features=2, hidden_features=hidden,
                           hidden_layers=hidden_l, out_features=3,
                           omega_0=omega).to(dev)
        model.load_base_from_numpy(base_dict)
        model.eval()

        # Build coords
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing='ij',
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

        # Read per-image data
        images = []
        all_match = True
        per_image_sha = []
        for i in range(N):
            mod_size = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            packed_mod = buf[off:off+mod_size]; off += mod_size
            n_mod_meta = struct.unpack('<H', buf[off:off+2])[0]; off += 2
            mod_meta = []
            for _ in range(n_mod_meta):
                name_len = struct.unpack('<B', buf[off:off+1])[0]; off += 1
                name = buf[off:off+name_len].decode('utf-8'); off += name_len
                n_bytes = struct.unpack('<I', buf[off:off+4])[0]; off += 4
                ndim = struct.unpack('<B', buf[off:off+1])[0]; off += 1
                shape = tuple(struct.unpack('<' + 'i'*ndim, buf[off:off+4*ndim])); off += 4*ndim
                scale = struct.unpack('<d', buf[off:off+8])[0]; off += 8
                _ = struct.unpack('<I', buf[off:off+4])[0]; off += 4
                mod_meta.append((n_bytes, shape, scale, name))

            resid_size = struct.unpack('<Q', buf[off:off+8])[0]; off += 8
            residual_compressed = buf[off:off+resid_size]; off += resid_size
            sha_expected = buf[off:off+32]; off += 32

            # Dequantize modulation and load
            mod_dict = dequantize_int8(packed_mod, mod_meta)
            model.load_modulation_from_numpy(mod_dict)

            # Inference
            with torch.no_grad():
                pred = model(coords).cpu().numpy()
            predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8)
            predicted_bytes = predicted.tobytes()

            # Apply residual
            residual = zlib.decompress(residual_compressed)
            a = np.frombuffer(predicted_bytes, dtype=np.uint8)
            b = np.frombuffer(residual, dtype=np.uint8)
            rec_bytes = (a ^ b).tobytes()
            sha_got = hashlib.sha256(rec_bytes).digest()
            match = (sha_got == sha_expected)
            if not match:
                all_match = False
            per_image_sha.append(match)
            images.append(np.frombuffer(rec_bytes, dtype=np.uint8).reshape(H, W, C))

        return images, {
            'n_images': N,
            'mode': 'aggregate' if mode == 1 else 'single',
            'bits': bits,
            'all_sha256_match': all_match,
            'sha256_per_image': per_image_sha,
            'hidden_features': hidden,
            'hidden_layers': hidden_l,
            'omega_0': omega,
        }


# ============================================================
#  Self-test
# ============================================================
def _self_test():
    import zlib as _z
    print(f"[meta] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Generate 10 similar smooth images
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
    zip_total = sum(len(_z.compress(im.tobytes(), 9)) for im in images)
    print(f"[meta] {N} images x {SIZE}x{SIZE}x3 = {total_orig:,}B")
    print(f"[meta] ZIP per-file total: {zip_total:,}B")

    # Phase 1: train base
    comp = MetaCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
    print(f"\n[meta] Phase 1: training shared base on {N} images...")
    t0 = time.time()
    base_meta = comp.train_base(images, epochs=1500, lr=1e-3,
                                 batch_size=2048, verbose=True)
    print(f"  base trained in {base_meta['train_time_s']:.1f}s, "
          f"final loss {base_meta['final_loss']:.6e}")

    # Phase 2: compress all N images using shared base
    print(f"\n[meta] Phase 2: compressing {N} images (train only modulations)...")
    res = comp.compress_many(images, epochs=1000, lr=2e-3,
                              bits=8, batch_size=2048, verbose=False)
    print(f"\n[meta] BLKH Meta: {res['recipe_size']:,}B  "
          f"(ratio {total_orig/res['recipe_size']:.2f}x)")
    print(f"  base (shared):       {res['base_size']:,}B  "
          f"-> {res['base_size']/N:.0f}B/image amortized")
    print(f"  modulation per img:  {res['modulation_per_image']:,}B")
    print(f"  residual per img:    {res['residual_per_image']:.0f}B")
    print(f"  bit acc avg: {res['avg_bit_pct']:.1f}% "
          f"(min {res['min_bit_pct']:.1f}, max {res['max_bit_pct']:.1f})")
    print(f"  time: {res['total_time_s']:.1f}s "
          f"({res['per_image_time_s']:.2f}s/image)")

    # Verify
    t0 = time.time()
    recovered, dmeta = MetaCompressor.decompress(res['recipe_bytes'])
    print(f"\n[meta] Decompress: {time.time()-t0:.1f}s  "
          f"all_sha256_match: {dmeta['all_sha256_match']}")
    print(f"  per-image SHA match: {sum(dmeta['sha256_per_image'])}/{dmeta['n_images']}")

    winner = "BLKH" if res['recipe_size'] < zip_total else "ZIP"
    print(f"\n[meta] vs ZIP: {zip_total/res['recipe_size']:.3f}x  -> {winner}")


if __name__ == '__main__':
    _self_test()
