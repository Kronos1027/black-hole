# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

"""
siren_v5_distill.py — v5.31 Knowledge Distillation Compression
================================================================
PRODUCTION implementation of BHUH research Phase 85+89:
distillation + INT4 quantization for extreme compression.

Strategy:
  1. Train teacher SIREN (large, float32) on target image
  2. Distill to student SIREN (small, INT4 QAT) matching teacher output
  3. Serialize student as compact .blke recipe

This mode is OPTIMIZED FOR EXTREME COMPRESSION RATIO at the cost of
quality. Best for: smooth signals, satellite imagery, medical slices,
game textures. For natural photos, use siren_v5_dct or siren_v5_photo.

Recipe format (.blke):
  [magic 'BLKE'][version][student_hidden][n_layers]
  [H][W][scale_float][INT4-packed params]

Author: Darlan Pereira da Silva (Kronos1027)
"""
from __future__ import annotations
import os
import sys
import io
import time
import struct
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAGIC_DISTILL = b'BLKE'
VERSION_DISTILL = 1

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _compute_psnr(orig, recon):
    orig = orig.astype(np.float64)
    recon = recon.astype(np.float64)
    mse = float(np.mean((orig - recon) ** 2))
    if mse == 0:
        return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def _siren_param_count(hidden, n_layers=3, in_dim=2, out_dim=1):
    d = in_dim
    total = 0
    for k in range(n_layers - 1):
        total += d * hidden + hidden
        d = hidden
    total += d * out_dim + out_dim
    return total


def _quantize_ste(x, bits=4):
    """Straight-through estimator for quantization-aware training."""
    if bits >= 32:
        return x
    levels = 2 ** bits - 1
    max_val = x.abs().max().detach()
    scale = max_val / (levels / 2)
    x_scaled = x / (scale + 1e-9)
    x_rounded = torch.round(x_scaled)
    x_clipped = torch.clamp(x_rounded, -levels / 2, levels / 2)
    x_dequant = x_clipped * scale
    return x + (x_dequant - x).detach()


class _DistillSIREN(nn.Module):
    """SIREN with optional quantization-aware forward pass."""

    def __init__(self, hidden, n_layers=3, omega=15.0, quant_bits=32):
        super().__init__()
        self.layers = nn.ModuleList()
        d = 2
        for k in range(n_layers - 1):
            lin = nn.Linear(d, hidden)
            bound = 1.0 / d if k == 0 else np.sqrt(6.0 / hidden) / omega
            lin.weight.data.uniform_(-bound, bound)
            lin.bias.data.uniform_(-bound, bound)
            self.layers.append(lin)
            d = hidden
        self.head = nn.Linear(hidden, 1)
        bound = np.sqrt(6.0 / hidden) / omega
        self.head.weight.data.uniform_(-bound, bound)
        self.head.bias.data.uniform_(-bound, bound)
        self.omega = omega
        self.quant_bits = quant_bits

    def _quantize(self, w):
        if self.quant_bits >= 32:
            return w
        return _quantize_ste(w, self.quant_bits)

    def forward(self, x):
        h = x
        for layer in self.layers:
            w, b = self._quantize(layer.weight), self._quantize(layer.bias)
            h = torch.sin(self.omega * torch.nn.functional.linear(h, w, b))
        w_h = self._quantize(self.head.weight)
        b_h = self._quantize(self.head.bias)
        return torch.nn.functional.linear(h, w_h, b_h)


class DistillCompressor:
    """Production knowledge distillation compressor.

    Trains a teacher SIREN, then distills to a small student SIREN
    with INT4 quantization-aware training.
    """

    def __init__(self,
                 teacher_hidden=32,
                 student_hidden=8,
                 n_layers=3,
                 omega=15.0,
                 quant_bits=4,
                 teacher_epochs=500,
                 student_epochs=800,
                 lr=1e-3):
        self.teacher_hidden = teacher_hidden
        self.student_hidden = student_hidden
        self.n_layers = n_layers
        self.omega = omega
        self.quant_bits = quant_bits
        self.teacher_epochs = teacher_epochs
        self.student_epochs = student_epochs
        self.lr = lr

    def compress(self, image: np.ndarray, verbose: bool = False) -> dict:
        """Compress RGB image to .blke recipe.

        Args:
            image: uint8 array, shape (H, W, 3) or (H, W)
            verbose: print progress

        Returns:
            dict with 'recipe_bytes', 'original_size', 'compression_ratio',
            'psnr_db', 'train_time_s'
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for DistillCompressor")

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        H, W, C = image.shape
        assert image.dtype == np.uint8

        # Use luminance (Y channel) for SIREN fitting
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        gray_norm = (gray / 255.0).astype(np.float32)

        # Generate coordinates
        coords = np.stack(np.meshgrid(
            np.linspace(0, 1, W),
            np.linspace(0, 1, H)
        ), axis=-1).reshape(-1, 2)

        xt = torch.tensor(coords, dtype=torch.float32)
        yt = torch.tensor(gray_norm.flatten(), dtype=torch.float32)

        # Step 1: Train teacher SIREN
        if verbose:
            print(f"  Training teacher (h={self.teacher_hidden})...")
        torch.manual_seed(0)
        teacher = _DistillSIREN(self.teacher_hidden, self.n_layers,
                                self.omega, quant_bits=32)
        opt_t = torch.optim.Adam(teacher.parameters(), lr=self.lr)
        for ep in range(self.teacher_epochs):
            opt_t.zero_grad()
            pred = teacher(xt).squeeze(-1)
            loss = ((pred - yt) ** 2).mean()
            loss.backward()
            opt_t.step()

        teacher.eval()
        with torch.no_grad():
            teacher_out = teacher(xt).squeeze(-1)

        # Step 2: Distill to student with INT4 QAT
        if verbose:
            print(f"  Distilling to student (h={self.student_hidden}, INT{self.quant_bits})...")
        torch.manual_seed(42)
        student = _DistillSIREN(self.student_hidden, self.n_layers,
                                self.omega, quant_bits=self.quant_bits)
        opt_s = torch.optim.Adam(student.parameters(), lr=self.lr)
        for ep in range(self.student_epochs):
            opt_s.zero_grad()
            student_out = student(xt).squeeze(-1)
            loss = ((student_out - teacher_out) ** 2).mean()
            loss.backward()
            opt_s.step()

        # Step 3: Extract student params and serialize
        student.eval()
        with torch.no_grad():
            recon_gray = student(xt).squeeze(-1).numpy()
        recon_gray = (recon_gray * 255).clip(0, 255).astype(np.uint8).reshape(H, W)

        # Reconstruct RGB (use grayscale for Y, average for chroma)
        recon_rgb = np.stack([recon_gray] * 3, axis=-1)
        psnr = _compute_psnr(image, recon_rgb)

        # Extract params
        params = []
        for p in student.parameters():
            params.append(p.detach().numpy().flatten())
        params_flat = np.concatenate(params)

        recipe_bytes = self._serialize(params_flat, H, W, C)

        return {
            'recipe_bytes': recipe_bytes,
            'original_size': H * W * C,
            'compression_ratio': (H * W * C) / len(recipe_bytes),
            'psnr_db': psnr,
            'teacher_params': _siren_param_count(self.teacher_hidden, self.n_layers),
            'student_params': len(params_flat),
        }

    def _serialize(self, params_flat, H, W, C):
        """Serialize params as INT4-packed bytes."""
        levels = 2 ** self.quant_bits - 1
        max_val = np.abs(params_flat).max()
        scale = max_val / (levels / 2)
        quantized = np.round(params_flat / scale).astype(np.int8)
        quantized = np.clip(quantized, -levels // 2, levels // 2)

        # Pack 2 INT4 per byte
        unsigned = (quantized + levels // 2).astype(np.uint8)
        packed = np.zeros(len(unsigned) // 2 + 1, dtype=np.uint8)
        for i in range(0, len(unsigned) - 1, 2):
            packed[i // 2] = (unsigned[i] << 4) | unsigned[i + 1]
        if len(unsigned) % 2 == 1:
            packed[-1] = unsigned[-1] << 4

        recipe = bytearray()
        recipe.extend(MAGIC_DISTILL)
        recipe.append(VERSION_DISTILL)
        recipe.append(self.student_hidden)
        recipe.append(self.n_layers)
        recipe.extend(struct.pack('<H', H))
        recipe.extend(struct.pack('<H', W))
        recipe.append(C)
        recipe.extend(struct.pack('<f', float(scale)))
        recipe.extend(packed.tobytes())
        return bytes(recipe)

    @staticmethod
    def decompress(recipe_bytes: bytes) -> np.ndarray:
        """Decompress .blke recipe to RGB image."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for DistillCompressor")

        if recipe_bytes[:4] != MAGIC_DISTILL:
            raise ValueError(f"Bad magic: {recipe_bytes[:4]}")
        pos = 4
        version = recipe_bytes[pos]; pos += 1
        hidden = recipe_bytes[pos]; pos += 1
        n_layers = recipe_bytes[pos]; pos += 1
        H = struct.unpack('<H', recipe_bytes[pos:pos+2])[0]; pos += 2
        W = struct.unpack('<H', recipe_bytes[pos:pos+2])[0]; pos += 2
        C = recipe_bytes[pos]; pos += 1
        scale = struct.unpack('<f', recipe_bytes[pos:pos+4])[0]; pos += 4

        packed = recipe_bytes[pos:]
        n_params = _siren_param_count(hidden, n_layers)
        levels = 15  # INT4
        unsigned = np.zeros(n_params, dtype=np.uint8)
        for i in range(n_params):
            byte_idx = i // 2
            if i % 2 == 0:
                unsigned[i] = (packed[byte_idx] >> 4) & 0x0F
            else:
                unsigned[i] = packed[byte_idx] & 0x0F

        signed = unsigned.astype(np.int8) - levels // 2
        params_flat = signed.astype(np.float32) * scale

        omega = 15.0
        coords = np.stack(np.meshgrid(
            np.linspace(0, 1, W),
            np.linspace(0, 1, H)
        ), axis=-1).reshape(-1, 2)

        model = _DistillSIREN(hidden, n_layers, omega, quant_bits=32)
        idx = 0
        with torch.no_grad():
            for p in model.parameters():
                n = int(np.prod(p.shape))
                p.copy_(torch.tensor(params_flat[idx:idx+n].reshape(p.shape),
                                      dtype=torch.float32))
                idx += n

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(coords, dtype=torch.float32)).squeeze(-1).numpy()

        gray = (pred * 255).clip(0, 255).astype(np.uint8).reshape(H, W)
        return np.stack([gray] * 3, axis=-1)


def compress_image(image: np.ndarray, **kwargs) -> dict:
    """Convenience function to compress an image."""
    compressor = DistillCompressor(**kwargs)
    return compressor.compress(image)


def decompress_image(recipe_bytes: bytes) -> np.ndarray:
    """Convenience function to decompress."""
    return DistillCompressor.decompress(recipe_bytes)
