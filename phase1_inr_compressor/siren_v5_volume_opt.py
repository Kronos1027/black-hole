"""
siren_v5_volume_opt.py — v5.14 3D Volume with DCT-based residual
=================================================================
Improvement over v5.12: instead of using zlib on XOR bytes for the 3D
volume residual (which ignores 3D spatial structure), we use a 3D DCT
(Discrete Cosine Transform) approach similar to how JPEG uses 2D DCT.

Pipeline:
  1. Train SIREN f(x,y,z) -> value (same as v5.12)
  2. Quantize weights (INT8)
  3. Inference -> predicted volume
  4. residual = (original - predicted) mod 256 (int16, can be negative)
  5. Apply 3D DCT block-wise (8x8x8 blocks) to the residual
  6. Quantize DCT coefficients (drop high frequencies aggressively)
  7. zlib-compress the quantized coefficients
  8. Pack: SIREN weights + DCT-quantized residual + SHA-256

The 3D DCT captures spatial correlation along all 3 axes — much better
than zlib's byte-level LZ77 for volumetric data. This should close the
gap with ZIP on 3D volumes, especially for smooth medical data.

Recipe format (.blk4):
  [magic 'BLK4'][version][bits]
  [hidden_features][hidden_layers][omega_0]
  [D][H][W][C]
  [siren_packed + meta]
  [dct_block_size][dct_quality]
  [dct_residual_compressed][sha]
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
from siren_v5_volume import VolumeSIREN


MAGIC_VOLUME_OPT = b'BLK4'
VERSION_VOLUME_OPT = 1


# ============================================================
#  3D DCT utilities
# ============================================================
def make_dct_matrix(N: int) -> np.ndarray:
    """Create the 1D DCT-II matrix (N x N)."""
    n = np.arange(N)
    k = n.reshape(-1, 1)
    M = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    M *= np.sqrt(2.0 / N)
    M[0, :] *= 1.0 / np.sqrt(2)
    return M.astype(np.float32)


def dct3_blockwise(volume: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Apply 3D DCT block-wise to a volume.
    volume: (D, H, W) float32
    Returns: (D, H, W) DCT coefficients (same shape, may be padded)
    """
    D, H, W = volume.shape
    # Pad to multiple of block_size
    D_pad = (D + block_size - 1) // block_size * block_size
    H_pad = (H + block_size - 1) // block_size * block_size
    W_pad = (W + block_size - 1) // block_size * block_size
    padded = np.zeros((D_pad, H_pad, W_pad), dtype=np.float32)
    padded[:D, :H, :W] = volume

    # Build DCT matrices
    M_d = make_dct_matrix(D_pad // block_size * block_size // block_size * block_size)
    # Actually we need per-block DCT, so use block_size matrices
    M = make_dct_matrix(block_size)

    result = np.zeros_like(padded)
    # Process each block
    for d in range(0, D_pad, block_size):
        for h in range(0, H_pad, block_size):
            for w in range(0, W_pad, block_size):
                block = padded[d:d+block_size, h:h+block_size, w:w+block_size]
                # 3D DCT = apply 1D DCT along each axis
                # block shape: (block_size, block_size, block_size)
                # Apply along axis 0: result = M @ block
                t = np.einsum('ij,jkl->ikl', M, block)
                # Apply along axis 1: t2 = einsum('ij,kjl->kil', M, t)
                t = np.einsum('ij,kjl->kil', M, t)
                # Apply along axis 2: t3 = einsum('ij,klj->kli', M, t)
                t = np.einsum('ij,klj->kli', M, t)
                result[d:d+block_size, h:h+block_size, w:w+block_size] = t
    return result[:D, :H, :W]


def idct3_blockwise(coeffs: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Inverse 3D DCT (block-wise)."""
    D, H, W = coeffs.shape
    D_pad = (D + block_size - 1) // block_size * block_size
    H_pad = (H + block_size - 1) // block_size * block_size
    W_pad = (W + block_size - 1) // block_size * block_size
    padded = np.zeros((D_pad, H_pad, W_pad), dtype=np.float32)
    padded[:D, :H, :W] = coeffs

    M = make_dct_matrix(block_size)
    M_inv = M.T  # DCT matrix is orthogonal, so inverse = transpose

    result = np.zeros_like(padded)
    for d in range(0, D_pad, block_size):
        for h in range(0, H_pad, block_size):
            for w in range(0, W_pad, block_size):
                block = padded[d:d+block_size, h:h+block_size, w:w+block_size]
                # Inverse: apply M_inv along each axis (reverse order)
                t = np.einsum('ij,klj->kli', M_inv, block)
                t = np.einsum('ij,kjl->kil', M_inv, t)
                t = np.einsum('ij,jkl->ikl', M_inv, t)
                result[d:d+block_size, h:h+block_size, w:w+block_size] = t
    return result[:D, :H, :W]


def quantize_dct_coeffs(coeffs: np.ndarray, quality: int = 50) -> np.ndarray:
    """Quantize DCT coefficients. Quality 1-100 (higher = better quality).
    Low frequencies kept, high frequencies aggressively quantized.
    """
    # Build quantization matrix (zigzag-like: low freq small, high freq large)
    block_size = 8
    # Simple quality scaling
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    # Quantization matrix: smaller for low freq, larger for high freq
    # Use a 3D zigzag approximation: q[i,j,k] increases with i+j+k
    q = np.zeros((block_size, block_size, block_size), dtype=np.float32)
    for i in range(block_size):
        for j in range(block_size):
            for k in range(block_size):
                # Frequency magnitude ~ i+j+k
                base = 1 + (i + j + k) * 2
                q[i, j, k] = max(1, base * scale / 100.0)

    D, H, W = coeffs.shape
    D_pad = (D + block_size - 1) // block_size * block_size
    H_pad = (H + block_size - 1) // block_size * block_size
    W_pad = (W + block_size - 1) // block_size * block_size
    padded = np.zeros((D_pad, H_pad, W_pad), dtype=np.float32)
    padded[:D, :H, :W] = coeffs

    quantized = np.zeros_like(padded)
    for d in range(0, D_pad, block_size):
        for h in range(0, H_pad, block_size):
            for w in range(0, W_pad, block_size):
                block = padded[d:d+block_size, h:h+block_size, w:w+block_size]
                quantized[d:d+block_size, h:h+block_size, w:w+block_size] = np.round(block / q)
    return quantized[:D, :H, :W].astype(np.int16)


def dequantize_dct_coeffs(quantized: np.ndarray, quality: int = 50) -> np.ndarray:
    """Inverse quantization."""
    block_size = 8
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    q = np.zeros((block_size, block_size, block_size), dtype=np.float32)
    for i in range(block_size):
        for j in range(block_size):
            for k in range(block_size):
                base = 1 + (i + j + k) * 2
                q[i, j, k] = max(1, base * scale / 100.0)

    D, H, W = quantized.shape
    D_pad = (D + block_size - 1) // block_size * block_size
    H_pad = (H + block_size - 1) // block_size * block_size
    W_pad = (W + block_size - 1) // block_size * block_size
    padded = np.zeros((D_pad, H_pad, W_pad), dtype=np.float32)
    padded[:D, :H, :W] = quantized.astype(np.float32)

    result = np.zeros_like(padded)
    for d in range(0, D_pad, block_size):
        for h in range(0, H_pad, block_size):
            for w in range(0, W_pad, block_size):
                block = padded[d:d+block_size, h:h+block_size, w:w+block_size]
                result[d:d+block_size, h:h+block_size, w:w+block_size] = block * q
    return result[:D, :H, :W]


# ============================================================
#  Optimized Volume Compressor
# ============================================================
class VolumeCompressorOpt:
    """
    3D Volume compressor with DCT-based residual coding.
    Improvement over v5.12: uses 3D DCT to compress residual, not zlib on XOR.
    """

    def __init__(self, hidden_features=64, hidden_layers=3, omega_0=30.0,
                 dct_block_size: int = 8, dct_quality: int = 50,
                 device: str | None = None):
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = float(omega_0)
        self.dct_block_size = dct_block_size
        self.dct_quality = dct_quality
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _make_model(self, out_features: int) -> VolumeSIREN:
        return VolumeSIREN(
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            omega_0=self.omega_0,
            out_features=out_features,
        ).to(self.device)

    def _make_coords(self, D: int, H: int, W: int) -> torch.Tensor:
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
        """Compress 3D volume with DCT residual coding. Volume: (D, H, W, C) uint8."""
        assert volume.dtype == np.uint8 and volume.ndim == 4
        D, H, W, C = volume.shape
        t0 = time.time()

        target_np = (volume.astype(np.float32) / 127.5 - 1.0).reshape(-1, C)
        target = torch.from_numpy(target_np).to(self.device)
        coords = self._make_coords(D, H, W)
        n_total = coords.shape[0]
        if batch_size is None:
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
            opt.zero_grad(); loss.backward(); opt.step()
            if epoch >= warmup:
                sched.step()
            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  vol-opt epoch {epoch}/{epochs}  loss={loss.item():.6e}")
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(float(loss.item()))

        train_time = time.time() - t0

        # Quantize weights
        weights_np = model.state_to_numpy()
        packed, packed_meta = quantize_int8(weights_np)
        q_weights = dequantize_int8(packed, packed_meta)
        model.load_from_numpy(q_weights)
        model.eval()

        # Inference
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
        predicted = np.clip((predicted + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(D, H, W, C)

        # DCT-based residual coding (per channel)
        original_bytes = volume.tobytes()
        # For each channel, compute residual and DCT-encode it
        residual_coeffs_all = []
        for c in range(C):
            # residual in [-255, 255] range
            residual = volume[..., c].astype(np.int16) - predicted[..., c].astype(np.int16)
            # DCT works on float
            residual_f = residual.astype(np.float32)
            # 3D DCT
            coeffs = dct3_blockwise(residual_f, self.dct_block_size)
            # Quantize
            quantized = quantize_dct_coeffs(coeffs, self.dct_quality)
            residual_coeffs_all.append(quantized)

        # Pack all channels' quantized coefficients
        # Convert to int16, then to bytes
        coeffs_packed = b''
        for c in range(C):
            coeffs_packed += residual_coeffs_all[c].tobytes()
        # zlib compress
        residual_compressed = zlib.compress(coeffs_packed, 9)

        sha = hashlib.sha256(original_bytes).digest()

        # Bit accuracy (informational)
        a = np.frombuffer(original_bytes, dtype=np.uint8)
        b = np.frombuffer(predicted.tobytes(), dtype=np.uint8)
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
            'dct_quality': self.dct_quality,
        }

    def _pack_recipe(self, bits: int, packed: bytes, packed_meta: list,
                     D: int, H: int, W: int, C: int,
                     residual_compressed: bytes, sha: bytes) -> bytes:
        out = bytearray()
        out += MAGIC_VOLUME_OPT
        out += struct.pack('<B', VERSION_VOLUME_OPT)
        out += struct.pack('<B', bits)
        out += struct.pack('<H', self.hidden_features)
        out += struct.pack('<B', self.hidden_layers)
        out += struct.pack('<f', self.omega_0)
        out += struct.pack('<H', D)
        out += struct.pack('<H', H)
        out += struct.pack('<H', W)
        out += struct.pack('<B', C)
        out += struct.pack('<B', self.dct_block_size)
        out += struct.pack('<B', self.dct_quality)
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
        if buf[:4] != MAGIC_VOLUME_OPT:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_VOLUME_OPT
        bits = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        hidden = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden_l = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        omega = struct.unpack('<f', buf[off:off+4])[0]; off += 4
        D = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        H = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        W = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        C = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        dct_block = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        dct_quality = struct.unpack('<B', buf[off:off+1])[0]; off += 1

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

        # Inference
        zs, ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, D, device=dev),
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing='ij',
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], dim=-1)
        n_total = D * H * W
        CHUNK = 1 << 18
        preds = []
        with torch.no_grad():
            for i in range(0, n_total, CHUNK):
                preds.append(model(coords[i:i+CHUNK]).cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(D, H, W, C)

        # Decompress DCT residual
        coeffs_packed = zlib.decompress(residual_compressed)
        # Each channel: D*H*W int16 = D*H*W*2 bytes
        chan_bytes = D * H * W * 2
        recovered = np.zeros((D, H, W, C), dtype=np.uint8)
        for c in range(C):
            chan_coeffs_bytes = coeffs_packed[c*chan_bytes:(c+1)*chan_bytes]
            quantized = np.frombuffer(chan_coeffs_bytes, dtype=np.int16).astype(np.float32).reshape(D, H, W)
            # Dequantize
            coeffs = dequantize_dct_coeffs(quantized, dct_quality)
            # Inverse DCT
            residual = idct3_blockwise(coeffs, dct_block)
            # Add to predicted, clip to [0, 255]
            combined = predicted[..., c].astype(np.float32) + residual
            recovered[..., c] = np.clip(np.round(combined), 0, 255).astype(np.uint8)

        # NOTE: DCT residual is LOSSY (due to quantization). The recovered
        # volume may NOT be bit-perfect. We use SHA-256 to detect this.
        sha_got = hashlib.sha256(recovered.tobytes()).digest()

        return recovered, {
            'D': D, 'H': H, 'W': W, 'C': C,
            'bits': bits,
            'dct_quality': dct_quality,
            'sha256_match': sha_got == sha_expected,
            'exact_match': sha_got == sha_expected,
            'mode': 'volume_opt',
            'lossless': sha_got == sha_expected,
            'shape': (D, H, W, C),
        }


# ============================================================
#  Self-test
# ============================================================
def _self_test():
    import zlib as _z
    print(f"[vol-opt] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"[vol-opt] Mode: 3D SIREN + DCT residual (lossy residual coding)")

    # Generate a synthetic MRI-like volume
    D, H, W, C = 16, 32, 32, 1
    zs, ys, xs = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
    zs = zs / (D-1) * 2 - 1; ys = ys / (H-1) * 2 - 1; xs = xs / (W-1) * 2 - 1
    rng = np.random.default_rng(42)
    vol = np.zeros((D, H, W), dtype=np.float32)
    vol += 80
    vol += 30 * np.sin(xs * 3) * np.cos(ys * 2)
    for _ in range(3):
        cz, cy, cx = rng.uniform(-0.5, 0.5, 3)
        sigma = rng.uniform(0.2, 0.4)
        amp = rng.uniform(80, 150)
        vol += amp * np.exp(-((xs-cx)**2 + (ys-cy)**2 + (zs-cz)**2) / (2*sigma**2))
    vol = np.clip(vol, 0, 255).astype(np.uint8)[..., None]

    total_orig = vol.nbytes
    zip_size = len(_z.compress(vol.tobytes(), 9))
    print(f"[vol-opt] Volume: {vol.shape} = {total_orig:,}B")
    print(f"[vol-opt] ZIP: {zip_size:,}B ({total_orig/zip_size:.2f}x)")

    # Compare v5.12 (XOR+zlib) vs v5.14 (DCT)
    from siren_v5_volume import VolumeCompressor

    print(f"\n--- v5.12 (XOR+zlib residual) ---")
    comp12 = VolumeCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0)
    res12 = comp12.compress(vol, epochs=800, lr=1e-3, bits=8, batch_size=4096, verbose=False)
    rec12, meta12 = VolumeCompressor.decompress(res12['recipe_bytes'])
    print(f"  recipe: {res12['recipe_size']:,}B  bit%: {res12['model_bit_accuracy']:.1f}  "
          f"SHA: {meta12['exact_match']}")

    print(f"\n--- v5.14 (DCT residual, quality=50) ---")
    comp14 = VolumeCompressorOpt(hidden_features=64, hidden_layers=3, omega_0=30.0,
                                   dct_block_size=8, dct_quality=50)
    res14 = comp14.compress(vol, epochs=800, lr=1e-3, bits=8, batch_size=4096, verbose=False)
    rec14, meta14 = VolumeCompressorOpt.decompress(res14['recipe_bytes'])
    # PSNR for DCT (lossy residual)
    mse14 = np.mean((vol.astype(float) - rec14.astype(float))**2)
    psnr14 = 10 * np.log10(255**2 / mse14) if mse14 > 0 else float('inf')
    print(f"  recipe: {res14['recipe_size']:,}B  bit%: {res14['model_bit_accuracy']:.1f}  "
          f"SHA: {meta14['exact_match']}  PSNR: {psnr14:.1f}dB")

    print(f"\n--- v5.14 (DCT residual, quality=90) ---")
    comp14b = VolumeCompressorOpt(hidden_features=64, hidden_layers=3, omega_0=30.0,
                                    dct_block_size=8, dct_quality=90)
    res14b = comp14b.compress(vol, epochs=800, lr=1e-3, bits=8, batch_size=4096, verbose=False)
    rec14b, meta14b = VolumeCompressorOpt.decompress(res14b['recipe_bytes'])
    mse14b = np.mean((vol.astype(float) - rec14b.astype(float))**2)
    psnr14b = 10 * np.log10(255**2 / mse14b) if mse14b > 0 else float('inf')
    print(f"  recipe: {res14b['recipe_size']:,}B  bit%: {res14b['model_bit_accuracy']:.1f}  "
          f"SHA: {meta14b['exact_match']}  PSNR: {psnr14b:.1f}dB")

    print(f"\n=== Summary ===")
    print(f"  ZIP:           {zip_size:,}B")
    print(f"  v5.12 (lossless):  {res12['recipe_size']:,}B  (bit-perfect: {meta12['exact_match']})")
    print(f"  v5.14 q=50 (lossy): {res14['recipe_size']:,}B  (PSNR: {psnr14:.1f}dB)")
    print(f"  v5.14 q=90 (lossy): {res14b['recipe_size']:,}B  (PSNR: {psnr14b:.1f}dB)")


if __name__ == '__main__':
    _self_test()
