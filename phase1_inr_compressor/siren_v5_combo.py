"""
siren_v5_combo.py — v5.9 Combo: Hypernetwork + Hybrid Residual
================================================================
Combines the two best optimizations from v5.7 and v5.8:

  v5.7 Hypernetwork: shared HyperNet generates SIREN weights from a small
                     per-image latent (16 bytes INT8).
  v5.8 Hybrid residual: encode the prediction error as WebP/PNG lossless
                         instead of XOR+zlib (1.1x to 3x smaller).

The combo should be the best of both worlds for datacenter use cases:
  - Shared hypernetwork amortizes across N images (6.3KB shared, 16B/img)
  - WebP residual crushes the per-image error (2-3x smaller than zlib)
  - Total per-image cost: 16B latent + ~3-5KB residual = ~5KB/img

This is the recommended mode for compressing 10-100+ similar smooth images.
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
from siren_v5_hyper import HyperNetwork, SIRENFunction
from siren_v5_hybrid import (
    encode_residual_webp, decode_residual_webp,
    encode_residual_png, decode_residual_png,
    encode_residual_zlib, decode_residual_zlib,
)


MAGIC_COMBO = b'BLK9'
VERSION_COMBO = 1


class ComboCompressor:
    """
    v5.9 Combo: Hypernetwork + Hybrid residual.

    Pipeline:
      Phase 1 (train_base): train shared HyperNetwork on corpus
      Phase 2 (compress_many): per image, train latent z, encode residual as WebP

    Recipe format (.blkh9):
      [magic 'BLK9'][version][bits][codec_id]
      [latent_dim][hidden_features][hidden_layers][omega_0][H][W][C][N]
      [hyper_packed + meta]
      for each image: [latent_q + scale][residual_compressed][sha]
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

        # SIREN function (forward-only)
        self.siren_fn = SIRENFunction(
            in_features=2, hidden_features=hidden_features,
            hidden_layers=hidden_layers, out_features=3, omega_0=omega_0
        ).to(self.device)
        target_params = self.siren_fn.total_params

        # HyperNetwork (single linear layer)
        self.hyper = HyperNetwork(
            latent_dim=latent_dim,
            target_total_params=target_params,
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

    # ============================================================
    #  Phase 1: Train the HyperNetwork on a corpus
    # ============================================================
    def train_base(self, images: list[np.ndarray],
                   epochs: int = 3000, lr: float = 1e-3,
                   batch_size: int = 2048,
                   verbose: bool = False) -> dict:
        """Train the shared HyperNetwork on a corpus of similar images."""
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

        opt = torch.optim.Adam(
            list(self.hyper.parameters()) + latents, lr=lr
        )
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

            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch >= warmup:
                sched.step()

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  combo base epoch {epoch}/{epochs}  img={idx}  loss={loss.item():.6e}")
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(float(loss.item()))

        train_time = time.time() - t0
        self._base_trained = True
        self._cached_hyper_state = {
            name: p.detach().cpu().numpy().astype(np.float32)
            for name, p in self.hyper.named_parameters()
        }
        return {
            'train_time_s': train_time,
            'n_images': N,
            'final_loss': history[-1] if history else float('nan'),
            'history': history,
        }

    # ============================================================
    #  Phase 2: Compress N images (train latents + hybrid residual)
    # ============================================================
    def compress_many(self, images: list[np.ndarray],
                      epochs: int = 1000, lr: float = 3e-3,
                      bits: int = 8,
                      batch_size: int | None = None,
                      verbose: bool = False) -> dict:
        """Compress N images. Per-image cost: latent (16B) + WebP residual."""
        if not self._base_trained or self._cached_hyper_state is None:
            raise RuntimeError("Must call train_base() first")
        assert all(im.shape == images[0].shape for im in images)
        H, W, C = images[0].shape
        N = len(images)

        # Quantize hypernetwork weights (paid ONCE)
        hyper_state_np = self._cached_hyper_state
        packed_hyper, hyper_meta = quantize_int8(hyper_state_np)
        hyper_q = dequantize_int8(packed_hyper, hyper_meta)
        with torch.no_grad():
            for name, p in self.hyper.named_parameters():
                p.copy_(torch.from_numpy(hyper_q[name]))
        self.hyper.eval()

        per_image_data = []
        total_residual = 0
        bit_pcts = []
        t0 = time.time()
        for i, img in enumerate(images):
            if verbose:
                print(f"  combo compressing image {i+1}/{N}...")
            target = torch.from_numpy(
                (img.astype(np.float32) / 127.5 - 1.0).reshape(-1, 3)
            ).to(self.device)
            coords = self._make_coords(H, W)
            n = coords.shape[0]
            bs = min(batch_size or n, n)

            # Train latent z
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

            # Quantize latent
            z_np = z.detach().cpu().numpy().astype(np.float32)
            z_max_abs = float(np.max(np.abs(z_np))) if z_np.size > 0 else 0.0
            z_scale = z_max_abs / 127.0 if z_max_abs > 0 else 1.0
            z_q = np.clip(np.round(z_np / z_scale), -127, 127).astype(np.int8)
            z_q_tensor = torch.from_numpy((z_q.astype(np.float32) * z_scale)).to(self.device)

            # Inference with quantized latent + quantized hypernetwork
            with torch.no_grad():
                flat_w = self.hyper(z_q_tensor.unsqueeze(0)).squeeze(0)
                pred = self.siren_fn(coords, flat_w).cpu().numpy()
            predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(H, W, C)

            # HYBRID RESIDUAL: encode as image, not XOR bytes
            original_bytes = img.tobytes()
            residual_img = ((img.astype(np.int16) - predicted.astype(np.int16)) % 256).astype(np.uint8)
            # Sanity
            recovered_check = ((predicted.astype(np.int16) + residual_img.astype(np.int16)) % 256).astype(np.uint8)
            assert np.array_equal(recovered_check, img), f"image {i}: residual math failed"

            # Encode residual with chosen codec
            if self.residual_codec == 'webp':
                residual_compressed = encode_residual_webp(residual_img)
            elif self.residual_codec == 'png':
                residual_compressed = encode_residual_png(residual_img)
            else:
                residual_compressed = encode_residual_zlib(residual_img)

            sha = hashlib.sha256(original_bytes).digest()

            # Bit accuracy (informational)
            a = np.frombuffer(original_bytes, dtype=np.uint8)
            b = np.frombuffer(predicted.tobytes(), dtype=np.uint8)
            bit_acc = float(np.mean(np.unpackbits(a) == np.unpackbits(b))) * 100

            per_image_data.append({
                'latent_q': z_q.tobytes(),
                'latent_scale': z_scale,
                'residual_compressed': residual_compressed,
                'sha': sha,
            })
            total_residual += len(residual_compressed)
            bit_pcts.append(bit_acc)

        total_time = time.time() - t0

        recipe = self._pack_recipe(bits, packed_hyper, hyper_meta,
                                     H, W, C, per_image_data)
        return {
            'recipe_bytes': recipe,
            'recipe_size': len(recipe),
            'hyper_size': len(packed_hyper),
            'latent_per_image': self.latent_dim,
            'residual_total': total_residual,
            'residual_per_image': total_residual / N,
            'avg_bit_pct': float(np.mean(bit_pcts)),
            'min_bit_pct': float(np.min(bit_pcts)),
            'max_bit_pct': float(np.max(bit_pcts)),
            'n_images': N,
            'total_time_s': total_time,
            'per_image_time_s': total_time / N,
            'residual_codec': self.residual_codec,
        }

    def _pack_recipe(self, bits: int, packed_hyper: bytes, hyper_meta: list,
                     H: int, W: int, C: int, per_image: list[dict]) -> bytes:
        out = bytearray()
        out += MAGIC_COMBO
        out += struct.pack('<B', VERSION_COMBO)
        out += struct.pack('<B', bits)
        codec_id = {'zlib': 0, 'png': 1, 'webp': 2}[self.residual_codec]
        out += struct.pack('<B', codec_id)
        out += struct.pack('<H', self.latent_dim)
        out += struct.pack('<H', self.hidden_features)
        out += struct.pack('<B', self.hidden_layers)
        out += struct.pack('<f', self.omega_0)
        out += struct.pack('<H', H)
        out += struct.pack('<H', W)
        out += struct.pack('<B', C)
        out += struct.pack('<H', len(per_image))
        # Hypernetwork weights (shared)
        out += struct.pack('<I', len(packed_hyper))
        out += packed_hyper
        out += struct.pack('<H', len(hyper_meta))
        for entry in hyper_meta:
            n_bytes, shape, scale, name = entry
            name_b = name.encode('utf-8')
            out += struct.pack('<B', len(name_b)); out += name_b
            out += struct.pack('<I', n_bytes)
            out += struct.pack('<B', len(shape))
            for d in shape: out += struct.pack('<i', int(d))
            out += struct.pack('<d', float(scale))
            out += struct.pack('<I', 0)
        # Per-image: latent + residual + sha
        for d in per_image:
            out += struct.pack('<I', len(d['latent_q']))
            out += d['latent_q']
            out += struct.pack('<d', float(d['latent_scale']))
            out += struct.pack('<Q', len(d['residual_compressed']))
            out += d['residual_compressed']
            out += d['sha']
        return bytes(out)

    @staticmethod
    def decompress(recipe_bytes: bytes, device: str | None = None) -> tuple[list[np.ndarray], dict]:
        buf = recipe_bytes
        off = 0
        if buf[:4] != MAGIC_COMBO:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_COMBO
        bits = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        codec_id = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        codec_name = {0: 'zlib', 1: 'png', 2: 'webp'}[codec_id]
        latent_dim = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden_l = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        omega = struct.unpack('<f', buf[off:off+4])[0]; off += 4
        H = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        W = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        C = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        N = struct.unpack('<H', buf[off:off+2])[0]; off += 2

        # Hypernetwork
        hyper_size = struct.unpack('<I', buf[off:off+4])[0]; off += 4
        packed_hyper = buf[off:off+hyper_size]; off += hyper_size
        n_hyper_meta = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hyper_meta = []
        for _ in range(n_hyper_meta):
            name_len = struct.unpack('<B', buf[off:off+1])[0]; off += 1
            name = buf[off:off+name_len].decode('utf-8'); off += name_len
            n_bytes = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            ndim = struct.unpack('<B', buf[off:off+1])[0]; off += 1
            shape = tuple(struct.unpack('<' + 'i'*ndim, buf[off:off+4*ndim])); off += 4*ndim
            scale = struct.unpack('<d', buf[off:off+8])[0]; off += 8
            _ = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            hyper_meta.append((n_bytes, shape, scale, name))

        hyper_dict = dequantize_int8(packed_hyper, hyper_meta)

        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        siren_fn = SIRENFunction(in_features=2, hidden_features=hidden,
                                  hidden_layers=hidden_l, out_features=3,
                                  omega_0=omega).to(dev)
        hyper = HyperNetwork(latent_dim=latent_dim,
                              target_total_params=siren_fn.total_params).to(dev)
        with torch.no_grad():
            for name, p in hyper.named_parameters():
                p.copy_(torch.from_numpy(hyper_dict[name]))
        hyper.eval()

        # Coords
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing='ij',
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

        # Per-image
        images = []
        all_match = True
        per_image_sha = []
        for i in range(N):
            latent_size = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            latent_q = buf[off:off+latent_size]; off += latent_size
            latent_scale = struct.unpack('<d', buf[off:off+8])[0]; off += 8
            resid_size = struct.unpack('<Q', buf[off:off+8])[0]; off += 8
            residual_compressed = buf[off:off+resid_size]; off += resid_size
            sha_expected = buf[off:off+32]; off += 32

            # Dequantize latent
            z_q = np.frombuffer(latent_q, dtype=np.int8).astype(np.float32)
            z = torch.from_numpy(z_q * latent_scale).to(dev)

            # Inference
            with torch.no_grad():
                flat_w = hyper(z.unsqueeze(0)).squeeze(0)
                pred = siren_fn(coords, flat_w).cpu().numpy()
            predicted = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8).reshape(H, W, C)

            # Decode residual (HYBRID mode)
            if codec_id == 2:
                residual_img = decode_residual_webp(residual_compressed)
            elif codec_id == 1:
                residual_img = decode_residual_png(residual_compressed)
            else:
                residual_img = decode_residual_zlib(residual_compressed, (H, W, C))

            # Recover: (predicted + residual) mod 256
            recovered = ((predicted.astype(np.int16) + residual_img.astype(np.int16)) % 256).astype(np.uint8)
            sha_got = hashlib.sha256(recovered.tobytes()).digest()
            match = (sha_got == sha_expected)
            if not match:
                all_match = False
            per_image_sha.append(match)
            images.append(recovered)

        return images, {
            'n_images': N,
            'bits': bits,
            'latent_dim': latent_dim,
            'residual_codec': codec_name,
            'all_sha256_match': all_match,
            'sha256_per_image': per_image_sha,
            'mode': 'combo',
        }


# ============================================================
#  Self-test
# ============================================================
def _self_test():
    import zlib as _z
    print(f"[combo] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"[combo] Mode: hypernetwork + WebP residual")

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
    print(f"[combo] {N} images x {SIZE}x{SIZE}x3 = {total_orig:,}B")
    print(f"[combo] ZIP per-file total: {zip_total:,}B")

    comp = ComboCompressor(latent_dim=16, hidden_features=16, hidden_layers=1,
                            omega_0=30.0, residual_codec='webp')
    print(f"\n[combo] Phase 1: training HyperNetwork on {N} images...")
    t0 = time.time()
    base_meta = comp.train_base(images, epochs=3000, lr=1e-3,
                                 batch_size=2048, verbose=True)
    print(f"  base trained in {base_meta['train_time_s']:.1f}s, "
          f"final loss {base_meta['final_loss']:.6e}")

    print(f"\n[combo] Phase 2: compressing {N} images (latent + WebP residual)...")
    res = comp.compress_many(images, epochs=1000, lr=3e-3,
                              bits=8, batch_size=2048, verbose=False)
    print(f"\n[combo] BLKH Combo: {res['recipe_size']:,}B  "
          f"(ratio {total_orig/res['recipe_size']:.2f}x)")
    print(f"  hyper (shared):       {res['hyper_size']:,}B  "
          f"-> {res['hyper_size']/N:.0f}B/image amortized")
    print(f"  latent per image:     {res['latent_per_image']}B (INT8)")
    print(f"  residual per img:     {res['residual_per_image']:.0f}B ({res['residual_codec']})")
    print(f"  bit acc avg: {res['avg_bit_pct']:.1f}%")
    print(f"  time: {res['total_time_s']:.1f}s ({res['per_image_time_s']:.2f}s/image)")

    # Verify
    t0 = time.time()
    recovered, dmeta = ComboCompressor.decompress(res['recipe_bytes'])
    print(f"\n[combo] Decompress: {time.time()-t0:.1f}s  "
          f"all_sha256_match: {dmeta['all_sha256_match']}")
    print(f"  per-image SHA match: {sum(dmeta['sha256_per_image'])}/{dmeta['n_images']}")

    winner = "BLKH" if res['recipe_size'] < zip_total else "ZIP"
    print(f"\n[combo] vs ZIP: {zip_total/res['recipe_size']:.3f}x  -> {winner}")


if __name__ == '__main__':
    _self_test()
