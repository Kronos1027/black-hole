"""
siren_v5_hyper.py — v5.7 Hypernetwork Meta-Learning (true COIN++)
==================================================================
Replaces the v5.3 FiLM modulations (which didn't beat ZIP) with a proper
hypernetwork: a small shared network that GENERATES the SIREN weights from
a per-image latent vector.

Architecture:
  - Latent vector z (per image, e.g. 64 floats, quantized to INT8 = 256 bytes)
  - HyperNetwork H(z) -> produces all SIREN weights
  - SIREN(coords) -> RGB

Training:
  Phase 1 (train_base): for each epoch, sample one image, init a fresh latent
    z (or reuse if it's the same image), train BOTH the HyperNetwork and z.
    The HyperNetwork learns to map any z to a useful SIREN initialization.
  Phase 2 (compress): for each new image, freeze HyperNetwork, train only z.
    Per-image cost: latent_dim * (1 byte if INT8) = e.g. 256 bytes.

This is the proper COIN++ architecture. The HyperNetwork is a learned
"weight prior" — given a latent code, it produces the right SIREN weights
for ANY smooth image in the corpus distribution.

Recipe format (.blkh7):
  [magic 'BLK7'][version][base_meta][N x [latent + residual + sha]]
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
from siren_v5_torch import SineLayer, quantize_int8, dequantize_int8


MAGIC_HYPER = b'BLK7'
VERSION_HYPER = 1


# ============================================================
#  HyperNetwork: z -> SIREN weights
# ============================================================
class HyperNetwork(nn.Module):
    """
    Maps a latent vector z to a full set of SIREN weights.

    Architecture: a SINGLE linear layer from z to the flattened weight vector.
    This is the COIN++ design — the hypernetwork is just a learned linear
    projection, but it's enough because we train z per-image and the
    HyperNetwork learns the right "prior" over SIREN weights.

    For SIREN(2, 32, 2, 3) target = 2307 params:
      Linear(64, 2307) = 64*2307 + 2307 = ~150K params (vs 320K with MLP)
    """

    def __init__(self, latent_dim: int = 64, target_total_params: int = 2307):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_total_params = target_total_params
        # Single linear layer: z -> flat weights
        # Bias is critical — represents the "average" SIREN weights
        self.linear = nn.Linear(latent_dim, target_total_params)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns flattened weights (batch_size, target_total_params)."""
        return self.linear(z)


# ============================================================
#  SIREN with externally-provided weights (no internal params)
# ============================================================
class SIRENFunction(nn.Module):
    """SIREN forward pass using externally-provided flattened weights.
    Used during hypernetwork training so we can backprop through the
    hypernetwork into the latent z.
    """

    def __init__(self, in_features=2, hidden_features=32, hidden_layers=2,
                 out_features=3, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.omega_0 = float(omega_0)

        # Compute layout: which slice of the flat weight vector corresponds to which param
        self.layout = []  # list of (name, shape, numel)
        # Layer 1: linear(2 -> 32)
        self.layout.append(('layer1.weight', (hidden_features, in_features),
                            hidden_features * in_features))
        self.layout.append(('layer1.bias', (hidden_features,), hidden_features))
        # Hidden layers: linear(32 -> 32) x hidden_layers
        for i in range(hidden_layers):
            self.layout.append((f'hidden{i}.weight', (hidden_features, hidden_features),
                                hidden_features * hidden_features))
            self.layout.append((f'hidden{i}.bias', (hidden_features,), hidden_features))
        # Final: linear(32 -> 3)
        self.layout.append(('final.weight', (out_features, hidden_features),
                            out_features * hidden_features))
        self.layout.append(('final.bias', (out_features,), out_features))

        self.total_params = sum(n for _, _, n in self.layout)
        # Track offsets for slicing
        self.offsets = []
        o = 0
        for name, shape, n in self.layout:
            self.offsets.append((name, shape, o, o + n))
            o += n

    def forward(self, coords: torch.Tensor, flat_weights: torch.Tensor) -> torch.Tensor:
        """
        coords: (N, 2) input
        flat_weights: (total_params,) flattened weight vector
        Returns: (N, 3) RGB predictions
        """
        # Slice flat weights into per-layer tensors
        def get(name):
            for n, shape, start, end in self.offsets:
                if n == name:
                    return flat_weights[start:end].reshape(shape)
            raise KeyError(name)

        x = coords
        # Layer 1 (first SIREN layer, no omega scaling on input)
        w = get('layer1.weight'); b = get('layer1.bias')
        x = torch.sin(torch.nn.functional.linear(x, w, b))
        # Hidden layers
        for i in range(self.hidden_layers):
            w = get(f'hidden{i}.weight'); b = get(f'hidden{i}.bias')
            x = torch.sin(self.omega_0 * torch.nn.functional.linear(x, w, b))
        # Final
        w = get('final.weight'); b = get('final.bias')
        x = torch.nn.functional.linear(x, w, b)
        return x


# ============================================================
#  HyperCompressor
# ============================================================
class HyperCompressor:
    """
    Hypernetwork-based meta-learning compressor (true COIN++).
    """

    def __init__(self, latent_dim=64, hidden_features=32, hidden_layers=2,
                 omega_0=30.0,
                 device: str | None = None):
        self.latent_dim = latent_dim
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = float(omega_0)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # SIREN function (no internal params — used for forward only)
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
        # Cache the quantized hypernetwork weights for compress phase
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
        """Train the HyperNetwork on a corpus of similar images.
        Each image gets its own latent z; both z and HyperNetwork are trained.
        """
        assert all(im.shape == images[0].shape for im in images)
        H, W, C = images[0].shape
        assert C == 3
        N = len(images)

        # Pre-encode all images
        target_tensors = []
        for im in images:
            t = torch.from_numpy(
                (im.astype(np.float32) / 127.5 - 1.0).reshape(-1, 3)
            ).to(self.device)
            target_tensors.append(t)

        coords = self._make_coords(H, W)  # (H*W, 2)

        # Per-image latent vectors
        latents = [torch.zeros(self.latent_dim, device=self.device, requires_grad=True)
                   for _ in range(N)]

        # Optimizer: HyperNetwork params + all latents
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

            # Cycle through images
            idx = epoch % N
            target = target_tensors[idx]
            z = latents[idx]

            # Mini-batch
            n = coords.shape[0]
            if batch_size < n:
                bidx = torch.randint(0, n, (batch_size,), device=self.device)
                xb = coords[bidx]; yb = target[bidx]
            else:
                xb, yb = coords, target

            # HyperNetwork generates SIREN weights from z
            flat_w = self.hyper(z.unsqueeze(0)).squeeze(0)  # (total_params,)
            pred = self.siren_fn(xb, flat_w)
            loss = torch.nn.functional.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch >= warmup:
                sched.step()

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  hyper base epoch {epoch}/{epochs}  img={idx}  loss={loss.item():.6e}")
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(float(loss.item()))

        train_time = time.time() - t0
        self._base_trained = True
        # Cache hypernetwork state (as numpy)
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
    #  Phase 2: Compress N images (train only latents)
    # ============================================================
    def compress_many(self, images: list[np.ndarray],
                      epochs: int = 1000, lr: float = 2e-3,
                      bits: int = 8,
                      batch_size: int | None = None,
                      zlib_level: int = 9,
                      verbose: bool = False) -> dict:
        """Compress N images using the trained HyperNetwork. Per-image cost
        is just the latent vector (quantized to INT8 = latent_dim bytes) + residual.
        """
        if not self._base_trained or self._cached_hyper_state is None:
            raise RuntimeError("Must call train_base() first")
        assert all(im.shape == images[0].shape for im in images)
        H, W, C = images[0].shape
        N = len(images)

        # Quantize the hypernetwork weights (paid ONCE for the whole corpus)
        hyper_state_np = self._cached_hyper_state
        packed_hyper, hyper_meta = quantize_int8(hyper_state_np)
        # Reload quantized hypernetwork
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
                print(f"  compressing image {i+1}/{N}...")
            target = torch.from_numpy(
                (img.astype(np.float32) / 127.5 - 1.0).reshape(-1, 3)
            ).to(self.device)
            coords = self._make_coords(H, W)
            n = coords.shape[0]
            if batch_size is None or batch_size >= n:
                bs = n
            else:
                bs = batch_size

            # Fresh latent for this image
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

            # Quantize latent to INT8
            z_np = z.detach().cpu().numpy().astype(np.float32)
            z_max_abs = float(np.max(np.abs(z_np))) if z_np.size > 0 else 0.0
            z_scale = z_max_abs / 127.0 if z_max_abs > 0 else 1.0
            z_q = np.clip(np.round(z_np / z_scale), -127, 127).astype(np.int8)
            # Reload quantized latent
            z_q_tensor = torch.from_numpy((z_q.astype(np.float32) * z_scale)).to(self.device)

            # Inference with quantized latent + quantized hypernetwork
            with torch.no_grad():
                flat_w = self.hyper(z_q_tensor.unsqueeze(0)).squeeze(0)
                pred = self.siren_fn(coords, flat_w).cpu().numpy()
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
                'latent_q': z_q.tobytes(),
                'latent_scale': z_scale,
                'residual_compressed': residual_compressed,
                'sha': sha,
            })
            total_residual += len(residual_compressed)
            bit_pcts.append(bit_acc)

        total_time = time.time() - t0

        # Pack aggregate recipe
        recipe = self._pack_recipe_aggregate(
            bits, packed_hyper, hyper_meta,
            H, W, C, per_image_data
        )
        return {
            'recipe_bytes': recipe,
            'recipe_size': len(recipe),
            'hyper_size': len(packed_hyper),
            'latent_per_image': self.latent_dim,  # bytes (INT8 = 1 byte per dim)
            'residual_total': total_residual,
            'residual_per_image': total_residual / N,
            'avg_bit_pct': float(np.mean(bit_pcts)),
            'min_bit_pct': float(np.min(bit_pcts)),
            'max_bit_pct': float(np.max(bit_pcts)),
            'n_images': N,
            'total_time_s': total_time,
            'per_image_time_s': total_time / N,
        }

    def _pack_recipe_aggregate(self, bits: int, packed_hyper: bytes, hyper_meta: list,
                                H: int, W: int, C: int,
                                per_image: list[dict]) -> bytes:
        out = bytearray()
        out += MAGIC_HYPER
        out += struct.pack('<B', VERSION_HYPER)
        out += struct.pack('<B', bits)
        out += struct.pack('<H', self.latent_dim)
        out += struct.pack('<H', self.hidden_features)
        out += struct.pack('<B', self.hidden_layers)
        out += struct.pack('<f', self.omega_0)
        out += struct.pack('<H', H)
        out += struct.pack('<H', W)
        out += struct.pack('<B', C)
        out += struct.pack('<H', len(per_image))
        # HyperNetwork weights (shared, paid once)
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
            out += struct.pack('<I', 0)  # n_elem (int8 only)
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
        if buf[:4] != MAGIC_HYPER:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_HYPER
        bits = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        latent_dim = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden_l = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        omega = struct.unpack('<f', buf[off:off+4])[0]; off += 4
        H = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        W = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        C = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        N = struct.unpack('<H', buf[off:off+2])[0]; off += 2

        # HyperNetwork weights
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
        # Build models
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
            'bits': bits,
            'latent_dim': latent_dim,
            'hidden_features': hidden,
            'hidden_layers': hidden_l,
            'all_sha256_match': all_match,
            'sha256_per_image': per_image_sha,
        }


# ============================================================
#  Self-test
# ============================================================
def _self_test():
    import zlib as _z
    print(f"[hyper] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")

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
    print(f"[hyper] {N} images x {SIZE}x{SIZE}x3 = {total_orig:,}B")
    print(f"[hyper] ZIP per-file total: {zip_total:,}B")

    comp = HyperCompressor(latent_dim=16, hidden_features=16, hidden_layers=1,
                            omega_0=30.0)
    print(f"\n[hyper] Config: latent=16, hidden=16, layers=1")
    print(f"[hyper] Phase 1: training HyperNetwork on {N} images...")
    t0 = time.time()
    base_meta = comp.train_base(images, epochs=3000, lr=1e-3,
                                 batch_size=2048, verbose=True)
    print(f"  base trained in {base_meta['train_time_s']:.1f}s, "
          f"final loss {base_meta['final_loss']:.6e}")

    print(f"\n[hyper] Phase 2: compressing {N} images (train only latents)...")
    res = comp.compress_many(images, epochs=1000, lr=3e-3,
                              bits=8, batch_size=2048, verbose=False)
    print(f"\n[hyper] BLKH Hyper: {res['recipe_size']:,}B  "
          f"(ratio {total_orig/res['recipe_size']:.2f}x)")
    print(f"  hyper (shared):       {res['hyper_size']:,}B  "
          f"-> {res['hyper_size']/N:.0f}B/image amortized")
    print(f"  latent per image:     {res['latent_per_image']}B (INT8)")
    print(f"  residual per img:     {res['residual_per_image']:.0f}B")
    print(f"  bit acc avg: {res['avg_bit_pct']:.1f}% "
          f"(min {res['min_bit_pct']:.1f}, max {res['max_bit_pct']:.1f})")
    print(f"  time: {res['total_time_s']:.1f}s "
          f"({res['per_image_time_s']:.2f}s/image)")

    # Verify
    t0 = time.time()
    recovered, dmeta = HyperCompressor.decompress(res['recipe_bytes'])
    print(f"\n[hyper] Decompress: {time.time()-t0:.1f}s  "
          f"all_sha256_match: {dmeta['all_sha256_match']}")
    print(f"  per-image SHA match: {sum(dmeta['sha256_per_image'])}/{dmeta['n_images']}")

    winner = "BLKH" if res['recipe_size'] < zip_total else "ZIP"
    print(f"\n[hyper] vs ZIP: {zip_total/res['recipe_size']:.3f}x  -> {winner}")


if __name__ == '__main__':
    _self_test()
