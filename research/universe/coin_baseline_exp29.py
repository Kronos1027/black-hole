"""
coin_baseline.py
=================
Clean-room reimplementation of the COIN baseline (Dupont et al. 2021).

Reference:
    Dupont, E., Golinski, A., Aliee, M., Teh, Y. W., Doucet, A.
    "Coin: Compression with Implicit Neural Representations".
    arXiv:2103.03123, DCC 2021.

This baseline trains one SIREN MLP per image (no clustering, no pruning, no
entropy coding). The MLP weights are quantized to `bits_per_weight` (default 16)
and saved as the compressed representation.

This is the comparison baseline for Experiment 29.

Usage:
    python3.13 coin_baseline.py --seed 42 --num-images 100
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_olivetti_faces  # noqa: F401  (ensures sklearn data works)


# ---------------------------------------------------------------------------
# SIREN reference implementation (single-omega, standard COIN configuration)
# ---------------------------------------------------------------------------

class _SirenLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_first: bool, omega: float):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega = omega
        self.is_first = is_first
        self._init_weights(in_features)

    def _init_weights(self, in_features: int):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_features
            else:
                import math
                bound = math.sqrt(6.0 / in_features) / self.omega
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class CoinMLP(nn.Module):
    """Standard COIN MLP: 1 input coord -> 5 hidden SIREN layers -> 1 output."""

    def __init__(self, hidden_features: int = 64, hidden_layers: int = 5, omega: float = 30.0):
        super().__init__()
        layers: List[nn.Module] = [_SirenLayer(2, hidden_features, is_first=True, omega=omega)]
        for _ in range(hidden_layers):
            layers.append(_SirenLayer(hidden_features, hidden_features, is_first=False, omega=omega))
        layers.append(nn.Linear(hidden_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_scikit_images(num_images: int = 100, size: int = 64) -> Tuple[np.ndarray, List[str]]:
    """
    Load `num_images` 64×64 grayscale images from scikit-image.

    The first 10 are the canonical images mentioned in the experiment brief:
    astronaut, camera, cell, coins, moon, page, text, clock, coffee, chelsea.
    The remaining 90 are sampled from scikit-image's standard set.
    """
    from skimage import data, color, transform

    canonical = [
        ('astronaut', data.astronaut),
        ('camera', data.camera),
        ('cell', data.cell),
        ('coins', data.coins),
        ('moon', data.moon),
        ('page', data.page),
        ('text', data.text),
        ('clock', data.clock),
        ('coffee', data.coffee),
        ('chelsea', data.chelsea),
    ]

    images: List[np.ndarray] = []
    names: List[str] = []

    for name, fetcher in canonical:
        try:
            img = fetcher()
        except Exception as e:
            print(f"[WARN] could not fetch {name}: {e}", flush=True)
            continue
        if img.ndim == 3:
            img = color.rgb2gray(img)
        img = transform.resize(img, (size, size), anti_aliasing=True, preserve_range=True)
        images.append(img.astype(np.float32))
        names.append(name)
        if len(images) >= num_images:
            return np.stack(images), names

    # Pad with rotated / flipped versions of canonical images to reach num_images.
    rng = np.random.default_rng(0)
    base = list(images)
    while len(images) < num_images:
        src = base[int(rng.integers(0, len(base)))]
        angle = float(rng.uniform(0, 360))
        rotated = transform.rotate(src, angle, resize=False, preserve_range=True)
        images.append(rotated.astype(np.float32))
        names.append(f"aug_{len(images)}")

    return np.stack(images[:num_images]), names[:num_images]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def normalize_to_pm1(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    return 2.0 * (img - lo) / (hi - lo) - 1.0


def denormalize_from_pm1(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi - lo < 1e-6:
        return np.full_like(arr, lo)
    return (arr + 1.0) * 0.5 * (hi - lo) + lo


def train_coin_one_image(img: np.ndarray, hidden_features: int = 64,
                          hidden_layers: int = 5, omega: float = 30.0,
                          epochs: int = 500, lr: float = 1e-3,
                          seed: int = 42) -> Tuple[CoinMLP, float, float]:
    """Train a single COIN MLP on one image. Returns (model, psnr_db, train_time_s)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    H, W = img.shape
    lo, hi = float(img.min()), float(img.max())
    target = normalize_to_pm1(img)

    # Coordinate grid in [-1, 1]
    ys, xs = np.meshgrid(np.linspace(-1, 1, H, dtype=np.float32),
                          np.linspace(-1, 1, W, dtype=np.float32), indexing='ij')
    coords = np.stack([xs.flatten(), ys.flatten()], axis=1)
    targets = target.flatten()

    coords_t = torch.tensor(coords)
    targets_t = torch.tensor(targets).unsqueeze(1)

    model = CoinMLP(hidden_features=hidden_features, hidden_layers=hidden_layers, omega=omega)
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # CONSTANT LR per Exp 23

    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(coords_t)
        loss = F.mse_loss(pred, targets_t)
        loss.backward()
        opt.step()
    train_time = time.time() - t0

    with torch.no_grad():
        pred = model(coords_t).cpu().numpy().flatten()
    pred_img = denormalize_from_pm1(pred, lo, hi).reshape(H, W)
    mse = float(np.mean((pred_img - img) ** 2))
    if mse < 1e-12:
        psnr = 99.0
    else:
        psnr = 10.0 * np.log10((float(img.max()) - float(img.min())) ** 2 / mse)
    return model, float(psnr), float(train_time)


# ---------------------------------------------------------------------------
# Weight serialization (no entropy coding, just float16)
# ---------------------------------------------------------------------------

def serialize_weights_float16(model: CoinMLP) -> bytes:
    """Quantize weights to float16 and pack as raw bytes (COIN baseline)."""
    out = bytearray()
    for p in model.parameters():
        w = p.detach().cpu().numpy().astype(np.float16)
        out += w.tobytes()
    return bytes(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--hidden-features', type=int, default=64)
    parser.add_argument('--hidden-layers', type=int, default=5)
    parser.add_argument('--omega', type=float, default=30.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bits-per-weight', type=int, default=16)
    parser.add_argument('--output-dir', type=str, default='/home/z/my-project/research/universe/_coin_baseline_out')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[coin_baseline] loading {args.num_images} images...", flush=True)
    images, names = load_scikit_images(args.num_images, args.size)
    print(f"[coin_baseline] loaded {len(images)} images ({images.dtype}, shape={images.shape})", flush=True)

    all_results = []
    for i in range(len(images)):
        img = images[i]
        name = names[i]
        model, psnr, t_train = train_coin_one_image(
            img, hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
            omega=args.omega, epochs=args.epochs, lr=args.lr, seed=args.seed,
        )
        weights_bytes = serialize_weights_float16(model)
        # float16 = 2 bytes per weight
        n_params = model.num_params()
        expected_bytes = n_params * 2
        all_results.append({
            'index': i,
            'name': name,
            'psnr_db': psnr,
            'train_time_s': t_train,
            'n_params': n_params,
            'weights_bytes_float16': len(weights_bytes),
            'expected_bytes_float16': expected_bytes,
        })
        if (i + 1) % 10 == 0:
            print(f"  [coin_baseline] {i+1}/{len(images)} done, last PSNR={psnr:.2f} dB, "
                  f"weights={len(weights_bytes)} B", flush=True)

    psnrs = np.array([r['psnr_db'] for r in all_results])
    sizes = np.array([r['weights_bytes_float16'] for r in all_results])
    summary = {
        'experiment': 'coin_baseline',
        'config': {
            'hidden_features': args.hidden_features,
            'hidden_layers': args.hidden_layers,
            'omega': args.omega,
            'epochs': args.epochs,
            'lr': args.lr,
            'bits_per_weight': args.bits_per_weight,
            'num_images': len(images),
            'image_size': args.size,
            'seed': args.seed,
        },
        'mean_psnr_db': float(psnrs.mean()),
        'std_psnr_db': float(psnrs.std()),
        'mean_weights_bytes': float(sizes.mean()),
        'std_weights_bytes': float(sizes.std()),
        'per_image': all_results,
    }

    out_json = os.path.join(args.output_dir, f'coin_baseline_seed{args.seed}.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    with open(out_json, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    summary['_output_json_sha256'] = sha

    print(f"\n[coin_baseline] DONE", flush=True)
    print(f"  mean PSNR = {psnrs.mean():.4f} dB (std {psnrs.std():.4f})", flush=True)
    print(f"  mean weights = {sizes.mean():.1f} B (std {sizes.std():.1f})", flush=True)
    print(f"  output JSON: {out_json}", flush=True)
    print(f"  JSON SHA-256: {sha}", flush=True)

    # Print final JSON to stdout (per DOCUMENTATION_PROTOCOL Rule 1)
    print("\n---JSON_BEGIN---")
    print(json.dumps(summary, indent=2, default=str))
    print("---JSON_END---")


if __name__ == '__main__':
    main()
