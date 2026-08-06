"""
experiment_37_real_photo_parity.py
====================================
Experiment 37 — Real Photo Byte Parity Test (the publishable test).

The entropy coding pipeline (single-omega SIREN ω=30 + KMeans K=50 +
arithmetic coding) was validated as the only surviving BHUH component
in Exp 34-36. But all those tests used 64×64 synthetic scikit-image
crops. This experiment tests on REAL 256×256 photographs (same dataset
as HONEST_BENCHMARK_RESULTS.md) against production codecs:

  (a) SIREN + entropy coding (the validated BHUH pipeline)
  (b) JPEG at matched byte budget
  (c) WebP at matched byte budget
  (d) COIN raw (float16, no entropy coding) — baseline

The key metric: PSNR at the SAME byte budget for all four methods.
This is the parity comparison that has been missing since the start.

If SIREN+entropy loses to JPEG/WebP on real photos (a real possibility
given HONEST_BENCHMARK_RESULTS.md showed BLKH losing to COIN on real
photos), that is a valid negative result — it means the entropy coding
pipeline is NOT competitive with production codecs on natural photography,
and the program ends with a purely negative conclusion.

ANTI-FABRICATION: same protocol as Exp 29-36.
3 seeds, output real, SHA-256, no tuning to match expectations.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans

torch.set_num_threads(2)
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coin_baseline_exp29 import (
    normalize_to_pm1, denormalize_from_pm1, CoinMLP,
)
from experiment_29_combined_pipeline import (
    hierarchical_kmeans_cluster, entropy_code_indices,
)


# ---------------------------------------------------------------------------
# Dataset: 3 real photos at 256×256 (same as HONEST_BENCHMARK_RESULTS.md)
# ---------------------------------------------------------------------------

def load_real_photos(size: int = 256) -> Tuple[np.ndarray, List[str]]:
    """Load 3 real scikit-image photographs at 256×256 grayscale."""
    from skimage import data, color, transform

    photos = [
        ('astronaut', data.astronaut),
        ('camera', data.camera),
        ('cell', data.cell),
    ]

    images: List[np.ndarray] = []
    names: List[str] = []
    for name, fetcher in photos:
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

    return np.stack(images), names


# ---------------------------------------------------------------------------
# SIREN + entropy coding (the validated pipeline)
# ---------------------------------------------------------------------------

def train_siren_one_image(img: np.ndarray, hidden_features: int,
                            hidden_layers: int, omega: float, epochs: int,
                            lr: float, seed: int) -> Tuple[CoinMLP, float, float]:
    """Train a single-omega SIREN on one image. Returns (model, psnr, time)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    H, W = img.shape
    lo, hi = float(img.min()), float(img.max())
    target = normalize_to_pm1(img)

    ys, xs = np.meshgrid(np.linspace(-1, 1, H, dtype=np.float32),
                          np.linspace(-1, 1, W, dtype=np.float32), indexing='ij')
    coords = np.stack([xs.flatten(), ys.flatten()], axis=1)
    targets = target.flatten()

    coords_t = torch.tensor(coords)
    targets_t = torch.tensor(targets).unsqueeze(1)

    model = CoinMLP(hidden_features=hidden_features, hidden_layers=hidden_layers, omega=omega)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(coords_t)
        loss = F.mse_loss(pred, targets_t)
        loss.backward()
        opt.step()
    train_time = time.time() - t0

    # Evaluate PSNR
    with torch.no_grad():
        pred = model(coords_t).cpu().numpy().flatten()
    pred_img = denormalize_from_pm1(pred, lo, hi).reshape(H, W)
    mse = float(np.mean((pred_img - img) ** 2))
    if mse < 1e-12:
        psnr = 99.0
    else:
        psnr = float(10.0 * np.log10((hi - lo) ** 2 / mse))

    return model, psnr, train_time


def get_model_weights_flat(model: nn.Module) -> np.ndarray:
    parts = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
    return np.concatenate(parts)


def entropy_code_weights(models: List[CoinMLP], K: int, seed: int) -> Tuple[List[int], int]:
    """Apply KMeans K=50 + arithmetic coding to model weights."""
    weights_per_image = [get_model_weights_flat(m).astype(np.float32) for m in models]
    cluster_result = hierarchical_kmeans_cluster(weights_per_image, K=K, seed=seed)
    entropy_result = entropy_code_indices(cluster_result)
    per_img_sizes = entropy_result['coded_sizes_per_image']
    codebook_bytes = entropy_result['total_bytes_codebook']
    return per_img_sizes, codebook_bytes


# ---------------------------------------------------------------------------
# JPEG / WebP at matched byte budget
# ---------------------------------------------------------------------------

def jpeg_compress_at_size(img: np.ndarray, target_size: int) -> Tuple[bytes, int]:
    """Compress image as JPEG, adjusting quality to hit target_size (±10%)."""
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')

    best_payload = None
    best_size = None
    best_quality = None

    # Binary search for quality that hits target size
    lo_q, hi_q = 1, 95
    for _ in range(20):
        mid_q = (lo_q + hi_q) // 2
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=mid_q)
        size = buf.tell()
        if abs(size - target_size) < target_size * 0.1:
            return buf.getvalue(), size
        if size < target_size:
            lo_q = mid_q + 1
        else:
            hi_q = mid_q - 1
        best_payload = buf.getvalue()
        best_size = size
        best_quality = mid_q
        if lo_q > hi_q:
            break

    return best_payload, best_size


def webp_compress_at_size(img: np.ndarray, target_size: int) -> Tuple[bytes, int]:
    """Compress image as WebP, adjusting quality to hit target_size (±10%)."""
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')

    best_payload = None
    best_size = None

    lo_q, hi_q = 1, 95
    for _ in range(20):
        mid_q = (lo_q + hi_q) // 2
        buf = io.BytesIO()
        pil_img.save(buf, format='WebP', quality=mid_q)
        size = buf.tell()
        if abs(size - target_size) < target_size * 0.1:
            return buf.getvalue(), size
        if size < target_size:
            lo_q = mid_q + 1
        else:
            hi_q = mid_q - 1
        best_payload = buf.getvalue()
        best_size = size
        if lo_q > hi_q:
            break

    return best_payload, best_size


def compute_psnr(img: np.ndarray, recon: np.ndarray) -> float:
    """Compute PSNR between original and reconstruction."""
    mse = float(np.mean((img - recon) ** 2))
    if mse < 1e-12:
        return 99.0
    lo, hi = float(img.min()), float(img.max())
    return float(10.0 * np.log10((hi - lo) ** 2 / mse))


def jpeg_decompress(payload: bytes, shape: Tuple[int, int]) -> np.ndarray:
    """Decompress JPEG and return as float32 array (grayscale)."""
    pil_img = Image.open(io.BytesIO(payload))
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    arr = np.array(pil_img, dtype=np.float32)
    return arr


def webp_decompress(payload: bytes, shape: Tuple[int, int]) -> np.ndarray:
    """Decompress WebP and return as float32 array (grayscale)."""
    pil_img = Image.open(io.BytesIO(payload))
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    arr = np.array(pil_img, dtype=np.float32)
    return arr


# ---------------------------------------------------------------------------
# COIN raw (float16, no entropy coding)
# ---------------------------------------------------------------------------

def coin_raw_size(model: CoinMLP) -> int:
    """Size of COIN model weights as float16."""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 2  # float16 = 2 bytes per weight


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(seeds: List[int], image_size: int, hidden_features: int,
                    hidden_layers: int, omega: float, epochs: int, lr: float,
                    K: int, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[exp37] loading real photos at {image_size}x{image_size}...", flush=True)
    images, names = load_real_photos(image_size)
    print(f"[exp37] loaded {len(images)} real photos: {names}", flush=True)

    all_runs = []
    for seed in seeds:
        ckpt_path = os.path.join(output_dir, f'ckpt_seed{seed}.json')
        if os.path.exists(ckpt_path):
            print(f"\n[exp37] LOADING checkpoint seed={seed}", flush=True)
            with open(ckpt_path) as f:
                run_result = json.load(f)
            all_runs.append(run_result)
            print(f"  CACHED", flush=True)
            continue

        print(f"\n[exp37] === seed={seed} ===", flush=True)
        t0 = time.time()

        # 1. Train SIREN on each image
        models = []
        siren_psnrs = []
        siren_train_times = []
        for i, img in enumerate(images):
            model, psnr, t_train = train_siren_one_image(
                img, hidden_features=hidden_features, hidden_layers=hidden_layers,
                omega=omega, epochs=epochs, lr=lr, seed=seed,
            )
            models.append(model)
            siren_psnrs.append(psnr)
            siren_train_times.append(t_train)
            print(f"  SIREN {names[i]}: PSNR={psnr:.2f} dB, t={t_train:.1f}s", flush=True)

        # 2. Entropy code the weights (shared codebook across 3 images)
        per_img_sizes, codebook_bytes = entropy_code_weights(models, K=K, seed=seed)
        codebook_share = codebook_bytes / len(per_img_sizes)
        siren_sizes = [s + codebook_share for s in per_img_sizes]

        # 3. COIN raw sizes (float16)
        coin_raw_sizes = [coin_raw_size(m) for m in models]

        # 4. JPEG and WebP at matched byte budget (use SIREN+entropy size as target)
        jpeg_psnrs = []
        jpeg_sizes = []
        webp_psnrs = []
        webp_sizes = []

        for i, img in enumerate(images):
            target_size = int(siren_sizes[i])

            # JPEG at matched size
            jpeg_payload, jpeg_actual_size = jpeg_compress_at_size(img, target_size)
            jpeg_recon = jpeg_decompress(jpeg_payload, img.shape)
            jpeg_psnr = compute_psnr(img, jpeg_recon)
            jpeg_psnrs.append(jpeg_psnr)
            jpeg_sizes.append(jpeg_actual_size)

            # WebP at matched size
            webp_payload, webp_actual_size = webp_compress_at_size(img, target_size)
            webp_recon = webp_decompress(webp_payload, img.shape)
            webp_psnr = compute_psnr(img, webp_recon)
            webp_psnrs.append(webp_psnr)
            webp_sizes.append(webp_actual_size)

            print(f"  {names[i]} (target {target_size:.0f} B):", flush=True)
            print(f"    SIREN+entropy: {siren_sizes[i]:.0f} B, {siren_psnrs[i]:.2f} dB", flush=True)
            print(f"    JPEG:          {jpeg_actual_size} B, {jpeg_psnr:.2f} dB", flush=True)
            print(f"    WebP:          {webp_actual_size} B, {webp_psnr:.2f} dB", flush=True)
            print(f"    COIN raw:      {coin_raw_sizes[i]} B, {siren_psnrs[i]:.2f} dB (same PSNR, no entropy)", flush=True)

        # Save weights file for SHA-256
        weights_payload = bytearray()
        for m in models:
            w = get_model_weights_flat(m).astype(np.float32)
            weights_payload += w.tobytes()
        weights_file = os.path.join(output_dir, f'exp37_weights_seed{seed}.bin')
        with open(weights_file, 'wb') as f:
            f.write(bytes(weights_payload))
        with open(weights_file, 'rb') as f:
            weights_sha = hashlib.sha256(f.read()).hexdigest()

        run_result = {
            'seed': seed,
            'per_image': [
                {
                    'name': names[i],
                    'siren_entropy_psnr_db': siren_psnrs[i],
                    'siren_entropy_size_bytes': siren_sizes[i],
                    'jpeg_psnr_db': jpeg_psnrs[i],
                    'jpeg_size_bytes': jpeg_sizes[i],
                    'webp_psnr_db': webp_psnrs[i],
                    'webp_size_bytes': webp_sizes[i],
                    'coin_raw_psnr_db': siren_psnrs[i],  # same model, no entropy
                    'coin_raw_size_bytes': coin_raw_sizes[i],
                }
                for i in range(len(images))
            ],
            'mean_siren_entropy_psnr_db': float(np.mean(siren_psnrs)),
            'mean_siren_entropy_size_bytes': float(np.mean(siren_sizes)),
            'mean_jpeg_psnr_db': float(np.mean(jpeg_psnrs)),
            'mean_jpeg_size_bytes': float(np.mean(jpeg_sizes)),
            'mean_webp_psnr_db': float(np.mean(webp_psnrs)),
            'mean_webp_size_bytes': float(np.mean(webp_sizes)),
            'mean_coin_raw_psnr_db': float(np.mean(siren_psnrs)),
            'mean_coin_raw_size_bytes': float(np.mean(coin_raw_sizes)),
            'total_time_s': time.time() - t0,
            'weights_file': weights_file,
            'weights_sha256': weights_sha,
        }

        with open(ckpt_path, 'w') as f:
            json.dump(run_result, f, indent=2, default=str)
        all_runs.append(run_result)

        print(f"\n  MEAN (seed={seed}):", flush=True)
        print(f"    SIREN+entropy: {run_result['mean_siren_entropy_psnr_db']:.2f} dB, {run_result['mean_siren_entropy_size_bytes']:.0f} B", flush=True)
        print(f"    JPEG:          {run_result['mean_jpeg_psnr_db']:.2f} dB, {run_result['mean_jpeg_size_bytes']:.0f} B", flush=True)
        print(f"    WebP:          {run_result['mean_webp_psnr_db']:.2f} dB, {run_result['mean_webp_size_bytes']:.0f} B", flush=True)
        print(f"    COIN raw:      {run_result['mean_coin_raw_psnr_db']:.2f} dB, {run_result['mean_coin_raw_size_bytes']:.0f} B", flush=True)

    # Aggregate across seeds
    siren_psnrs = np.array([r['mean_siren_entropy_psnr_db'] for r in all_runs])
    siren_sizes = np.array([r['mean_siren_entropy_size_bytes'] for r in all_runs])
    jpeg_psnrs = np.array([r['mean_jpeg_psnr_db'] for r in all_runs])
    jpeg_sizes = np.array([r['mean_jpeg_size_bytes'] for r in all_runs])
    webp_psnrs = np.array([r['mean_webp_psnr_db'] for r in all_runs])
    webp_sizes = np.array([r['mean_webp_size_bytes'] for r in all_runs])
    coin_psnrs = np.array([r['mean_coin_raw_psnr_db'] for r in all_runs])
    coin_sizes = np.array([r['mean_coin_raw_size_bytes'] for r in all_runs])

    aggregated = {
        'siren_entropy': {
            'mean_psnr_db': float(siren_psnrs.mean()), 'std_psnr_db': float(siren_psnrs.std()),
            'mean_size_bytes': float(siren_sizes.mean()), 'std_size_bytes': float(siren_sizes.std()),
        },
        'jpeg': {
            'mean_psnr_db': float(jpeg_psnrs.mean()), 'std_psnr_db': float(jpeg_psnrs.std()),
            'mean_size_bytes': float(jpeg_sizes.mean()), 'std_size_bytes': float(jpeg_sizes.std()),
        },
        'webp': {
            'mean_psnr_db': float(webp_psnrs.mean()), 'std_psnr_db': float(webp_psnrs.std()),
            'mean_size_bytes': float(webp_sizes.mean()), 'std_size_bytes': float(webp_sizes.std()),
        },
        'coin_raw': {
            'mean_psnr_db': float(coin_psnrs.mean()), 'std_psnr_db': float(coin_psnrs.std()),
            'mean_size_bytes': float(coin_sizes.mean()), 'std_size_bytes': float(coin_sizes.std()),
        },
    }

    print(f"\n[exp37] AGGREGATED across {len(all_runs)} seeds:", flush=True)
    for name, agg in aggregated.items():
        print(f"  {name}: PSNR={agg['mean_psnr_db']:.2f}±{agg['std_psnr_db']:.2f} dB, "
              f"size={agg['mean_size_bytes']:.0f}±{agg['std_size_bytes']:.0f} B", flush=True)

    # Determine winner at matched byte budget
    siren_psnr = aggregated['siren_entropy']['mean_psnr_db']
    jpeg_psnr = aggregated['jpeg']['mean_psnr_db']
    webp_psnr = aggregated['webp']['mean_psnr_db']

    psnr_diff_jpeg_minus_siren = jpeg_psnr - siren_psnr
    psnr_diff_webp_minus_siren = webp_psnr - siren_psnr

    if siren_psnr > jpeg_psnr and siren_psnr > webp_psnr:
        winner = "SIREN+entropy (beats JPEG and WebP)"
        conclusion = "POSITIVE — entropy coding pipeline is competitive with production codecs on real photos"
    elif siren_psnr > jpeg_psnr or siren_psnr > webp_psnr:
        winner = "mixed (SIREN beats one but not the other)"
        conclusion = "MIXED — entropy coding is competitive with one codec but not the other"
    else:
        winner = "JPEG/WebP (both beat SIREN+entropy)"
        conclusion = "NEGATIVE — entropy coding pipeline is NOT competitive with production codecs on real photos"

    comparison = {
        'psnr_diff_jpeg_minus_siren': float(psnr_diff_jpeg_minus_siren),
        'psnr_diff_webp_minus_siren': float(psnr_diff_webp_minus_siren),
        'winner': winner,
        'conclusion': conclusion,
    }

    print(f"\n[exp37] PARITY COMPARISON:", flush=True)
    print(f"  PSNR diff (JPEG - SIREN): {psnr_diff_jpeg_minus_siren:+.2f} dB", flush=True)
    print(f"  PSNR diff (WebP - SIREN): {psnr_diff_webp_minus_siren:+.2f} dB", flush=True)
    print(f"  WINNER: {winner}", flush=True)
    print(f"  CONCLUSION: {conclusion}", flush=True)

    output = {
        'experiment': 'experiment_37_real_photo_parity',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'SIREN + entropy coding (validated in Exp 34-36) is competitive '
                       'with JPEG and WebP on real 256×256 photographs at matched byte budget.',
        'config': {
            'image_size': image_size,
            'hidden_features': hidden_features,
            'hidden_layers': hidden_layers,
            'omega': omega,
            'epochs': epochs,
            'lr': lr,
            'K': K,
            'seeds': seeds,
            'num_images': len(images),
            'dataset': '3 real scikit-image photos (astronaut, camera, cell) at 256×256 grayscale',
        },
        'all_runs': all_runs,
        'aggregated': aggregated,
        'parity_comparison': comparison,
    }

    out_json_path = os.path.join(output_dir, 'experiment_37_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha

    print(f"\n[exp37] DONE", flush=True)
    print(f"  JSON SHA-256: {json_sha}", flush=True)

    print("\n---JSON_BEGIN---")
    print(json.dumps(output, indent=2, default=str))
    print("---JSON_END---")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='42,123,2024')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--hidden-features', type=int, default=64)
    parser.add_argument('--hidden-layers', type=int, default=2)
    parser.add_argument('--omega', type=float, default=30.0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp37_out')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    run_experiment(
        seeds=seeds, image_size=args.size,
        hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
        omega=args.omega, epochs=args.epochs, lr=args.lr, K=args.K,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
