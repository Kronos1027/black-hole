"""
experiment_29_combined_pipeline.py
====================================
Experiment 29 — Validation of the Combined Pipeline Projection (BHUH).

Tests the projection recorded in BHUH_BREAKTHROUGH_RESULTS.md / SPECULATIVE.md:
    combining (a) multi-omega SIREN [10,50], (b) 5 hidden layers,
    (c) hierarchical KMeans K=50, (d) 500 epochs constant lr=1e-3,
    (e) arithmetic coding, (f) L1 pruning threshold=0.01
should yield ~34-35 dB PSNR and ~5.1x smaller weights than COIN.

ANTI-FABRICATION RULE (per DOCUMENTATION_PROTOCOL.md):
    - Hyperparameters are FIXED per the projection. They are NOT tuned to match
      the projected 34-35 dB. Whatever the experiment produces IS the result.
    - The script prints a JSON object to stdout at the end with all raw numbers.
    - SHA-256 of the final weights file and of the output JSON are computed and
      printed.
    - 3 seeds are run; mean ± std are reported.

Usage:
    python3.13 research/universe/experiment_29_combined_pipeline.py
    python3.13 research/universe/experiment_29_combined_pipeline.py --seeds 42,123,2024
    python3.13 research/universe/experiment_29_combined_pipeline.py --quick  # 5 imgs, 50 ep
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# Limit thread counts to reduce memory pressure in 4GB cgroup
torch.set_num_threads(2)
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

# Local imports (same directory)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arithmetic_codec import encode_weights, decode_weights
from coin_baseline_exp29 import load_scikit_images, normalize_to_pm1, denormalize_from_pm1, train_coin_one_image


# ---------------------------------------------------------------------------
# Multi-omega SIREN
# ---------------------------------------------------------------------------

class MultiOmegaSirenLayer(nn.Module):
    """
    SIREN layer that applies multiple omega values in parallel and concatenates
    the outputs. This is the multi-scale frequency representation validated in
    Experiment 24.
    """

    def __init__(self, in_features: int, out_features: int, omegas: List[float],
                 is_first: bool = False):
        super().__init__()
        self.omegas = list(omegas)
        self.is_first = is_first
        self.n_omega = len(self.omegas)
        # One linear per omega; outputs are concatenated then projected
        self.linears = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in self.omegas
        ])
        self.proj = nn.Linear(out_features * self.n_omega, out_features)
        self._init_weights(in_features)

    def _init_weights(self, in_features: int):
        with torch.no_grad():
            for i, lin in enumerate(self.linears):
                if self.is_first:
                    bound = 1.0 / in_features
                else:
                    bound = math.sqrt(6.0 / in_features) / self.omegas[i]
                lin.weight.uniform_(-bound, bound)

    def forward(self, x):
        outs = []
        for i, lin in enumerate(self.linears):
            outs.append(torch.sin(self.omegas[i] * lin(x)))
        cat = torch.cat(outs, dim=-1)
        return self.proj(cat)


class MultiOmegaSirenMLP(nn.Module):
    """
    SIREN MLP with multi-omega input layer and standard SIREN hidden layers.
    Architecture: 2 -> multi_omega_layer(h) -> h -> ... -> h -> 1
    """

    def __init__(self, hidden_features: int = 64, hidden_layers: int = 5,
                 omegas: List[float] = (10.0, 50.0)):
        super().__init__()
        self.omegas = list(omegas)
        # First layer: multi-omega
        layers: List[nn.Module] = [
            MultiOmegaSirenLayer(2, hidden_features, omegas=self.omegas, is_first=True)
        ]
        # Subsequent layers: standard SIREN with omega=30 (canonical)
        for _ in range(hidden_layers):
            layers.append(_StandardSirenLayer(hidden_features, hidden_features, omega=30.0))
        # Output: linear
        layers.append(nn.Linear(hidden_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_flat_weights(self) -> np.ndarray:
        parts = [p.detach().cpu().numpy().flatten() for p in self.parameters()]
        return np.concatenate(parts)


class _StandardSirenLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega = omega
        bound = math.sqrt(6.0 / in_features) / omega
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


# ---------------------------------------------------------------------------
# L1 pruning
# ---------------------------------------------------------------------------

def l1_prune(model: nn.Module, threshold: float = 0.01) -> Tuple[nn.Module, Dict]:
    """
    Apply L1 pruning: zero out weights whose absolute value is below `threshold`.
    Returns (model_with_masks, stats_dict).

    We do NOT actually remove the parameters (which would change the architecture);
    we apply a binary mask. The arithmetic codec then encodes only the non-zero
    values + a bitmask of which positions are non-zero.
    """
    n_before = 0
    n_zero = 0
    masks: List[np.ndarray] = []
    for p in model.parameters():
        w = p.detach().cpu().numpy()
        mask = (np.abs(w) >= threshold).astype(np.float32)
        n_before += w.size
        n_zero += int((mask == 0).sum())
        masks.append(mask)
        # Apply mask in-place
        with torch.no_grad():
            p.copy_(torch.from_numpy(w * mask))
    sparsity = n_zero / max(1, n_before)
    stats = {
        'n_params_total': n_before,
        'n_params_zeroed': n_zero,
        'n_params_kept': n_before - n_zero,
        'sparsity': float(sparsity),
        'threshold': threshold,
    }
    return model, stats


# ---------------------------------------------------------------------------
# Hierarchical KMeans clustering
# ---------------------------------------------------------------------------

def hierarchical_kmeans_cluster(weights_per_image: List[np.ndarray], K: int = 50,
                                  seed: int = 42) -> Dict:
    """
    Cluster weight values across all images into K=50 centroids. Each image's
    weights are then replaced by indices into the shared codebook.

    Returns dict with:
        codebook: np.ndarray (K,)
        indices_per_image: List[np.ndarray] of int32
        total_bytes_codebook: int
        total_bytes_indices: int  (raw, no entropy coding — that's added later)
    """
    # Concatenate all weights across images
    all_w = np.concatenate([w.flatten() for w in weights_per_image])
    # Subsample for KMeans fitting if too large
    rng = np.random.default_rng(seed)
    if all_w.size > 100_000:
        sample_idx = rng.choice(all_w.size, size=100_000, replace=False)
        sample = all_w[sample_idx].reshape(-1, 1)
    else:
        sample = all_w.reshape(-1, 1)

    km = KMeans(n_clusters=K, random_state=seed, n_init=4)
    km.fit(sample)
    codebook = km.cluster_centers_.flatten().astype(np.float32)

    # Assign each weight to nearest centroid
    indices_per_image = []
    for w in weights_per_image:
        flat = w.flatten()
        # Compute distances in chunks to avoid memory blowup
        chunk_size = 50_000
        idx = np.empty(flat.size, dtype=np.int32)
        for start in range(0, flat.size, chunk_size):
            end = min(start + chunk_size, flat.size)
            chunk = flat[start:end, None]  # (N, 1)
            dists = np.abs(chunk - codebook[None, :])  # (N, K)
            idx[start:end] = dists.argmin(axis=1).astype(np.int32)
        indices_per_image.append(idx)

    # Size accounting (raw, before arithmetic coding)
    total_bytes_codebook = codebook.nbytes  # K * 4 bytes (float32)
    total_bytes_indices = sum(idx.nbytes for idx in indices_per_image)  # 4 bytes per index (int32)

    return {
        'codebook': codebook,
        'indices_per_image': indices_per_image,
        'total_bytes_codebook': int(total_bytes_codebook),
        'total_bytes_indices_raw': int(total_bytes_indices),
        'K': K,
    }


def entropy_code_indices(cluster_result: Dict) -> Dict:
    """
    Apply arithmetic coding to the cluster indices. Returns bytes + size.
    """
    codebook = cluster_result['codebook']
    K = cluster_result['K']

    total_bytes = cluster_result['total_bytes_codebook']
    coded_sizes = []
    for idx in cluster_result['indices_per_image']:
        # Encode indices as symbols in [0, K-1]
        # Use arithmetic_codec.encode_weights with bits_per_weight = log2(K) rounded up
        bits = max(1, int(math.ceil(math.log2(K))))
        # Pack as uint8 if K<=256, else uint16
        if K <= 256:
            arr = idx.astype(np.uint8).astype(np.float32)
        else:
            arr = idx.astype(np.uint16).astype(np.float32)
        payload = encode_weights(arr, bits_per_weight=bits)
        coded_sizes.append(len(payload))
        total_bytes += len(payload)

    return {
        'total_bytes_codebook': cluster_result['total_bytes_codebook'],
        'coded_sizes_per_image': coded_sizes,
        'total_bytes_with_entropy_coding': int(total_bytes),
    }


# ---------------------------------------------------------------------------
# Training the combined pipeline
# ---------------------------------------------------------------------------

def train_combined_pipeline_one_image(img: np.ndarray, hidden_features: int,
                                        hidden_layers: int, omegas: List[float],
                                        epochs: int, lr: float, seed: int
                                        ) -> Tuple[MultiOmegaSirenMLP, float, float]:
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

    model = MultiOmegaSirenMLP(hidden_features=hidden_features,
                                hidden_layers=hidden_layers, omegas=omegas)
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # CONSTANT LR

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
        psnr = 10.0 * np.log10((hi - lo) ** 2 / mse)
    return model, float(psnr), float(train_time)


def evaluate_post_pruning_psnr(model: MultiOmegaSirenMLP, img: np.ndarray) -> float:
    """Re-evaluate PSNR after L1 pruning has been applied to the model."""
    H, W = img.shape
    lo, hi = float(img.min()), float(img.max())
    target = normalize_to_pm1(img)
    ys, xs = np.meshgrid(np.linspace(-1, 1, H, dtype=np.float32),
                          np.linspace(-1, 1, W, dtype=np.float32), indexing='ij')
    coords = np.stack([xs.flatten(), ys.flatten()], axis=1)
    coords_t = torch.tensor(coords)
    with torch.no_grad():
        pred = model(coords_t).cpu().numpy().flatten()
    pred_img = denormalize_from_pm1(pred, lo, hi).reshape(H, W)
    mse = float(np.mean((pred_img - img) ** 2))
    if mse < 1e-12:
        return 99.0
    return float(10.0 * np.log10((hi - lo) ** 2 / mse))


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def run_experiment(seeds: List[int], num_images: int, image_size: int,
                    hidden_features: int, hidden_layers_options: List[int],
                    omegas: List[float], epochs: int, lr: float,
                    pruning_threshold: float, K: int,
                    output_dir: str, quick: bool = False) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    import gc

    print(f"[exp29] loading {num_images} images at {image_size}x{image_size}...", flush=True)
    images, names = load_scikit_images(num_images, image_size)
    print(f"[exp29] loaded {len(images)} images, names[:5]={names[:5]}", flush=True)

    # ------------------------------------------------------------------
    # COIN baseline (computed once, seed=42)
    # ------------------------------------------------------------------
    coin_cache = os.path.join(output_dir, 'coin_baseline_cache.json')
    if os.path.exists(coin_cache):
        with open(coin_cache) as f:
            coin_data = json.load(f)
        coin_mean_psnr = coin_data['mean_psnr_db']
        coin_mean_size = coin_data['mean_weights_bytes_float16']
        print(f"[coin_baseline] CACHED mean PSNR={coin_mean_psnr:.4f} dB, mean size={coin_mean_size:.1f} B", flush=True)
    else:
        print(f"\n[exp29] === Running COIN baseline (seed=42) ===", flush=True)
        coin_psnrs = []
        coin_sizes = []
        for i, img in enumerate(images):
            model, psnr, _ = train_coin_one_image(
                img, hidden_features=hidden_features, hidden_layers=5,
                omega=30.0, epochs=epochs, lr=lr, seed=42,
            )
            from coin_baseline_exp29 import serialize_weights_float16
            w_bytes = serialize_weights_float16(model)
            coin_psnrs.append(psnr)
            coin_sizes.append(len(w_bytes))
            del model
            gc.collect()
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [coin] {i+1}/{len(images)}: PSNR={psnr:.2f} dB, weights={len(w_bytes)} B", flush=True)
        coin_mean_psnr = float(np.mean(coin_psnrs))
        coin_mean_size = float(np.mean(coin_sizes))
        print(f"[coin_baseline] mean PSNR={coin_mean_psnr:.4f} dB, mean size={coin_mean_size:.1f} B", flush=True)
        with open(coin_cache, 'w') as f:
            json.dump({
                'mean_psnr_db': coin_mean_psnr,
                'mean_weights_bytes_float16': coin_mean_size,
                'std_psnr_db': float(np.std(coin_psnrs)),
                'std_weights_bytes': float(np.std(coin_sizes)),
                'config': {'hidden_features': hidden_features, 'hidden_layers': 5,
                            'omega': 30.0, 'epochs': epochs, 'lr': lr, 'seed': 42},
            }, f, indent=2)

    # ------------------------------------------------------------------
    # Combined pipeline runs (with per-(hl, seed) checkpointing)
    # ------------------------------------------------------------------
    all_runs = []
    for hidden_layers in hidden_layers_options:
        for seed in seeds:
            ckpt_path = os.path.join(output_dir, f'ckpt_hl{hidden_layers}_seed{seed}.json')
            if os.path.exists(ckpt_path):
                print(f"\n[exp29] LOADING checkpoint hl={hidden_layers} seed={seed}", flush=True)
                with open(ckpt_path) as f:
                    run_summary = json.load(f)
                all_runs.append(run_summary)
                # Re-print summary
                print(f"  [comb hl={hidden_layers} seed={seed}] CACHED:", flush=True)
                print(f"    PSNR post-prune = {run_summary['mean_psnr_post_prune_db']:.4f} ± "
                      f"{run_summary['std_psnr_post_prune_db']:.4f} dB", flush=True)
                print(f"    Size per image  = {run_summary['mean_total_size_bytes']:.1f} ± "
                      f"{run_summary['std_total_size_bytes']:.1f} B", flush=True)
                print(f"    vs COIN: PSNR Δ = {run_summary['psnr_delta_vs_coin_db']:+.4f} dB, "
                      f"size reduction = {run_summary['size_reduction_vs_coin_x']:.4f}x", flush=True)
                continue

            print(f"\n[exp29] === Combined pipeline: hidden_layers={hidden_layers}, seed={seed} ===",
                  flush=True)
            run_results = {
                'hidden_layers': hidden_layers,
                'seed': seed,
                'per_image': [],
            }
            weights_per_image: List[np.ndarray] = []
            psnrs_pre_prune = []
            psnrs_post_prune = []
            train_times = []

            # Per-image checkpoint: load if exists
            img_ckpt = os.path.join(output_dir, f'imgs_hl{hidden_layers}_seed{seed}.json')
            if os.path.exists(img_ckpt):
                with open(img_ckpt) as f:
                    img_state = json.load(f)
                # We can't reconstruct weights_per_image from JSON, but we can skip already-done images
                # by re-running them — for true resumability we'd need to save .npy files.
                # For now, just re-run from scratch if the img checkpoint is incomplete.
                if len(img_state.get('per_image', [])) >= len(images):
                    # All images done — load summary from final ckpt
                    if os.path.exists(ckpt_path):
                        with open(ckpt_path) as f:
                            run_summary = json.load(f)
                        all_runs.append(run_summary)
                        print(f"  [comb hl={hidden_layers} seed={seed}] CACHED (all images done)", flush=True)
                        continue

            for i, img in enumerate(images):
                model, psnr_pre, t_train = train_combined_pipeline_one_image(
                    img, hidden_features=hidden_features, hidden_layers=hidden_layers,
                    omegas=omegas, epochs=epochs, lr=lr, seed=seed,
                )
                model_pruned, prune_stats = l1_prune(model, threshold=pruning_threshold)
                psnr_post = evaluate_post_pruning_psnr(model_pruned, img)

                weights = model_pruned.get_flat_weights()
                weights_per_image.append(weights.astype(np.float32))

                psnrs_pre_prune.append(psnr_pre)
                psnrs_post_prune.append(psnr_post)
                train_times.append(t_train)

                run_results['per_image'].append({
                    'index': i,
                    'name': names[i],
                    'psnr_pre_prune_db': psnr_pre,
                    'psnr_post_prune_db': psnr_post,
                    'train_time_s': t_train,
                    'n_params': model.num_params(),
                    'n_params_kept': prune_stats['n_params_kept'],
                    'sparsity': prune_stats['sparsity'],
                })
                del model, model_pruned
                gc.collect()
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [comb hl={hidden_layers} seed={seed}] {i+1}/{len(images)}: "
                          f"pre={psnr_pre:.2f} dB, post={psnr_post:.2f} dB, "
                          f"sparsity={prune_stats['sparsity']:.2%}", flush=True)

            print(f"  [comb hl={hidden_layers} seed={seed}] clustering K={K}...", flush=True)
            cluster_result = hierarchical_kmeans_cluster(weights_per_image, K=K, seed=seed)
            entropy_result = entropy_code_indices(cluster_result)

            del weights_per_image
            gc.collect()

            codebook_bytes = cluster_result['codebook'].astype(np.float32).tobytes()
            payload = bytearray()
            payload += codebook_bytes
            for idx in cluster_result['indices_per_image']:
                payload += idx.astype(np.int32).tobytes()
            payload_bytes = bytes(payload)
            weights_file = os.path.join(
                output_dir, f'exp29_weights_hl{hidden_layers}_seed{seed}.bin')
            with open(weights_file, 'wb') as f:
                f.write(payload_bytes)
            with open(weights_file, 'rb') as f:
                weights_sha = hashlib.sha256(f.read()).hexdigest()

            psnr_pre_arr = np.array(psnrs_pre_prune)
            psnr_post_arr = np.array(psnrs_post_prune)
            per_img_sizes = entropy_result['coded_sizes_per_image']
            codebook_share = entropy_result['total_bytes_codebook'] / len(per_img_sizes)
            total_per_img_size = [s + codebook_share for s in per_img_sizes]
            total_size_arr = np.array(total_per_img_size)

            run_summary = {
                'hidden_layers': hidden_layers,
                'seed': seed,
                'config': {
                    'hidden_features': hidden_features,
                    'omegas': omegas,
                    'epochs': epochs,
                    'lr': lr,
                    'pruning_threshold': pruning_threshold,
                    'K': K,
                    'num_images': len(images),
                    'image_size': image_size,
                },
                'mean_psnr_pre_prune_db': float(psnr_pre_arr.mean()),
                'std_psnr_pre_prune_db': float(psnr_pre_arr.std()),
                'mean_psnr_post_prune_db': float(psnr_post_arr.mean()),
                'std_psnr_post_prune_db': float(psnr_post_arr.std()),
                'mean_total_size_bytes': float(total_size_arr.mean()),
                'std_total_size_bytes': float(total_size_arr.std()),
                'mean_sparsity': float(np.mean([r['sparsity'] for r in run_results['per_image']])),
                'mean_train_time_s': float(np.mean(train_times)),
                'codebook_bytes': entropy_result['total_bytes_codebook'],
                'total_bytes_all_images': entropy_result['total_bytes_with_entropy_coding'],
                'weights_file': weights_file,
                'weights_sha256': weights_sha,
                'size_reduction_vs_coin_x': float(coin_mean_size / total_size_arr.mean()),
                'psnr_delta_vs_coin_db': float(psnr_post_arr.mean() - coin_mean_psnr),
                'per_image': run_results['per_image'],
            }
            # Save checkpoint
            with open(ckpt_path, 'w') as f:
                json.dump(run_summary, f, indent=2, default=str)
            all_runs.append(run_summary)

            print(f"  [comb hl={hidden_layers} seed={seed}] SUMMARY:", flush=True)
            print(f"    PSNR pre-prune  = {psnr_pre_arr.mean():.4f} ± {psnr_pre_arr.std():.4f} dB", flush=True)
            print(f"    PSNR post-prune = {psnr_post_arr.mean():.4f} ± {psnr_post_arr.std():.4f} dB", flush=True)
            print(f"    Size per image  = {total_size_arr.mean():.1f} ± {total_size_arr.std():.1f} B", flush=True)
            print(f"    Sparsity mean   = {run_summary['mean_sparsity']:.4f}", flush=True)
            print(f"    vs COIN: PSNR Δ = {run_summary['psnr_delta_vs_coin_db']:+.4f} dB, "
                  f"size reduction = {run_summary['size_reduction_vs_coin_x']:.4f}x", flush=True)
            print(f"    weights SHA-256 = {weights_sha}", flush=True)

            del cluster_result, entropy_result
            gc.collect()

    # Aggregate across seeds for each hidden_layers option
    aggregated = []
    for hidden_layers in hidden_layers_options:
        runs = [r for r in all_runs if r['hidden_layers'] == hidden_layers]
        psnrs = np.array([r['mean_psnr_post_prune_db'] for r in runs])
        sizes = np.array([r['mean_total_size_bytes'] for r in runs])
        reds = np.array([r['size_reduction_vs_coin_x'] for r in runs])
        agg = {
            'hidden_layers': hidden_layers,
            'n_seeds': len(runs),
            'seeds': [r['seed'] for r in runs],
            'mean_psnr_post_prune_db_across_seeds': float(psnrs.mean()),
            'std_psnr_post_prune_db_across_seeds': float(psnrs.std()),
            'mean_total_size_bytes_across_seeds': float(sizes.mean()),
            'std_total_size_bytes_across_seeds': float(sizes.std()),
            'mean_size_reduction_vs_coin_x': float(reds.mean()),
            'std_size_reduction_vs_coin_x': float(reds.std()),
        }
        aggregated.append(agg)
        print(f"\n[exp29] AGGREGATED hidden_layers={hidden_layers}:", flush=True)
        print(f"  PSNR  = {agg['mean_psnr_post_prune_db_across_seeds']:.4f} ± "
              f"{agg['std_psnr_post_prune_db_across_seeds']:.4f} dB", flush=True)
        print(f"  Size  = {agg['mean_total_size_bytes_across_seeds']:.1f} ± "
              f"{agg['std_total_size_bytes_across_seeds']:.1f} B", flush=True)
        print(f"  Reduction vs COIN = {agg['mean_size_reduction_vs_coin_x']:.4f}x ± "
              f"{agg['std_size_reduction_vs_coin_x']:.4f}", flush=True)

    # Final output JSON
    output = {
        'experiment': 'experiment_29_combined_pipeline',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'projection_target': {
            'psnr_db': 34.0,
            'size_reduction_vs_coin_x': 5.1,
            'source': 'BHUH_BREAKTHROUGH_RESULTS.md',
        },
        'coin_baseline': {
            'mean_psnr_db': coin_mean_psnr,
            'mean_weights_bytes_float16': coin_mean_size,
            'config': {
                'hidden_features': hidden_features,
                'hidden_layers': 5,
                'omega': 30.0,
                'epochs': epochs,
                'lr': lr,
            },
        },
        'combined_pipeline_runs': all_runs,
        'aggregated': aggregated,
    }

    out_json_path = os.path.join(output_dir, 'experiment_29_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha

    print(f"\n[exp29] DONE", flush=True)
    print(f"  output JSON: {out_json_path}", flush=True)
    print(f"  JSON SHA-256: {json_sha}", flush=True)

    print("\n---JSON_BEGIN---")
    print(json.dumps(output, indent=2, default=str))
    print("---JSON_END---")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='42,123,2024')
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--hidden-features', type=int, default=64)
    parser.add_argument('--hidden-layers', type=str, default='2,5',
                        help='Comma-separated list of hidden layer counts to test')
    parser.add_argument('--omegas', type=str, default='10,50')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pruning-threshold', type=float, default=0.01)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp29_out')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test: 5 images, 50 epochs')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    hidden_layers_opts = [int(h) for h in args.hidden_layers.split(',')]
    omegas = [float(o) for o in args.omegas.split(',')]

    if args.quick:
        args.num_images = 5
        args.epochs = 50
        seeds = [42]
        hidden_layers_opts = [2]

    run_experiment(
        seeds=seeds,
        num_images=args.num_images,
        image_size=args.size,
        hidden_features=args.hidden_features,
        hidden_layers_options=hidden_layers_opts,
        omegas=omegas,
        epochs=args.epochs,
        lr=args.lr,
        pruning_threshold=args.pruning_threshold,
        K=args.K,
        output_dir=args.output_dir,
        quick=args.quick,
    )


if __name__ == '__main__':
    main()
