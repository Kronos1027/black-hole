"""
experiment_36_hierarchical_isolated.py
=======================================
Experiment 36 — Isolate Hierarchical Sharing (the original BHUH claim).

The original BHUH_BREAKTHROUGH_RESULTS.md claims:
  COIN (separate SIRENs): 28.10 dB, ~86000 B
  BHUH Hierarchical K=50: 31.21 dB, 56983 B  (+3.11 dB, 1.5x smaller)

But that claim combines THREE mechanisms:
  1. Hierarchical K=50 sharing (backbone shared across clustered images)
  2. Multi-omega [10,50] architecture (REFUTED in Exp 35)
  3. Entropy coding (KMeans + arithmetic, validated in Exp 34)

This experiment ISOLATES mechanism #1 (hierarchical sharing) by:
  - Using SINGLE omega=30 for both configs (no multi-omega)
  - Using SAME entropy coding (KMeans K=50 + arithmetic) for both configs
  - The ONLY difference is: (a) per-image SIREN vs (b) shared backbone

If BHUH hierarchical still wins, the sharing mechanism has real value.
If COIN wins again, the original "breakthrough" was entirely an artifact
of combining sharing with multi-omega and uncontrolled entropy coding.

ANTI-FABRICATION: same protocol as Exp 29-35.
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

torch.set_num_threads(2)
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coin_baseline_exp29 import (
    load_scikit_images, normalize_to_pm1, denormalize_from_pm1,
    train_coin_one_image, CoinMLP, serialize_weights_float16,
)
from experiment_29_combined_pipeline import (
    hierarchical_kmeans_cluster, entropy_code_indices,
)
from experiment_32_coin_rd_curve import (
    get_model_weights_flat, set_model_weights_flat, evaluate_model_psnr,
    quantize_weights_minmax, compute_size_bytes,
)


def get_model_weights_flat(model: nn.Module) -> np.ndarray:
    parts = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
    return np.concatenate(parts)


def train_per_image_siren(images: np.ndarray, hidden_features: int,
                            hidden_layers: int, omega: float, epochs: int,
                            lr: float, seed: int) -> List[Tuple[CoinMLP, float, float]]:
    """Train a separate SIREN for each image (COIN approach)."""
    torch.manual_seed(seed)
    models = []
    for img in images:
        model, psnr, t_train = train_coin_one_image(
            img, hidden_features=hidden_features, hidden_layers=hidden_layers,
            omega=omega, epochs=epochs, lr=lr, seed=seed,
        )
        models.append((model, psnr, t_train))
    return models


def train_shared_backbone_siren(images: np.ndarray, hidden_features: int,
                                  hidden_layers: int, omega: float, epochs: int,
                                  lr: float, seed: int, K: int) -> List[Tuple[CoinMLP, float, float]]:
    """
    Train a shared backbone SIREN approach (BHUH hierarchical).
    
    Approach:
    1. Cluster images into K=50 groups by image statistics (mean, std, etc.)
    2. For each cluster, train ONE SIREN on all images in that cluster
       (multi-image training: random sample from cluster each epoch)
    3. For each image, use its cluster's SIREN for reconstruction
    
    This is the "hierarchical sharing" mechanism: images in the same cluster
    share a backbone, reducing per-image cost.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Extract image features for clustering
    features = []
    for img in images:
        features.append([
            float(img.mean()), float(img.std()),
            float(img.min()), float(img.max()),
            float(np.percentile(img, 25)), float(np.percentile(img, 75)),
        ])
    features = np.array(features)
    
    # Cluster images into K groups
    n_clusters = min(K, len(images))
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=4)
    cluster_labels = kmeans.fit_predict(features)
    
    # Train one SIREN per cluster (shared backbone)
    cluster_models: Dict[int, CoinMLP] = {}
    cluster_psnrs: Dict[int, float] = {}
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        
        # Train on all images in this cluster (multi-image)
        cluster_images = [images[i] for i in cluster_indices]
        
        # Build training data: coordinates + targets from all images in cluster
        all_coords = []
        all_targets = []
        for img in cluster_images:
            H, W = img.shape
            lo, hi = float(img.min()), float(img.max())
            target = normalize_to_pm1(img)
            ys, xs = np.meshgrid(np.linspace(-1, 1, H, dtype=np.float32),
                                  np.linspace(-1, 1, W, dtype=np.float32), indexing='ij')
            coords = np.stack([xs.flatten(), ys.flatten()], axis=1)
            all_coords.append(coords)
            all_targets.append(target.flatten())
        
        all_coords = np.concatenate(all_coords, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Subsample if too large
        if len(all_coords) > 50000:
            idx = np.random.choice(len(all_coords), 50000, replace=False)
            all_coords = all_coords[idx]
            all_targets = all_targets[idx]
        
        coords_t = torch.tensor(all_coords)
        targets_t = torch.tensor(all_targets).unsqueeze(1)
        
        # Train SIREN for this cluster
        model = CoinMLP(hidden_features=hidden_features, hidden_layers=hidden_layers, omega=omega)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(coords_t)
            loss = F.mse_loss(pred, targets_t)
            loss.backward()
            opt.step()
        
        # Evaluate PSNR on each image in cluster
        cluster_models[cluster_id] = model
        # Will compute per-image PSNR below
    
    # Assign each image to its cluster's model and compute PSNR
    results = []
    for i, img in enumerate(images):
        cluster_id = int(cluster_labels[i])
        model = cluster_models[cluster_id]
        psnr = evaluate_model_psnr(model, img)
        t_train = 0.0  # training time tracked separately if needed
        results.append((model, psnr, t_train))
    
    return results


def compute_entropy_coded_size(models: List[CoinMLP], K: int, seed: int) -> Tuple[List[int], int]:
    """
    Apply KMeans K=50 + arithmetic coding to the weights of all models.
    Returns (per_image_sizes, codebook_bytes).
    """
    weights_per_image = [get_model_weights_flat(m).astype(np.float32) for m, _, _ in models]
    cluster_result = hierarchical_kmeans_cluster(weights_per_image, K=K, seed=seed)
    entropy_result = entropy_code_indices(cluster_result)
    per_img_sizes = entropy_result['coded_sizes_per_image']
    codebook_bytes = entropy_result['total_bytes_codebook']
    return per_img_sizes, codebook_bytes


def run_config(config_name: str, train_fn, images: np.ndarray, hidden_features: int,
               hidden_layers: int, omega: float, epochs: int, lr: float,
               K: int, seed: int, output_dir: str) -> Dict:
    """Run a single config and return results dict."""
    import gc
    t0 = time.time()
    
    # Train
    if config_name == 'coin_per_image':
        models = train_per_image_siren(images, hidden_features, hidden_layers, omega, epochs, lr, seed)
    elif config_name == 'bhuh_hierarchical':
        models = train_shared_backbone_siren(images, hidden_features, hidden_layers, omega, epochs, lr, seed, K)
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    train_time = time.time() - t0
    
    # Evaluate PSNRs
    psnrs = [psnr for _, psnr, _ in models]
    
    # Compute entropy-coded size
    per_img_sizes, codebook_bytes = compute_entropy_coded_size(models, K, seed)
    codebook_share = codebook_bytes / len(per_img_sizes)
    total_per_img_size = [s + codebook_share for s in per_img_sizes]
    
    # Save weights for SHA-256
    weights_payload = bytearray()
    for m, _, _ in models:
        w = get_model_weights_flat(m).astype(np.float32)
        weights_payload += w.tobytes()
    weights_file = os.path.join(output_dir, f'exp36_{config_name}_seed{seed}.bin')
    with open(weights_file, 'wb') as f:
        f.write(bytes(weights_payload))
    with open(weights_file, 'rb') as f:
        weights_sha = hashlib.sha256(f.read()).hexdigest()
    
    # Cleanup
    del models
    gc.collect()
    
    psnr_arr = np.array(psnrs)
    size_arr = np.array(total_per_img_size)
    
    return {
        'config_name': config_name,
        'seed': seed,
        'omega': omega,
        'hidden_layers': hidden_layers,
        'mean_psnr_db': float(psnr_arr.mean()),
        'std_psnr_db': float(psnr_arr.std()),
        'mean_total_size_bytes': float(size_arr.mean()),
        'std_total_size_bytes': float(size_arr.std()),
        'train_time_s': train_time,
        'codebook_bytes': codebook_bytes,
        'weights_file': weights_file,
        'weights_sha256': weights_sha,
    }


def run_experiment(seeds: List[int], num_images: int, image_size: int,
                    hidden_features: int, hidden_layers: int, omega: float,
                    epochs: int, lr: float, K: int, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[exp36] loading {num_images} images at {image_size}x{image_size}...", flush=True)
    images, names = load_scikit_images(num_images, image_size)
    print(f"[exp36] loaded {len(images)} images", flush=True)
    print(f"[exp36] SINGLE omega={omega} for both configs (no multi-omega)", flush=True)
    print(f"[exp36] SAME entropy coding (KMeans K={K} + arithmetic) for both", flush=True)
    print(f"[exp36] Only difference: per-image SIREN vs shared backbone SIREN", flush=True)

    configs = ['coin_per_image', 'bhuh_hierarchical']
    all_runs = []
    
    for config_name in configs:
        for seed in seeds:
            ckpt_path = os.path.join(output_dir, f'ckpt_{config_name}_seed{seed}.json')
            if os.path.exists(ckpt_path):
                print(f"\n[exp36] LOADING {config_name} seed={seed}", flush=True)
                with open(ckpt_path) as f:
                    run_result = json.load(f)
                all_runs.append(run_result)
                print(f"  CACHED: PSNR={run_result['mean_psnr_db']:.4f}, "
                      f"size={run_result['mean_total_size_bytes']:.1f}", flush=True)
                continue

            print(f"\n[exp36] === {config_name}, seed={seed} ===", flush=True)
            run_result = run_config(
                config_name=config_name, train_fn=None,
                images=images, hidden_features=hidden_features,
                hidden_layers=hidden_layers, omega=omega, epochs=epochs,
                lr=lr, K=K, seed=seed, output_dir=output_dir,
            )
            with open(ckpt_path, 'w') as f:
                json.dump(run_result, f, indent=2, default=str)
            all_runs.append(run_result)

            print(f"  PSNR  = {run_result['mean_psnr_db']:.4f} ± {run_result['std_psnr_db']:.4f} dB", flush=True)
            print(f"  Size  = {run_result['mean_total_size_bytes']:.1f} ± {run_result['std_total_size_bytes']:.1f} B", flush=True)
            print(f"  SHA   = {run_result['weights_sha256']}", flush=True)

    # Aggregate
    aggregated = {}
    for cfg in configs:
        runs = [r for r in all_runs if r['config_name'] == cfg]
        if not runs:
            continue
        psnrs = np.array([r['mean_psnr_db'] for r in runs])
        sizes = np.array([r['mean_total_size_bytes'] for r in runs])
        aggregated[cfg] = {
            'n_seeds': len(runs),
            'seeds': [r['seed'] for r in runs],
            'mean_psnr_db': float(psnrs.mean()),
            'std_psnr_db': float(psnrs.std()),
            'mean_size_bytes': float(sizes.mean()),
            'std_size_bytes': float(sizes.std()),
        }

    print(f"\n[exp36] AGGREGATED:", flush=True)
    for name, agg in aggregated.items():
        print(f"  {name}:", flush=True)
        print(f"    PSNR = {agg['mean_psnr_db']:.4f} ± {agg['std_psnr_db']:.4f} dB", flush=True)
        print(f"    Size = {agg['mean_size_bytes']:.1f} ± {agg['std_size_bytes']:.1f} B", flush=True)

    # Comparison
    coin = aggregated.get('coin_per_image', {})
    bhuh = aggregated.get('bhuh_hierarchical', {})
    comparison = {}
    if coin and bhuh:
        psnr_diff = bhuh['mean_psnr_db'] - coin['mean_psnr_db']
        size_ratio = bhuh['mean_size_bytes'] / max(1, coin['mean_size_bytes'])
        
        if coin['mean_psnr_db'] >= bhuh['mean_psnr_db'] and coin['mean_size_bytes'] <= bhuh['mean_size_bytes']:
            winner = "COIN (dominates on both axes)"
        elif bhuh['mean_psnr_db'] >= coin['mean_psnr_db'] and bhuh['mean_size_bytes'] <= coin['mean_size_bytes']:
            winner = "BHUH hierarchical (dominates on both axes)"
        else:
            winner = "trade-off (neither dominates)"
        
        comparison = {
            'coin_per_image': {'psnr_db': coin['mean_psnr_db'], 'size_bytes': coin['mean_size_bytes']},
            'bhuh_hierarchical': {'psnr_db': bhuh['mean_psnr_db'], 'size_bytes': bhuh['mean_size_bytes']},
            'psnr_diff_bhuh_minus_coin': float(psnr_diff),
            'size_ratio_bhuh_over_coin': float(size_ratio),
            'winner': winner,
        }
        print(f"\n[exp36] CONTROLLED COMPARISON (single-omega, same entropy coding):", flush=True)
        print(f"  COIN (per-image SIREN): {coin['mean_psnr_db']:.2f} dB, {coin['mean_size_bytes']:.0f} B", flush=True)
        print(f"  BHUH (hierarchical shared backbone): {bhuh['mean_psnr_db']:.2f} dB, {bhuh['mean_size_bytes']:.0f} B", flush=True)
        print(f"  PSNR diff (BHUH - COIN): {psnr_diff:+.2f} dB", flush=True)
        print(f"  Size ratio (BHUH/COIN): {size_ratio:.4f}x", flush=True)
        print(f"  WINNER: {winner}", flush=True)

    output = {
        'experiment': 'experiment_36_hierarchical_isolated',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'Hierarchical K=50 sharing (shared backbone across clustered images) '
                       'beats per-image COIN when multi-omega and entropy coding are controlled.',
        'config': {
            'hidden_features': hidden_features,
            'hidden_layers': hidden_layers,
            'omega': omega,
            'epochs': epochs,
            'lr': lr,
            'K': K,
            'num_images': num_images,
            'image_size': image_size,
            'seeds': seeds,
            'note': 'SINGLE omega=30 for both configs. SAME entropy coding for both. '
                    'Only difference: per-image SIREN vs shared backbone.',
        },
        'all_runs': all_runs,
        'aggregated': aggregated,
        'controlled_comparison': comparison,
    }

    out_json_path = os.path.join(output_dir, 'experiment_36_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha

    print(f"\n[exp36] DONE", flush=True)
    print(f"  JSON SHA-256: {json_sha}", flush=True)

    print("\n---JSON_BEGIN---")
    print(json.dumps(output, indent=2, default=str))
    print("---JSON_END---")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='42,123,2024')
    parser.add_argument('--num-images', type=int, default=30)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--hidden-features', type=int, default=64)
    parser.add_argument('--hidden-layers', type=int, default=2)
    parser.add_argument('--omega', type=float, default=30.0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp36_out')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    run_experiment(
        seeds=seeds, num_images=args.num_images, image_size=args.size,
        hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
        omega=args.omega, epochs=args.epochs, lr=args.lr, K=args.K,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
