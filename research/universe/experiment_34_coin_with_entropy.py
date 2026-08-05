"""
experiment_34_coin_with_entropy.py
====================================
Experiment 34 — COIN with Entropy Coding (isolating the architecture advantage).

Experiments 29-33 established that BHUH beats COIN at 8/8 configurations.
But BHUH and COIN differ in TWO ways:
  1. Architecture: BHUH uses multi-omega SIREN [10,50]; COIN uses single-omega=30
  2. Entropy coding: BHUH uses KMeans K=50 + arithmetic coding; COIN uses
     simple min-max quantization at fixed bit depths

This experiment gives COIN the SAME entropy coding pipeline as BHUH
(KMeans K=50 + arithmetic coding), so the only remaining difference is
the architecture (multi-omega vs single-omega).

Hypothesis: if BHUH still wins, the advantage comes from the multi-omega
architecture. If COIN+entropy catches up, the advantage was entirely from
the entropy coding pipeline.

ANTI-FABRICATION: same protocol as Exp 29-33.
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
    train_coin_one_image, CoinMLP,
)
from experiment_29_combined_pipeline import (
    hierarchical_kmeans_cluster, entropy_code_indices,
)


def get_model_weights_flat(model: nn.Module) -> np.ndarray:
    """Get all model weights as a single flat numpy array."""
    parts = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
    return np.concatenate(parts)


def evaluate_model_psnr(model: CoinMLP, img: np.ndarray) -> float:
    """Evaluate PSNR of model on image."""
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


def run_coin_with_entropy(images: np.ndarray, hidden_features: int,
                            hidden_layers: int, omega: float, epochs: int,
                            lr: float, K: int, seed: int, output_dir: str) -> Dict:
    """Run COIN (single-omega SIREN) with KMeans + arithmetic coding pipeline."""
    import gc

    weights_per_image: List[np.ndarray] = []
    psnrs = []
    train_times = []

    for i, img in enumerate(images):
        # Train COIN model (single-omega SIREN)
        model, psnr, t_train = train_coin_one_image(
            img, hidden_features=hidden_features, hidden_layers=hidden_layers,
            omega=omega, epochs=epochs, lr=lr, seed=seed,
        )
        weights = get_model_weights_flat(model)
        weights_per_image.append(weights.astype(np.float32))
        psnrs.append(psnr)
        train_times.append(t_train)
        del model
        gc.collect()

    # Apply KMeans clustering + arithmetic coding (same as BHUH)
    cluster_result = hierarchical_kmeans_cluster(weights_per_image, K=K, seed=seed)
    entropy_result = entropy_code_indices(cluster_result)

    del weights_per_image
    gc.collect()

    # Compute per-image total size
    per_img_sizes = entropy_result['coded_sizes_per_image']
    codebook_share = entropy_result['total_bytes_codebook'] / len(per_img_sizes)
    total_per_img_size = [s + codebook_share for s in per_img_sizes]

    psnr_arr = np.array(psnrs)
    size_arr = np.array(total_per_img_size)

    # Save weights file for SHA-256
    codebook_bytes = cluster_result['codebook'].astype(np.float32).tobytes()
    payload = bytearray()
    payload += codebook_bytes
    for idx in cluster_result['indices_per_image']:
        payload += idx.astype(np.int32).tobytes()
    weights_file = os.path.join(
        output_dir, f'exp34_weights_seed{seed}.bin')
    with open(weights_file, 'wb') as f:
        f.write(bytes(payload))
    with open(weights_file, 'rb') as f:
        weights_sha = hashlib.sha256(f.read()).hexdigest()

    return {
        'seed': seed,
        'mean_psnr_db': float(psnr_arr.mean()),
        'std_psnr_db': float(psnr_arr.std()),
        'mean_total_size_bytes': float(size_arr.mean()),
        'std_total_size_bytes': float(size_arr.std()),
        'mean_train_time_s': float(np.mean(train_times)),
        'codebook_bytes': entropy_result['total_bytes_codebook'],
        'weights_file': weights_file,
        'weights_sha256': weights_sha,
    }


def run_experiment(seeds: List[int], num_images: int, image_size: int,
                    hidden_features: int, hidden_layers: int, omega: float,
                    epochs: int, lr: float, K: int, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    import gc

    print(f"[exp34] loading {num_images} images at {image_size}x{image_size}...", flush=True)
    images, names = load_scikit_images(num_images, image_size)
    print(f"[exp34] loaded {len(images)} images", flush=True)

    # Use cached COIN baseline (raw, from Exp 29) for comparison
    exp29_out = os.path.join(os.path.dirname(output_dir), '_exp29_out')
    coin_cache = os.path.join(exp29_out, 'coin_baseline_cache.json')
    if os.path.exists(coin_cache):
        with open(coin_cache) as f:
            coin_data = json.load(f)
        coin_raw_psnr = coin_data['mean_psnr_db']
        coin_raw_size = coin_data['mean_weights_bytes_float16']
        print(f"[exp34] COIN raw (float16, no entropy): PSNR={coin_raw_psnr:.4f}, size={coin_raw_size:.1f} B", flush=True)
    else:
        coin_raw_psnr = None
        coin_raw_size = None
        print(f"[exp34] WARNING: COIN raw baseline not found", flush=True)

    # Run COIN + entropy coding for each seed
    all_runs = []
    for seed in seeds:
        ckpt_path = os.path.join(output_dir, f'ckpt_seed{seed}.json')
        if os.path.exists(ckpt_path):
            print(f"\n[exp34] LOADING checkpoint seed={seed}", flush=True)
            with open(ckpt_path) as f:
                run_result = json.load(f)
            all_runs.append(run_result)
            print(f"  CACHED: PSNR={run_result['mean_psnr_db']:.4f} dB, "
                  f"size={run_result['mean_total_size_bytes']:.1f} B", flush=True)
            continue

        print(f"\n[exp34] === COIN+entropy, seed={seed} ===", flush=True)
        t0 = time.time()
        run_result = run_coin_with_entropy(
            images=images, hidden_features=hidden_features,
            hidden_layers=hidden_layers, omega=omega, epochs=epochs,
            lr=lr, K=K, seed=seed, output_dir=output_dir,
        )
        run_result['total_time_s'] = time.time() - t0

        with open(ckpt_path, 'w') as f:
            json.dump(run_result, f, indent=2, default=str)
        all_runs.append(run_result)

        print(f"  PSNR            = {run_result['mean_psnr_db']:.4f} ± "
              f"{run_result['std_psnr_db']:.4f} dB", flush=True)
        print(f"  Size per image  = {run_result['mean_total_size_bytes']:.1f} ± "
              f"{run_result['std_total_size_bytes']:.1f} B", flush=True)
        print(f"  weights SHA-256 = {run_result['weights_sha256']}", flush=True)

    # Aggregate across seeds
    psnrs = np.array([r['mean_psnr_db'] for r in all_runs])
    sizes = np.array([r['mean_total_size_bytes'] for r in all_runs])
    aggregated = {
        'n_seeds': len(all_runs),
        'seeds': [r['seed'] for r in all_runs],
        'mean_psnr_db': float(psnrs.mean()),
        'std_psnr_db': float(psnrs.std()),
        'mean_size_bytes': float(sizes.mean()),
        'std_size_bytes': float(sizes.std()),
    }
    print(f"\n[exp34] AGGREGATED COIN+entropy:", flush=True)
    print(f"  PSNR  = {aggregated['mean_psnr_db']:.4f} ± {aggregated['std_psnr_db']:.4f} dB", flush=True)
    print(f"  Size  = {aggregated['mean_size_bytes']:.1f} ± {aggregated['std_size_bytes']:.1f} B", flush=True)

    # Load BHUH results from Exp 31 (no-pruning config from Exp 30 is the fair comparison)
    # The fair comparison is: same entropy coding pipeline, different architecture
    # BHUH (multi-omega) with no pruning: Exp 30 results
    exp30_path = os.path.join(os.path.dirname(output_dir), '_exp30_out', 'experiment_30_results.json')
    bhuH_no_prune = None
    if os.path.exists(exp30_path):
        with open(exp30_path) as f:
            exp30_data = json.load(f)
        # Use hl=2 (comparable architecture depth)
        bhuH_no_prune = next(
            (a for a in exp30_data['aggregated'] if a['hidden_layers'] == 2), None
        )
        if bhuH_no_prune:
            print(f"\n[exp34] Loaded BHUH (no-prune, hl=2) from Exp 30 for comparison", flush=True)
            print(f"  BHUH PSNR = {bhuH_no_prune['mean_psnr_db_across_seeds']:.4f} dB", flush=True)
            print(f"  BHUH size = {bhuH_no_prune['mean_total_size_bytes_across_seeds']:.1f} B", flush=True)

    # Build comparison table
    comparison = {
        'coin_raw_float16': {
            'psnr_db': coin_raw_psnr,
            'size_bytes': coin_raw_size,
            'description': 'COIN baseline: single-omega SIREN, float16 weights, no entropy coding',
        },
        'coin_with_entropy': {
            'psnr_db': aggregated['mean_psnr_db'],
            'size_bytes': aggregated['mean_size_bytes'],
            'description': 'COIN: single-omega SIREN + KMeans K=50 + arithmetic coding',
        },
        'bhuH_multi_omega_no_prune_hl2': {
            'psnr_db': bhuH_no_prune['mean_psnr_db_across_seeds'] if bhuH_no_prune else None,
            'size_bytes': bhuH_no_prune['mean_total_size_bytes_across_seeds'] if bhuH_no_prune else None,
            'description': 'BHUH: multi-omega [10,50] SIREN + KMeans K=50 + arithmetic coding (no pruning)',
        },
    }

    # Compute isolations
    if coin_raw_psnr and bhuH_no_prune:
        # Entropy coding contribution (COIN raw → COIN+entropy)
        entropy_psnr_gain = aggregated['mean_psnr_db'] - coin_raw_psnr
        entropy_size_reduction = coin_raw_size / max(1, aggregated['mean_size_bytes'])

        # Architecture contribution (COIN+entropy → BHUH)
        arch_psnr_gain = bhuH_no_prune['mean_psnr_db_across_seeds'] - aggregated['mean_psnr_db']
        arch_size_reduction = aggregated['mean_size_bytes'] / max(1, bhuH_no_prune['mean_total_size_bytes_across_seeds'])

        # Total BHUH advantage (COIN raw → BHUH)
        total_psnr_gain = bhuH_no_prune['mean_psnr_db_across_seeds'] - coin_raw_psnr
        total_size_reduction = coin_raw_size / max(1, bhuH_no_prune['mean_total_size_bytes_across_seeds'])

        isolation = {
            'entropy_coding_contribution': {
                'psnr_gain_db': float(entropy_psnr_gain),
                'size_reduction_x': float(entropy_size_reduction),
                'note': 'COIN raw (float16) → COIN + KMeans + arithmetic coding',
            },
            'architecture_contribution': {
                'psnr_gain_db': float(arch_psnr_gain),
                'size_reduction_x': float(arch_size_reduction),
                'note': 'COIN + entropy → BHUH (multi-omega, same entropy pipeline)',
            },
            'total_bhuH_advantage': {
                'psnr_gain_db': float(total_psnr_gain),
                'size_reduction_x': float(total_size_reduction),
                'note': 'COIN raw → BHUH (combined architecture + entropy)',
            },
        }

        print(f"\n[exp34] ISOLATION ANALYSIS:", flush=True)
        print(f"  Entropy coding contribution:", flush=True)
        print(f"    PSNR gain: {entropy_psnr_gain:+.4f} dB", flush=True)
        print(f"    Size reduction: {entropy_size_reduction:.4f}x", flush=True)
        print(f"  Architecture (multi-omega) contribution:", flush=True)
        print(f"    PSNR gain: {arch_psnr_gain:+.4f} dB", flush=True)
        print(f"    Size reduction: {arch_size_reduction:.4f}x", flush=True)
        print(f"  Total BHUH advantage:", flush=True)
        print(f"    PSNR gain: {total_psnr_gain:+.4f} dB", flush=True)
        print(f"    Size reduction: {total_size_reduction:.4f}x", flush=True)
    else:
        isolation = None

    output = {
        'experiment': 'experiment_34_coin_with_entropy',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'If BHUH still wins after COIN gets the same entropy coding, '
                       'the advantage comes from the multi-omega architecture.',
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
        },
        'all_runs': all_runs,
        'aggregated': aggregated,
        'comparison': comparison,
        'isolation_analysis': isolation,
    }

    out_json_path = os.path.join(output_dir, 'experiment_34_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha

    print(f"\n[exp34] DONE", flush=True)
    print(f"  output JSON: {out_json_path}", flush=True)
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
    parser.add_argument('--hidden-layers', type=int, default=5)
    parser.add_argument('--omega', type=float, default=30.0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp34_out')
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
