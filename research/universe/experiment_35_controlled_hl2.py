"""
experiment_35_controlled_hl2.py
================================
Experiment 35 — Controlled Architecture Comparison at hl=2.

Experiment 34 found that COIN+entropy dominates BHUH, but the comparison
was unfair: COIN used hl=5 while BHUH used hl=2. This experiment runs
COIN+entropy at hl=2 (same depth as BHUH) for a fully controlled comparison.

If COIN+entropy hl=2 still dominates BHUH hl=2, the multi-omega architecture
is definitively proven to not add value over single-omega when entropy coding
is controlled.

ANTI-FABRICATION: same protocol as Exp 29-34.
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
    parts = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
    return np.concatenate(parts)


def run_config(images: np.ndarray, hidden_features: int, hidden_layers: int,
               omega: float, epochs: int, lr: float, K: int, seed: int,
               output_dir: str, config_name: str) -> Dict:
    """Run a single config with entropy coding."""
    import gc

    weights_per_image: List[np.ndarray] = []
    psnrs = []
    train_times = []

    for i, img in enumerate(images):
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

    cluster_result = hierarchical_kmeans_cluster(weights_per_image, K=K, seed=seed)
    entropy_result = entropy_code_indices(cluster_result)
    del weights_per_image
    gc.collect()

    per_img_sizes = entropy_result['coded_sizes_per_image']
    codebook_share = entropy_result['total_bytes_codebook'] / len(per_img_sizes)
    total_per_img_size = [s + codebook_share for s in per_img_sizes]

    psnr_arr = np.array(psnrs)
    size_arr = np.array(total_per_img_size)

    codebook_bytes = cluster_result['codebook'].astype(np.float32).tobytes()
    payload = bytearray()
    payload += codebook_bytes
    for idx in cluster_result['indices_per_image']:
        payload += idx.astype(np.int32).tobytes()
    weights_file = os.path.join(output_dir, f'exp35_{config_name}_seed{seed}.bin')
    with open(weights_file, 'wb') as f:
        f.write(bytes(payload))
    with open(weights_file, 'rb') as f:
        weights_sha = hashlib.sha256(f.read()).hexdigest()

    return {
        'config_name': config_name,
        'seed': seed,
        'hidden_layers': hidden_layers,
        'omega': omega,
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
                    hidden_features: int, epochs: int, lr: float,
                    K: int, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[exp35] loading {num_images} images at {image_size}x{image_size}...", flush=True)
    images, names = load_scikit_images(num_images, image_size)
    print(f"[exp35] loaded {len(images)} images", flush=True)

    configs = [
        {'name': 'coin_single_omega_hl2', 'hidden_layers': 2, 'omega': 30.0},
        {'name': 'bhuH_multi_omega_hl2', 'hidden_layers': 2, 'omega': None},  # multi-omega handled differently
    ]

    all_runs = []
    for cfg in configs:
        for seed in seeds:
            ckpt_path = os.path.join(output_dir, f'ckpt_{cfg["name"]}_seed{seed}.json')
            if os.path.exists(ckpt_path):
                print(f"\n[exp35] LOADING {cfg['name']} seed={seed}", flush=True)
                with open(ckpt_path) as f:
                    run_result = json.load(f)
                all_runs.append(run_result)
                print(f"  CACHED: PSNR={run_result['mean_psnr_db']:.4f}, "
                      f"size={run_result['mean_total_size_bytes']:.1f}", flush=True)
                continue

            print(f"\n[exp35] === {cfg['name']}, seed={seed} ===", flush=True)
            t0 = time.time()

            if cfg['omega'] is not None:
                # Single-omega COIN
                run_result = run_config(
                    images=images, hidden_features=hidden_features,
                    hidden_layers=cfg['hidden_layers'], omega=cfg['omega'],
                    epochs=epochs, lr=lr, K=K, seed=seed,
                    output_dir=output_dir, config_name=cfg['name'],
                )
            else:
                # Multi-omega BHUH (import from experiment_29)
                from experiment_29_combined_pipeline import (
                    MultiOmegaSirenMLP, train_combined_pipeline_one_image,
                    evaluate_post_pruning_psnr,
                )
                import gc
                omegas = [10.0, 50.0]
                weights_per_image = []
                psnrs = []
                train_times = []
                for img in images:
                    model, psnr, t_train = train_combined_pipeline_one_image(
                        img, hidden_features=hidden_features,
                        hidden_layers=cfg['hidden_layers'],
                        omegas=omegas, epochs=epochs, lr=lr, seed=seed,
                    )
                    weights = model.get_flat_weights()
                    weights_per_image.append(weights.astype(np.float32))
                    psnrs.append(psnr)
                    train_times.append(t_train)
                    del model
                    gc.collect()

                cluster_result = hierarchical_kmeans_cluster(weights_per_image, K=K, seed=seed)
                entropy_result = entropy_code_indices(cluster_result)
                del weights_per_image
                gc.collect()

                per_img_sizes = entropy_result['coded_sizes_per_image']
                codebook_share = entropy_result['total_bytes_codebook'] / len(per_img_sizes)
                total_per_img_size = [s + codebook_share for s in per_img_sizes]
                psnr_arr = np.array(psnrs)
                size_arr = np.array(total_per_img_size)

                codebook_bytes = cluster_result['codebook'].astype(np.float32).tobytes()
                payload = bytearray()
                payload += codebook_bytes
                for idx in cluster_result['indices_per_image']:
                    payload += idx.astype(np.int32).tobytes()
                weights_file = os.path.join(output_dir, f'exp35_{cfg["name"]}_seed{seed}.bin')
                with open(weights_file, 'wb') as f:
                    f.write(bytes(payload))
                with open(weights_file, 'rb') as f:
                    weights_sha = hashlib.sha256(f.read()).hexdigest()

                run_result = {
                    'config_name': cfg['name'],
                    'seed': seed,
                    'hidden_layers': cfg['hidden_layers'],
                    'omega': 'multi [10,50]',
                    'mean_psnr_db': float(psnr_arr.mean()),
                    'std_psnr_db': float(psnr_arr.std()),
                    'mean_total_size_bytes': float(size_arr.mean()),
                    'std_total_size_bytes': float(size_arr.std()),
                    'mean_train_time_s': float(np.mean(train_times)),
                    'codebook_bytes': entropy_result['total_bytes_codebook'],
                    'weights_file': weights_file,
                    'weights_sha256': weights_sha,
                }

            run_result['total_time_s'] = time.time() - t0
            with open(ckpt_path, 'w') as f:
                json.dump(run_result, f, indent=2, default=str)
            all_runs.append(run_result)

            print(f"  PSNR  = {run_result['mean_psnr_db']:.4f} ± {run_result['std_psnr_db']:.4f} dB", flush=True)
            print(f"  Size  = {run_result['mean_total_size_bytes']:.1f} ± {run_result['std_total_size_bytes']:.1f} B", flush=True)

    # Aggregate
    aggregated = {}
    for cfg in configs:
        runs = [r for r in all_runs if r['config_name'] == cfg['name']]
        if not runs:
            continue
        psnrs = np.array([r['mean_psnr_db'] for r in runs])
        sizes = np.array([r['mean_total_size_bytes'] for r in runs])
        aggregated[cfg['name']] = {
            'n_seeds': len(runs),
            'seeds': [r['seed'] for r in runs],
            'mean_psnr_db': float(psnrs.mean()),
            'std_psnr_db': float(psnrs.std()),
            'mean_size_bytes': float(sizes.mean()),
            'std_size_bytes': float(sizes.std()),
        }

    print(f"\n[exp35] AGGREGATED:", flush=True)
    for name, agg in aggregated.items():
        print(f"  {name}:", flush=True)
        print(f"    PSNR = {agg['mean_psnr_db']:.4f} ± {agg['std_psnr_db']:.4f} dB", flush=True)
        print(f"    Size = {agg['mean_size_bytes']:.1f} ± {agg['std_size_bytes']:.1f} B", flush=True)

    # Direct comparison
    coin = aggregated.get('coin_single_omega_hl2', {})
    bhuh = aggregated.get('bhuH_multi_omega_hl2', {})
    comparison = {}
    if coin and bhuh:
        psnr_diff = coin['mean_psnr_db'] - bhuh['mean_psnr_db']
        size_ratio = bhuh['mean_size_bytes'] / max(1, coin['mean_size_bytes'])
        winner = "COIN" if coin['mean_size_bytes'] <= bhuh['mean_size_bytes'] else "BHUH"
        # Actually winner should be based on RD dominance
        if coin['mean_psnr_db'] >= bhuh['mean_psnr_db'] and coin['mean_size_bytes'] <= bhuh['mean_size_bytes']:
            winner = "COIN (dominates on both axes)"
        elif bhuh['mean_psnr_db'] >= coin['mean_psnr_db'] and bhuh['mean_size_bytes'] <= coin['mean_size_bytes']:
            winner = "BHUH (dominates on both axes)"
        else:
            winner = "trade-off (neither dominates)"
        comparison = {
            'coin_hl2': {'psnr_db': coin['mean_psnr_db'], 'size_bytes': coin['mean_size_bytes']},
            'bhuh_hl2': {'psnr_db': bhuh['mean_psnr_db'], 'size_bytes': bhuh['mean_size_bytes']},
            'psnr_diff_coin_minus_bhuh': float(psnr_diff),
            'size_ratio_bhuh_over_coin': float(size_ratio),
            'winner': winner,
        }
        print(f"\n[exp35] CONTROLLED COMPARISON (hl=2, same entropy coding):", flush=True)
        print(f"  COIN (single-omega=30): {coin['mean_psnr_db']:.2f} dB, {coin['mean_size_bytes']:.0f} B", flush=True)
        print(f"  BHUH (multi-omega [10,50]): {bhuh['mean_psnr_db']:.2f} dB, {bhuh['mean_size_bytes']:.0f} B", flush=True)
        print(f"  PSNR diff (COIN - BHUH): {psnr_diff:+.2f} dB", flush=True)
        print(f"  Size ratio (BHUH/COIN): {size_ratio:.4f}x", flush=True)
        print(f"  WINNER: {winner}", flush=True)

    output = {
        'experiment': 'experiment_35_controlled_hl2',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'At hl=2 with same entropy coding, COIN (single-omega) still dominates BHUH (multi-omega).',
        'config': {
            'hidden_features': hidden_features,
            'epochs': epochs,
            'lr': lr,
            'K': K,
            'num_images': num_images,
            'image_size': image_size,
            'seeds': seeds,
        },
        'all_runs': all_runs,
        'aggregated': aggregated,
        'controlled_comparison': comparison,
    }

    out_json_path = os.path.join(output_dir, 'experiment_35_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha

    print(f"\n[exp35] DONE", flush=True)
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
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp35_out')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    run_experiment(
        seeds=seeds, num_images=args.num_images, image_size=args.size,
        hidden_features=args.hidden_features, epochs=args.epochs, lr=args.lr,
        K=args.K, output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
