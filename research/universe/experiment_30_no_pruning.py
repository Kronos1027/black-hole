"""
experiment_30_no_pruning.py
=============================
Experiment 30 — Isolate the L1 pruning hypothesis.

Experiment 29 found that the combined pipeline produced PSNR ~12-15 dB,
far below the projected 34-35 dB. The pre-prune PSNR was 37-49 dB,
suggesting L1 pruning with threshold=0.01 destroys the representation.

This experiment runs the EXACT same pipeline as Experiment 29 but
**skips the L1 pruning step entirely**. Weights go directly from training
to KMeans clustering + arithmetic coding.

Hypothesis: if pruning is the culprit, PSNR should remain at the pre-prune
level (~37 dB for hl=2, ~49 dB for hl=5).

ANTI-FABRICATION: same protocol as Experiment 29.
    - 3 seeds, 30 images, 300 epochs, hl=[2,5]
    - JSON output to stdout with SHA-256
    - No hyperparameter tuning to match any target
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
from arithmetic_codec import encode_weights, decode_weights
from coin_baseline_exp29 import load_scikit_images, normalize_to_pm1, denormalize_from_pm1, train_coin_one_image
from experiment_29_combined_pipeline import (
    MultiOmegaSirenMLP, train_combined_pipeline_one_image,
    evaluate_post_pruning_psnr, hierarchical_kmeans_cluster, entropy_code_indices,
)


def evaluate_psnr(model: MultiOmegaSirenMLP, img: np.ndarray) -> float:
    """Evaluate PSNR of model on image (no pruning applied)."""
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


def run_experiment(seeds: List[int], num_images: int, image_size: int,
                    hidden_features: int, hidden_layers_options: List[int],
                    omegas: List[float], epochs: int, lr: float,
                    K: int, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    import gc

    print(f"[exp30] loading {num_images} images at {image_size}x{image_size}...", flush=True)
    images, names = load_scikit_images(num_images, image_size)
    print(f"[exp30] loaded {len(images)} images", flush=True)

    # Use cached COIN baseline from Exp 29 if available
    exp29_out = os.path.join(os.path.dirname(output_dir), '_exp29_out')
    coin_cache_exp29 = os.path.join(exp29_out, 'coin_baseline_cache.json')
    coin_cache_local = os.path.join(output_dir, 'coin_baseline_cache.json')

    if os.path.exists(coin_cache_exp29):
        with open(coin_cache_exp29) as f:
            coin_data = json.load(f)
        coin_mean_psnr = coin_data['mean_psnr_db']
        coin_mean_size = coin_data['mean_weights_bytes_float16']
        print(f"[coin_baseline] REUSED from Exp 29: PSNR={coin_mean_psnr:.4f}, size={coin_mean_size:.1f} B", flush=True)
    elif os.path.exists(coin_cache_local):
        with open(coin_cache_local) as f:
            coin_data = json.load(f)
        coin_mean_psnr = coin_data['mean_psnr_db']
        coin_mean_size = coin_data['mean_weights_bytes_float16']
        print(f"[coin_baseline] CACHED locally: PSNR={coin_mean_psnr:.4f}, size={coin_mean_size:.1f} B", flush=True)
    else:
        print(f"\n[exp30] === Running COIN baseline (seed=42) ===", flush=True)
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
                print(f"  [coin] {i+1}/{len(images)}: PSNR={psnr:.2f} dB", flush=True)
        coin_mean_psnr = float(np.mean(coin_psnrs))
        coin_mean_size = float(np.mean(coin_sizes))
        with open(coin_cache_local, 'w') as f:
            json.dump({
                'mean_psnr_db': coin_mean_psnr,
                'mean_weights_bytes_float16': coin_mean_size,
                'std_psnr_db': float(np.std(coin_psnrs)),
                'std_weights_bytes': float(np.std(coin_sizes)),
                'config': {'hidden_features': hidden_features, 'hidden_layers': 5,
                            'omega': 30.0, 'epochs': epochs, 'lr': lr, 'seed': 42,
                            'num_images': num_images},
            }, f, indent=2)
        print(f"[coin_baseline] mean PSNR={coin_mean_psnr:.4f}, mean size={coin_mean_size:.1f} B", flush=True)

    all_runs = []
    for hidden_layers in hidden_layers_options:
        for seed in seeds:
            ckpt_path = os.path.join(output_dir, f'ckpt_hl{hidden_layers}_seed{seed}.json')
            if os.path.exists(ckpt_path):
                print(f"\n[exp30] LOADING checkpoint hl={hidden_layers} seed={seed}", flush=True)
                with open(ckpt_path) as f:
                    run_summary = json.load(f)
                all_runs.append(run_summary)
                print(f"  [comb hl={hidden_layers} seed={seed}] CACHED: "
                      f"PSNR={run_summary['mean_psnr_db']:.4f} dB, "
                      f"size={run_summary['mean_total_size_bytes']:.1f} B, "
                      f"red={run_summary['size_reduction_vs_coin_x']:.4f}x", flush=True)
                continue

            print(f"\n[exp30] === No-pruning pipeline: hidden_layers={hidden_layers}, seed={seed} ===",
                  flush=True)
            run_results = {'hidden_layers': hidden_layers, 'seed': seed, 'per_image': []}
            weights_per_image: List[np.ndarray] = []
            psnrs = []
            train_times = []

            for i, img in enumerate(images):
                model, _, t_train = train_combined_pipeline_one_image(
                    img, hidden_features=hidden_features, hidden_layers=hidden_layers,
                    omegas=omegas, epochs=epochs, lr=lr, seed=seed,
                )
                # KEY DIFFERENCE FROM EXP 29: NO L1 PRUNING
                psnr = evaluate_psnr(model, img)

                weights = model.get_flat_weights()
                weights_per_image.append(weights.astype(np.float32))
                psnrs.append(psnr)
                train_times.append(t_train)

                run_results['per_image'].append({
                    'index': i,
                    'name': names[i],
                    'psnr_db': psnr,
                    'train_time_s': t_train,
                    'n_params': model.num_params(),
                })
                del model
                gc.collect()
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [comb hl={hidden_layers} seed={seed}] {i+1}/{len(images)}: "
                          f"PSNR={psnr:.2f} dB", flush=True)

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
                output_dir, f'exp30_weights_hl{hidden_layers}_seed{seed}.bin')
            with open(weights_file, 'wb') as f:
                f.write(payload_bytes)
            with open(weights_file, 'rb') as f:
                weights_sha = hashlib.sha256(f.read()).hexdigest()

            psnr_arr = np.array(psnrs)
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
                    'pruning': 'NONE',  # KEY DIFFERENCE
                    'K': K,
                    'num_images': len(images),
                    'image_size': image_size,
                },
                'mean_psnr_db': float(psnr_arr.mean()),
                'std_psnr_db': float(psnr_arr.std()),
                'mean_total_size_bytes': float(total_size_arr.mean()),
                'std_total_size_bytes': float(total_size_arr.std()),
                'mean_train_time_s': float(np.mean(train_times)),
                'codebook_bytes': entropy_result['total_bytes_codebook'],
                'total_bytes_all_images': entropy_result['total_bytes_with_entropy_coding'],
                'weights_file': weights_file,
                'weights_sha256': weights_sha,
                'size_reduction_vs_coin_x': float(coin_mean_size / total_size_arr.mean()),
                'psnr_delta_vs_coin_db': float(psnr_arr.mean() - coin_mean_psnr),
                'per_image': run_results['per_image'],
            }
            with open(ckpt_path, 'w') as f:
                json.dump(run_summary, f, indent=2, default=str)
            all_runs.append(run_summary)

            print(f"  [comb hl={hidden_layers} seed={seed}] SUMMARY:", flush=True)
            print(f"    PSNR             = {psnr_arr.mean():.4f} ± {psnr_arr.std():.4f} dB", flush=True)
            print(f"    Size per image   = {total_size_arr.mean():.1f} ± {total_size_arr.std():.1f} B", flush=True)
            print(f"    vs COIN: PSNR Δ  = {run_summary['psnr_delta_vs_coin_db']:+.4f} dB, "
                  f"size reduction = {run_summary['size_reduction_vs_coin_x']:.4f}x", flush=True)
            print(f"    weights SHA-256  = {weights_sha}", flush=True)

            del cluster_result, entropy_result
            gc.collect()

    aggregated = []
    for hidden_layers in hidden_layers_options:
        runs = [r for r in all_runs if r['hidden_layers'] == hidden_layers]
        psnrs = np.array([r['mean_psnr_db'] for r in runs])
        sizes = np.array([r['mean_total_size_bytes'] for r in runs])
        reds = np.array([r['size_reduction_vs_coin_x'] for r in runs])
        agg = {
            'hidden_layers': hidden_layers,
            'n_seeds': len(runs),
            'seeds': [r['seed'] for r in runs],
            'mean_psnr_db_across_seeds': float(psnrs.mean()),
            'std_psnr_db_across_seeds': float(psnrs.std()),
            'mean_total_size_bytes_across_seeds': float(sizes.mean()),
            'std_total_size_bytes_across_seeds': float(sizes.std()),
            'mean_size_reduction_vs_coin_x': float(reds.mean()),
            'std_size_reduction_vs_coin_x': float(reds.std()),
        }
        aggregated.append(agg)
        print(f"\n[exp30] AGGREGATED hidden_layers={hidden_layers}:", flush=True)
        print(f"  PSNR  = {agg['mean_psnr_db_across_seeds']:.4f} ± "
              f"{agg['std_psnr_db_across_seeds']:.4f} dB", flush=True)
        print(f"  Size  = {agg['mean_total_size_bytes_across_seeds']:.1f} ± "
              f"{agg['std_total_size_bytes_across_seeds']:.1f} B", flush=True)
        print(f"  Reduction vs COIN = {agg['mean_size_reduction_vs_coin_x']:.4f}x ± "
              f"{agg['std_size_reduction_vs_coin_x']:.4f}", flush=True)

    output = {
        'experiment': 'experiment_30_no_pruning',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'Removing L1 pruning should preserve pre-prune PSNR (~37 dB hl=2, ~49 dB hl=5)',
        'coin_baseline': {
            'mean_psnr_db': coin_mean_psnr,
            'mean_weights_bytes_float16': coin_mean_size,
        },
        'combined_pipeline_runs': all_runs,
        'aggregated': aggregated,
    }

    out_json_path = os.path.join(output_dir, 'experiment_30_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha

    print(f"\n[exp30] DONE", flush=True)
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
    parser.add_argument('--hidden-layers', type=str, default='2,5')
    parser.add_argument('--omegas', type=str, default='10,50')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp30_out')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    hidden_layers_opts = [int(h) for h in args.hidden_layers.split(',')]
    omegas = [float(o) for o in args.omegas.split(',')]

    run_experiment(
        seeds=seeds, num_images=args.num_images, image_size=args.size,
        hidden_features=args.hidden_features, hidden_layers_options=hidden_layers_opts,
        omegas=omegas, epochs=args.epochs, lr=args.lr, K=args.K,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
