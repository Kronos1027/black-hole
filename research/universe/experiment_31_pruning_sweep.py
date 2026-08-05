"""
experiment_31_pruning_sweep.py
================================
Experiment 31 — L1 Pruning Threshold Sweep.

Experiments 29 and 30 established:
- Exp 29 (pruning threshold=0.01): PSNR collapsed to 12-15 dB
- Exp 30 (no pruning): PSNR preserved at 37-50 dB, but size only 2.7-4.3x

This experiment sweeps pruning thresholds [0.001, 0.002, 0.005, 0.01] to
map the size-PSNR tradeoff curve and identify the sweet spot.

For each threshold, we report:
- PSNR (mean ± std across 3 seeds)
- Size in bytes (mean ± std)
- Size reduction vs COIN (x)
- Sparsity achieved (%)
- **Byte parity vs COIN at same PSNR**: for each threshold, find the COIN
  configuration that achieves the same PSNR and compare byte sizes directly.

ANTI-FABRICATION: same protocol as Exp 29/30.
- 3 seeds (42, 123, 2024), 30 images, 300 epochs, hl=[2,5]
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
from coin_baseline_exp29 import (
    load_scikit_images, normalize_to_pm1, denormalize_from_pm1,
    train_coin_one_image, serialize_weights_float16, CoinMLP,
)
from experiment_29_combined_pipeline import (
    MultiOmegaSirenMLP, train_combined_pipeline_one_image,
    l1_prune, evaluate_post_pruning_psnr, hierarchical_kmeans_cluster,
    entropy_code_indices,
)


def evaluate_psnr(model: MultiOmegaSirenMLP, img: np.ndarray) -> float:
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


def run_single_threshold(threshold: float, images: np.ndarray, names: List[str],
                          hidden_features: int, hidden_layers: int,
                          omegas: List[float], epochs: int, lr: float,
                          K: int, seed: int, output_dir: str) -> Dict:
    """Run the pipeline for a single pruning threshold and seed."""
    import gc

    weights_per_image: List[np.ndarray] = []
    psnrs_pre_prune = []
    psnrs_post_prune = []
    train_times = []
    sparsities = []

    for i, img in enumerate(images):
        model, psnr_pre, t_train = train_combined_pipeline_one_image(
            img, hidden_features=hidden_features, hidden_layers=hidden_layers,
            omegas=omegas, epochs=epochs, lr=lr, seed=seed,
        )
        # Apply L1 pruning with the given threshold
        model_pruned, prune_stats = l1_prune(model, threshold=threshold)
        psnr_post = evaluate_post_pruning_psnr(model_pruned, img)

        weights = model_pruned.get_flat_weights()
        weights_per_image.append(weights.astype(np.float32))
        psnrs_pre_prune.append(psnr_pre)
        psnrs_post_prune.append(psnr_post)
        train_times.append(t_train)
        sparsities.append(prune_stats['sparsity'])

        del model, model_pruned
        gc.collect()

    # Cluster + entropy code
    cluster_result = hierarchical_kmeans_cluster(weights_per_image, K=K, seed=seed)
    entropy_result = entropy_code_indices(cluster_result)

    del weights_per_image
    gc.collect()

    # Compute per-image total size (codebook share + entropy-coded indices)
    per_img_sizes = entropy_result['coded_sizes_per_image']
    codebook_share = entropy_result['total_bytes_codebook'] / len(per_img_sizes)
    total_per_img_size = [s + codebook_share for s in per_img_sizes]

    psnr_pre_arr = np.array(psnrs_pre_prune)
    psnr_post_arr = np.array(psnrs_post_prune)
    size_arr = np.array(total_per_img_size)
    sparsity_arr = np.array(sparsities)

    # Save weights file for SHA-256
    codebook_bytes = cluster_result['codebook'].astype(np.float32).tobytes()
    payload = bytearray()
    payload += codebook_bytes
    for idx in cluster_result['indices_per_image']:
        payload += idx.astype(np.int32).tobytes()
    weights_file = os.path.join(
        output_dir, f'exp31_weights_hl{hidden_layers}_seed{seed}_thr{threshold}.bin')
    with open(weights_file, 'wb') as f:
        f.write(bytes(payload))
    with open(weights_file, 'rb') as f:
        weights_sha = hashlib.sha256(f.read()).hexdigest()

    return {
        'threshold': threshold,
        'hidden_layers': hidden_layers,
        'seed': seed,
        'mean_psnr_pre_prune_db': float(psnr_pre_arr.mean()),
        'std_psnr_pre_prune_db': float(psnr_pre_arr.std()),
        'mean_psnr_post_prune_db': float(psnr_post_arr.mean()),
        'std_psnr_post_prune_db': float(psnr_post_arr.std()),
        'mean_total_size_bytes': float(size_arr.mean()),
        'std_total_size_bytes': float(size_arr.std()),
        'mean_sparsity': float(sparsity_arr.mean()),
        'std_sparsity': float(sparsity_arr.std()),
        'codebook_bytes': entropy_result['total_bytes_codebook'],
        'weights_file': weights_file,
        'weights_sha256': weights_sha,
    }


def run_experiment(seeds: List[int], num_images: int, image_size: int,
                    hidden_features: int, hidden_layers_options: List[int],
                    omegas: List[float], epochs: int, lr: float,
                    thresholds: List[float], K: int, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    import gc

    print(f"[exp31] loading {num_images} images at {image_size}x{image_size}...", flush=True)
    images, names = load_scikit_images(num_images, image_size)
    print(f"[exp31] loaded {len(images)} images", flush=True)

    # Use cached COIN baseline from Exp 29 if available
    exp29_out = os.path.join(os.path.dirname(output_dir), '_exp29_out')
    coin_cache_exp29 = os.path.join(exp29_out, 'coin_baseline_cache.json')
    if os.path.exists(coin_cache_exp29):
        with open(coin_cache_exp29) as f:
            coin_data = json.load(f)
        coin_mean_psnr = coin_data['mean_psnr_db']
        coin_mean_size = coin_data['mean_weights_bytes_float16']
        print(f"[coin_baseline] REUSED from Exp 29: PSNR={coin_mean_psnr:.4f}, size={coin_mean_size:.1f} B", flush=True)
    else:
        print(f"\n[exp31] === Running COIN baseline (seed=42) ===", flush=True)
        coin_psnrs = []
        coin_sizes = []
        for i, img in enumerate(images):
            model, psnr, _ = train_coin_one_image(
                img, hidden_features=hidden_features, hidden_layers=5,
                omega=30.0, epochs=epochs, lr=lr, seed=42,
            )
            w_bytes = serialize_weights_float16(model)
            coin_psnrs.append(psnr)
            coin_sizes.append(len(w_bytes))
            del model
            gc.collect()
        coin_mean_psnr = float(np.mean(coin_psnrs))
        coin_mean_size = float(np.mean(coin_sizes))
        print(f"[coin_baseline] mean PSNR={coin_mean_psnr:.4f}, mean size={coin_mean_size:.1f} B", flush=True)

    # Run sweep: for each (hidden_layers, threshold, seed)
    all_runs = []
    for hidden_layers in hidden_layers_options:
        for threshold in thresholds:
            for seed in seeds:
                ckpt_path = os.path.join(
                    output_dir,
                    f'ckpt_hl{hidden_layers}_thr{threshold}_seed{seed}.json'
                )
                if os.path.exists(ckpt_path):
                    print(f"\n[exp31] LOADING checkpoint hl={hidden_layers} thr={threshold} seed={seed}", flush=True)
                    with open(ckpt_path) as f:
                        run_result = json.load(f)
                    all_runs.append(run_result)
                    print(f"  CACHED: PSNR={run_result['mean_psnr_post_prune_db']:.4f} dB, "
                          f"size={run_result['mean_total_size_bytes']:.1f} B, "
                          f"sparsity={run_result['mean_sparsity']:.4f}", flush=True)
                    continue

                print(f"\n[exp31] === hl={hidden_layers}, threshold={threshold}, seed={seed} ===", flush=True)
                t0 = time.time()
                run_result = run_single_threshold(
                    threshold=threshold, images=images, names=names,
                    hidden_features=hidden_features, hidden_layers=hidden_layers,
                    omegas=omegas, epochs=epochs, lr=lr, K=K, seed=seed,
                    output_dir=output_dir,
                )
                run_result['size_reduction_vs_coin_x'] = float(coin_mean_size / run_result['mean_total_size_bytes'])
                run_result['psnr_delta_vs_coin_db'] = float(run_result['mean_psnr_post_prune_db'] - coin_mean_psnr)
                run_result['train_time_s'] = time.time() - t0

                with open(ckpt_path, 'w') as f:
                    json.dump(run_result, f, indent=2, default=str)
                all_runs.append(run_result)

                print(f"  PSNR pre-prune  = {run_result['mean_psnr_pre_prune_db']:.4f} ± {run_result['std_psnr_pre_prune_db']:.4f} dB", flush=True)
                print(f"  PSNR post-prune = {run_result['mean_psnr_post_prune_db']:.4f} ± {run_result['std_psnr_post_prune_db']:.4f} dB", flush=True)
                print(f"  Size per image  = {run_result['mean_total_size_bytes']:.1f} ± {run_result['std_total_size_bytes']:.4f} B", flush=True)
                print(f"  Sparsity        = {run_result['mean_sparsity']:.4f}", flush=True)
                print(f"  vs COIN: PSNR Δ = {run_result['psnr_delta_vs_coin_db']:+.4f} dB, "
                      f"size reduction = {run_result['size_reduction_vs_coin_x']:.4f}x", flush=True)
                print(f"  weights SHA-256 = {run_result['weights_sha256']}", flush=True)

    # Aggregate across seeds for each (hidden_layers, threshold)
    aggregated = []
    for hidden_layers in hidden_layers_options:
        for threshold in thresholds:
            runs = [r for r in all_runs
                     if r['hidden_layers'] == hidden_layers and r['threshold'] == threshold]
            if not runs:
                continue
            psnrs = np.array([r['mean_psnr_post_prune_db'] for r in runs])
            sizes = np.array([r['mean_total_size_bytes'] for r in runs])
            reds = np.array([r['size_reduction_vs_coin_x'] for r in runs])
            spars = np.array([r['mean_sparsity'] for r in runs])
            agg = {
                'hidden_layers': hidden_layers,
                'threshold': threshold,
                'n_seeds': len(runs),
                'seeds': [r['seed'] for r in runs],
                'mean_psnr_db': float(psnrs.mean()),
                'std_psnr_db': float(psnrs.std()),
                'mean_size_bytes': float(sizes.mean()),
                'std_size_bytes': float(sizes.std()),
                'mean_size_reduction_vs_coin_x': float(reds.mean()),
                'std_size_reduction_vs_coin_x': float(reds.std()),
                'mean_sparsity': float(spars.mean()),
                'std_sparsity': float(spars.std()),
            }
            aggregated.append(agg)
            print(f"\n[exp31] AGGREGATED hl={hidden_layers} thr={threshold}:", flush=True)
            print(f"  PSNR  = {agg['mean_psnr_db']:.4f} ± {agg['std_psnr_db']:.4f} dB", flush=True)
            print(f"  Size  = {agg['mean_size_bytes']:.1f} ± {agg['std_size_bytes']:.1f} B", flush=True)
            print(f"  Red   = {agg['mean_size_reduction_vs_coin_x']:.4f}x ± {agg['std_size_reduction_vs_coin_x']:.4f}", flush=True)
            print(f"  Spars = {agg['mean_sparsity']:.4f} ± {agg['std_sparsity']:.4f}", flush=True)

    # Byte parity analysis: for each threshold, find COIN config at same PSNR
    # COIN baseline: PSNR 64.09 dB, size 42114 B (float16 weights)
    # To achieve same PSNR as BHUH at each threshold, COIN would need to quantize
    # its weights more aggressively. We estimate this via the rate-distortion
    # relationship: PSNR ~ 20*log10(1/quant_step). For float16 → float8 → float4,
    # PSNR drops by ~6 dB per bit removed.
    # This is an approximation; a rigorous comparison would require running COIN
    # at each quantization level. We document this limitation honestly.
    parity_analysis = []
    coin_psnr = coin_mean_psnr
    coin_size = coin_mean_size
    for agg in aggregated:
        bhuH_psnr = agg['mean_psnr_db']
        bhuH_size = agg['mean_size_bytes']
        # How many dB below COIN is BHUH?
        psnr_gap_db = coin_psnr - bhuH_psnr
        # Estimate COIN size at same PSNR as BHUH
        # COIN at float16: 42114 B, 64.09 dB
        # To drop PSNR by `psnr_gap_db`, we can reduce bit depth.
        # Approximate: each bit removed from quantization reduces PSNR by ~6 dB
        # and halves the size.
        bits_to_remove = psnr_gap_db / 6.0
        estimated_coin_size_at_same_psnr = coin_size / (2 ** bits_to_remove)
        parity = {
            'hidden_layers': agg['hidden_layers'],
            'threshold': agg['threshold'],
            'bhuH_psnr_db': bhuH_psnr,
            'coin_psnr_db': coin_psnr,
            'psnr_gap_db': psnr_gap_db,
            'bhuH_size_bytes': bhuH_size,
            'estimated_coin_size_at_same_psnr_bytes': estimated_coin_size_at_same_psnr,
            'bhuH_vs_coin_at_same_psnr_ratio': bhuH_size / max(1, estimated_coin_size_at_same_psnr),
            'note': 'COIN size at same PSNR is ESTIMATED via 6 dB/bit heuristic, not measured. '
                    'A rigorous comparison requires running COIN at multiple quantization levels.',
        }
        parity_analysis.append(parity)
        print(f"\n[exp31] PARITY hl={agg['hidden_layers']} thr={agg['threshold']}:", flush=True)
        print(f"  BHUH PSNR = {bhuH_psnr:.2f} dB, COIN PSNR = {coin_psnr:.2f} dB (gap {psnr_gap_db:.2f} dB)", flush=True)
        print(f"  BHUH size = {bhuH_size:.1f} B", flush=True)
        print(f"  Est. COIN size at same PSNR = {estimated_coin_size_at_same_psnr:.1f} B (heuristic)", flush=True)
        print(f"  BHUH/COIN ratio at same PSNR = {bhuH_size / max(1, estimated_coin_size_at_same_psnr):.4f}x", flush=True)

    output = {
        'experiment': 'experiment_31_pruning_sweep',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'A pruning threshold between 0.001 and 0.005 finds a sweet spot: '
                       'enough sparsity for size reduction, not so much that PSNR collapses.',
        'coin_baseline': {
            'mean_psnr_db': coin_mean_psnr,
            'mean_weights_bytes_float16': coin_mean_size,
        },
        'thresholds_tested': thresholds,
        'config': {
            'hidden_features': hidden_features,
            'omegas': omegas,
            'epochs': epochs,
            'lr': lr,
            'K': K,
            'num_images': num_images,
            'image_size': image_size,
            'seeds': seeds,
        },
        'all_runs': all_runs,
        'aggregated': aggregated,
        'byte_parity_analysis': parity_analysis,
    }

    out_json_path = os.path.join(output_dir, 'experiment_31_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha

    print(f"\n[exp31] DONE", flush=True)
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
    parser.add_argument('--thresholds', type=str, default='0.001,0.002,0.005,0.01')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp31_out')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    hidden_layers_opts = [int(h) for h in args.hidden_layers.split(',')]
    omegas = [float(o) for o in args.omegas.split(',')]
    thresholds = [float(t) for t in args.thresholds.split(',')]

    run_experiment(
        seeds=seeds, num_images=args.num_images, image_size=args.size,
        hidden_features=args.hidden_features, hidden_layers_options=hidden_layers_opts,
        omegas=omegas, epochs=args.epochs, lr=args.lr, thresholds=thresholds,
        K=args.K, output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
