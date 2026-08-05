"""
experiment_33_coin_rd_fine.py
================================
Experiment 33 — COIN Rate-Distortion Curve with Fine Bit Resolution.

Experiment 32 measured COIN's RD curve at 4 points (16, 8, 4, 2 bits) and
found BHUH wins 7/8 configs. But 4 points is too coarse — the "matched PSNR"
comparison used the closest available point, which may not be a true match.

This experiment fills in the curve with intermediate bit depths:
  - 12 bits (between 16 and 8)
  - 10 bits (between 12 and 8)
  - 6 bits  (between 8 and 4)
  - 3 bits  (between 4 and 2)

Combined with Exp 32's data (16, 8, 4, 2), we'll have 8 points on the COIN
RD curve, enabling proper interpolation for true PSNR matching.

ANTI-FABRICATION: same protocol as Exp 29-32.
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

torch.set_num_threads(2)
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coin_baseline_exp29 import (
    load_scikit_images, normalize_to_pm1, denormalize_from_pm1,
    train_coin_one_image, CoinMLP,
)
from experiment_32_coin_rd_curve import (
    quantize_weights_minmax, get_model_weights_flat, set_model_weights_flat,
    compute_size_bytes, evaluate_model_psnr,
)


def run_coin_at_quantization(images: np.ndarray, bits: int, hidden_features: int,
                               hidden_layers: int, omega: float, epochs: int,
                               lr: float, seed: int, output_dir: str) -> Dict:
    """Run COIN baseline with weights quantized to `bits` bits per weight."""
    import gc

    psnrs_pre_quant = []
    psnrs_post_quant = []
    sizes = []
    train_times = []

    for i, img in enumerate(images):
        model, psnr_pre, t_train = train_coin_one_image(
            img, hidden_features=hidden_features, hidden_layers=hidden_layers,
            omega=omega, epochs=epochs, lr=lr, seed=seed,
        )
        weights = get_model_weights_flat(model)
        quantized_weights, w_min, w_max = quantize_weights_minmax(weights, bits)
        set_model_weights_flat(model, quantized_weights)
        psnr_post = evaluate_model_psnr(model, img)
        size = compute_size_bytes(weights, bits)

        psnrs_pre_quant.append(psnr_pre)
        psnrs_post_quant.append(psnr_post)
        sizes.append(size)
        train_times.append(t_train)

        del model
        gc.collect()

    psnr_pre_arr = np.array(psnrs_pre_quant)
    psnr_post_arr = np.array(psnrs_post_quant)
    size_arr = np.array(sizes)

    return {
        'bits': bits,
        'seed': seed,
        'mean_psnr_pre_quant_db': float(psnr_pre_arr.mean()),
        'std_psnr_pre_quant_db': float(psnr_pre_arr.std()),
        'mean_psnr_post_quant_db': float(psnr_post_arr.mean()),
        'std_psnr_post_quant_db': float(psnr_post_arr.std()),
        'mean_size_bytes': float(size_arr.mean()),
        'std_size_bytes': float(size_arr.std()),
        'mean_train_time_s': float(np.mean(train_times)),
    }


def run_experiment(seeds: List[int], num_images: int, image_size: int,
                    hidden_features: int, hidden_layers: int, omega: float,
                    epochs: int, lr: float, bits_options: List[int],
                    output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    import gc

    print(f"[exp33] loading {num_images} images at {image_size}x{image_size}...", flush=True)
    images, names = load_scikit_images(num_images, image_size)
    print(f"[exp33] loaded {len(images)} images", flush=True)

    all_runs = []
    for bits in bits_options:
        for seed in seeds:
            ckpt_path = os.path.join(output_dir, f'ckpt_bits{bits}_seed{seed}.json')
            if os.path.exists(ckpt_path):
                print(f"\n[exp33] LOADING checkpoint bits={bits} seed={seed}", flush=True)
                with open(ckpt_path) as f:
                    run_result = json.load(f)
                all_runs.append(run_result)
                print(f"  CACHED: PSNR={run_result['mean_psnr_post_quant_db']:.4f} dB, "
                      f"size={run_result['mean_size_bytes']:.1f} B", flush=True)
                continue

            print(f"\n[exp33] === bits={bits}, seed={seed} ===", flush=True)
            t0 = time.time()
            run_result = run_coin_at_quantization(
                images=images, bits=bits, hidden_features=hidden_features,
                hidden_layers=hidden_layers, omega=omega, epochs=epochs,
                lr=lr, seed=seed, output_dir=output_dir,
            )
            run_result['total_time_s'] = time.time() - t0

            with open(ckpt_path, 'w') as f:
                json.dump(run_result, f, indent=2, default=str)
            all_runs.append(run_result)

            print(f"  PSNR pre-quant  = {run_result['mean_psnr_pre_quant_db']:.4f} ± "
                  f"{run_result['std_psnr_pre_quant_db']:.4f} dB", flush=True)
            print(f"  PSNR post-quant = {run_result['mean_psnr_post_quant_db']:.4f} ± "
                  f"{run_result['std_psnr_post_quant_db']:.4f} dB", flush=True)
            print(f"  Size per image  = {run_result['mean_size_bytes']:.1f} ± "
                  f"{run_result['std_size_bytes']:.1f} B", flush=True)

    # Aggregate across seeds for each bits level
    aggregated = []
    for bits in bits_options:
        runs = [r for r in all_runs if r['bits'] == bits]
        if not runs:
            continue
        psnrs = np.array([r['mean_psnr_post_quant_db'] for r in runs])
        sizes = np.array([r['mean_size_bytes'] for r in runs])
        agg = {
            'bits': bits,
            'n_seeds': len(runs),
            'seeds': [r['seed'] for r in runs],
            'mean_psnr_db': float(psnrs.mean()),
            'std_psnr_db': float(psnrs.std()),
            'mean_size_bytes': float(sizes.mean()),
            'std_size_bytes': float(sizes.std()),
        }
        aggregated.append(agg)
        print(f"\n[exp33] AGGREGATED bits={bits}:", flush=True)
        print(f"  PSNR  = {agg['mean_psnr_db']:.4f} ± {agg['std_psnr_db']:.4f} dB", flush=True)
        print(f"  Size  = {agg['mean_size_bytes']:.1f} ± {agg['std_size_bytes']:.1f} B", flush=True)

    # Merge with Exp 32 data (bits 16, 8, 4, 2) for the full RD curve
    exp32_path = os.path.join(os.path.dirname(output_dir), '_exp32_out', 'experiment_32_results.json')
    full_rd_curve = list(aggregated)
    if os.path.exists(exp32_path):
        with open(exp32_path) as f:
            exp32_data = json.load(f)
        for agg in exp32_data['aggregated']:
            # Only add if not already in our new data
            if not any(a['bits'] == agg['bits'] for a in full_rd_curve):
                full_rd_curve.append(agg)
        full_rd_curve.sort(key=lambda x: x['bits'], reverse=True)
        print(f"\n[exp33] Merged with Exp 32 data — full RD curve has {len(full_rd_curve)} points", flush=True)
    else:
        print(f"\n[exp33] WARNING: Exp 32 data not found, using only new data", flush=True)

    print(f"\n[exp33] FULL COIN RD CURVE (8 points):", flush=True)
    for point in full_rd_curve:
        print(f"  bits={point['bits']:2d}: PSNR={point['mean_psnr_db']:7.2f} dB, "
              f"size={point['mean_size_bytes']:8.1f} B", flush=True)

    # Now do proper interpolated PSNR matching with BHUH from Exp 31
    exp31_path = os.path.join(os.path.dirname(output_dir), '_exp31_out', 'experiment_31_results.json')
    rd_comparison = []
    if os.path.exists(exp31_path) and len(full_rd_curve) >= 2:
        with open(exp31_path) as f:
            bhuH_data = json.load(f)

        # Sort RD curve by PSNR for interpolation
        rd_by_psnr = sorted(full_rd_curve, key=lambda x: x['mean_psnr_db'])
        psnrs_curve = np.array([p['mean_psnr_db'] for p in rd_by_psnr])
        sizes_curve = np.array([p['mean_size_bytes'] for p in rd_by_psnr])
        bits_curve = np.array([p['bits'] for p in rd_by_psnr])

        for bhuH_agg in bhuH_data['aggregated']:
            bhuH_psnr = bhuH_agg['mean_psnr_db']
            bhuH_size = bhuH_agg['mean_size_bytes']

            # Interpolate COIN size at BHUH's exact PSNR
            # np.interp: if bhuH_psnr is outside [min, max], returns edge values
            interpolated_coin_size = float(np.interp(bhuH_psnr, psnrs_curve, sizes_curve))

            # Also find nearest measured point for comparison
            nearest_idx = int(np.argmin(np.abs(psnrs_curve - bhuH_psnr)))
            nearest = rd_by_psnr[nearest_idx]

            # Linear interpolation between two surrounding points
            if bhuH_psnr <= psnrs_curve.min() or bhuH_psnr >= psnrs_curve.max():
                interp_method = "extrapolation (outside curve range)"
            else:
                # Find the two points that surround bhuH_psnr
                upper_idx = int(np.searchsorted(psnrs_curve, bhuH_psnr))
                lower_idx = upper_idx - 1
                lower_psnr = psnrs_curve[lower_idx]
                upper_psnr = psnrs_curve[upper_idx]
                lower_size = sizes_curve[lower_idx]
                upper_size = sizes_curve[upper_idx]
                # Linear interpolation in log space (size is roughly exponential in PSNR)
                if upper_psnr > lower_psnr:
                    t = (bhuH_psnr - lower_psnr) / (upper_psnr - lower_psnr)
                    interp_size = lower_size + t * (upper_size - lower_size)
                    interp_method = f"linear interp between bits={bits_curve[lower_idx]} and bits={bits_curve[upper_idx]}"
                else:
                    interp_size = interpolated_coin_size
                    interp_method = "np.interp fallback"
                interpolated_coin_size = float(interp_size)

            ratio = bhuH_size / max(1, interpolated_coin_size)
            winner = "COIN" if interpolated_coin_size <= bhuH_size else "BHUH"

            rd_comparison.append({
                'bhuH_config': f"hl={bhuH_agg['hidden_layers']} thr={bhuH_agg['threshold']}",
                'bhuH_psnr_db': bhuH_psnr,
                'bhuH_size_bytes': bhuH_size,
                'coin_interpolated_size_bytes': interpolated_coin_size,
                'coin_interp_method': interp_method,
                'coin_nearest_measured': {
                    'bits': nearest['bits'],
                    'psnr_db': nearest['mean_psnr_db'],
                    'size_bytes': nearest['mean_size_bytes'],
                },
                'size_ratio_bhuH_over_coin': ratio,
                'winner': winner,
                'winner_smaller_by_x': ratio if winner == "COIN" else 1.0 / ratio,
            })
            print(f"\n[exp33] RD COMPARISON {bhuH_agg['hidden_layers']}/thr={bhuH_agg['threshold']}:", flush=True)
            print(f"  BHUH: {bhuH_size:.0f} B @ {bhuH_psnr:.2f} dB", flush=True)
            print(f"  COIN (interpolated): {interpolated_coin_size:.0f} B @ {bhuH_psnr:.2f} dB", flush=True)
            print(f"    method: {interp_method}", flush=True)
            print(f"  COIN (nearest measured): bits={nearest['bits']}, "
                  f"{nearest['mean_size_bytes']:.0f} B @ {nearest['mean_psnr_db']:.2f} dB", flush=True)
            print(f"  WINNER: {winner} (smaller by {ratio if winner == 'COIN' else 1.0/ratio:.2f}x)", flush=True)

    output = {
        'experiment': 'experiment_33_coin_rd_fine',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'Filling in the COIN RD curve with intermediate bit depths (12, 10, 6, 3) '
                       'will either confirm or refute the Exp 32 finding that BHUH wins 7/8 configs.',
        'config': {
            'hidden_features': hidden_features,
            'hidden_layers': hidden_layers,
            'omega': omega,
            'epochs': epochs,
            'lr': lr,
            'num_images': num_images,
            'image_size': image_size,
            'seeds': seeds,
            'bits_options_new': bits_options,
        },
        'new_runs': all_runs,
        'new_aggregated': aggregated,
        'full_rd_curve_merged_with_exp32': full_rd_curve,
        'rd_comparison_interpolated': rd_comparison,
    }

    out_json_path = os.path.join(output_dir, 'experiment_33_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha

    print(f"\n[exp33] DONE", flush=True)
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
    parser.add_argument('--bits', type=str, default='12,10,6,3')
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp33_out')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    bits_options = [int(b) for b in args.bits.split(',')]

    run_experiment(
        seeds=seeds, num_images=args.num_images, image_size=args.size,
        hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
        omega=args.omega, epochs=args.epochs, lr=args.lr, bits_options=bits_options,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
