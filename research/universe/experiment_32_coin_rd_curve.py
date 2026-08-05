"""
experiment_32_coin_rd_curve.py
================================
Experiment 32 — COIN Rate-Distortion Curve (Measured, Not Estimated).

Experiment 31 found that BHUH loses to COIN on byte parity at every pruning
threshold, but that comparison used a 6 dB/bit HEURISTIC to estimate COIN's
size at matched PSNR. This experiment MEASURES the actual COIN rate-distortion
curve by running COIN with multiple weight quantization levels:
  - float32 (baseline, no quantization)
  - float16 (standard, used in Exp 29-31)
  - float8  (8-bit quantization)
  - float4  (4-bit quantization)
  - float2  (2-bit quantization)

For each quantization level, we measure:
  - PSNR (mean ± std across 3 seeds)
  - Size in bytes (mean ± std)
  - Reconstruction quality

Then we compare BHUH (from Exp 31) to COIN at matched PSNR points using
the MEASURED curve, not the heuristic.

ANTI-FABRICATION: same protocol as Exp 29-31.
- 3 seeds (42, 123, 2024), 30 images, 300 epochs, hl=5 (standard COIN config)
- JSON output to stdout with SHA-256
- No hyperparameter tuning
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


# ---------------------------------------------------------------------------
# Weight quantization functions
# ---------------------------------------------------------------------------

def quantize_weights_minmax(weights: np.ndarray, bits: int) -> Tuple[np.ndarray, float, float]:
    """
    Quantize weights to `bits` bits per weight using min-max quantization.
    Returns (quantized_weights, w_min, w_max).
    """
    if bits >= 32:
        return weights.astype(np.float32), 0.0, 0.0
    w_min = float(weights.min())
    w_max = float(weights.max())
    if w_max - w_min < 1e-12:
        return np.zeros_like(weights), w_min, w_max
    n_levels = (1 << bits) - 1  # 2^bits - 1
    # Normalize to [0, n_levels], round, then scale back
    scaled = (weights - w_min) / (w_max - w_min) * n_levels
    quantized = np.round(scaled).astype(np.int32)
    # De-quantize back to float
    dequant = w_min + quantized.astype(np.float32) / n_levels * (w_max - w_min)
    return dequant, w_min, w_max


def get_model_weights_flat(model: nn.Module) -> np.ndarray:
    """Get all model weights as a single flat numpy array."""
    parts = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
    return np.concatenate(parts)


def set_model_weights_flat(model: nn.Module, flat_weights: np.ndarray):
    """Set model weights from a flat numpy array (inverse of get_model_weights_flat)."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            shape = p.shape
            chunk = flat_weights[offset:offset + n].reshape(shape)
            p.copy_(torch.from_numpy(chunk.astype(np.float32)))
            offset += n


def compute_size_bytes(weights: np.ndarray, bits: int) -> int:
    """
    Compute the size in bytes to store weights at `bits` bits per weight.
    Includes 8 bytes overhead for min/max float32 values.
    """
    n = weights.size
    # bits per weight + 8 bytes for (min, max) float32
    return (n * bits + 7) // 8 + 8


# ---------------------------------------------------------------------------
# PSNR evaluation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Run COIN at a specific quantization level
# ---------------------------------------------------------------------------

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
        # Train model
        model, psnr_pre, t_train = train_coin_one_image(
            img, hidden_features=hidden_features, hidden_layers=hidden_layers,
            omega=omega, epochs=epochs, lr=lr, seed=seed,
        )

        # Get weights, quantize, set back
        weights = get_model_weights_flat(model)
        quantized_weights, w_min, w_max = quantize_weights_minmax(weights, bits)
        set_model_weights_flat(model, quantized_weights)

        # Evaluate PSNR after quantization
        psnr_post = evaluate_model_psnr(model, img)

        # Compute size
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


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(seeds: List[int], num_images: int, image_size: int,
                    hidden_features: int, hidden_layers: int, omega: float,
                    epochs: int, lr: float, bits_options: List[int],
                    output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    import gc

    print(f"[exp32] loading {num_images} images at {image_size}x{image_size}...", flush=True)
    images, names = load_scikit_images(num_images, image_size)
    print(f"[exp32] loaded {len(images)} images", flush=True)

    # Run for each (bits, seed) combination
    all_runs = []
    for bits in bits_options:
        for seed in seeds:
            ckpt_path = os.path.join(output_dir, f'ckpt_bits{bits}_seed{seed}.json')
            if os.path.exists(ckpt_path):
                print(f"\n[exp32] LOADING checkpoint bits={bits} seed={seed}", flush=True)
                with open(ckpt_path) as f:
                    run_result = json.load(f)
                all_runs.append(run_result)
                print(f"  CACHED: PSNR={run_result['mean_psnr_post_quant_db']:.4f} dB, "
                      f"size={run_result['mean_size_bytes']:.1f} B", flush=True)
                continue

            print(f"\n[exp32] === bits={bits}, seed={seed} ===", flush=True)
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
        print(f"\n[exp32] AGGREGATED bits={bits}:", flush=True)
        print(f"  PSNR  = {agg['mean_psnr_db']:.4f} ± {agg['std_psnr_db']:.4f} dB", flush=True)
        print(f"  Size  = {agg['mean_size_bytes']:.1f} ± {agg['std_size_bytes']:.1f} B", flush=True)

    # Load BHUH results from Exp 31 for comparison
    exp31_path = os.path.join(os.path.dirname(output_dir), '_exp31_out', 'experiment_31_results.json')
    bhuH_results = None
    if os.path.exists(exp31_path):
        with open(exp31_path) as f:
            bhuH_results = json.load(f)
        print(f"\n[exp32] Loaded BHUH results from Exp 31 for comparison", flush=True)
    else:
        print(f"\n[exp32] WARNING: Exp 31 results not found at {exp31_path}", flush=True)
        print(f"         BHUH comparison will be skipped.", flush=True)

    # Build measured RD curve comparison
    # For each BHUH config (hl, threshold), find the COIN config that achieves
    # the closest PSNR and compare byte sizes directly.
    rd_comparison = []
    if bhuH_results:
        coin_curve = [(a['mean_psnr_db'], a['mean_size_bytes'], a['bits'])
                       for a in aggregated]
        coin_curve.sort(key=lambda x: x[0])  # sort by PSNR

        for bhuH_agg in bhuH_results['aggregated']:
            bhuH_psnr = bhuH_agg['mean_psnr_db']
            bhuH_size = bhuH_agg['mean_size_bytes']

            # Find COIN config with closest PSNR
            best_coin = None
            best_psnr_diff = float('inf')
            for coin_psnr, coin_size, coin_bits in coin_curve:
                diff = abs(coin_psnr - bhuH_psnr)
                if diff < best_psnr_diff:
                    best_psnr_diff = diff
                    best_coin = (coin_psnr, coin_size, coin_bits)

            if best_coin:
                coin_psnr, coin_size, coin_bits = best_coin
                ratio = bhuH_size / max(1, coin_size)
                winner = "COIN" if coin_size <= bhuH_size else "BHUH"
                rd_comparison.append({
                    'bhuH_config': f"hl={bhuH_agg['hidden_layers']} thr={bhuH_agg['threshold']}",
                    'bhuH_psnr_db': bhuH_psnr,
                    'bhuH_size_bytes': bhuH_size,
                    'coin_bits': coin_bits,
                    'coin_psnr_db': coin_psnr,
                    'coin_size_bytes': coin_size,
                    'psnr_diff_db': bhuH_psnr - coin_psnr,
                    'size_ratio_bhuH_over_coin': ratio,
                    'winner': winner,
                    'winner_smaller_by_x': ratio if winner == "COIN" else 1.0 / ratio,
                })
                print(f"\n[exp32] RD COMPARISON {bhuH_agg['hidden_layers']}/thr={bhuH_agg['threshold']}:", flush=True)
                print(f"  BHUH: {bhuH_size:.0f} B @ {bhuH_psnr:.2f} dB", flush=True)
                print(f"  COIN (bits={coin_bits}): {coin_size:.0f} B @ {coin_psnr:.2f} dB", flush=True)
                print(f"  PSNR diff: {bhuH_psnr - coin_psnr:+.2f} dB", flush=True)
                print(f"  Size ratio (BHUH/COIN): {ratio:.4f}x", flush=True)
                print(f"  WINNER: {winner} (smaller by {ratio if winner == 'COIN' else 1.0/ratio:.2f}x)", flush=True)

    output = {
        'experiment': 'experiment_32_coin_rd_curve',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'Measuring the actual COIN rate-distortion curve will confirm '
                       'whether BHUH loses to COIN at every PSNR level (as estimated '
                       'in Exp 31) or if BHUH wins in some region.',
        'config': {
            'hidden_features': hidden_features,
            'hidden_layers': hidden_layers,
            'omega': omega,
            'epochs': epochs,
            'lr': lr,
            'num_images': num_images,
            'image_size': image_size,
            'seeds': seeds,
            'bits_options': bits_options,
        },
        'all_runs': all_runs,
        'aggregated': aggregated,
        'rd_comparison_with_bhuH': rd_comparison,
    }

    out_json_path = os.path.join(output_dir, 'experiment_32_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha

    print(f"\n[exp32] DONE", flush=True)
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
    parser.add_argument('--bits', type=str, default='16,8,4,2')
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp32_out')
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
