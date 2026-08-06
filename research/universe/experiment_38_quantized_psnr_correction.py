"""
experiment_38_quantized_psnr_correction.py
============================================
Experiment 38 — Methodological Correction: Measure PSNR AFTER quantization.

BUG IDENTIFIED: In all experiments 30-37, PSNR was measured on the float32
model BEFORE KMeans clustering, while size was measured AFTER clustering +
entropy coding. This mixes two different versions of the model:
  - PSNR: pre-quantization (best case)
  - Size: post-quantization (compressed)

The true rate-distortion point requires measuring PSNR on the QUANTIZED
model (weights reconstructed from codebook[indices], reloaded into model).

This experiment:
1. Trains SIREN (same config as Exp 37: single-omega ω=30, hl=2, 256×256 real photos)
2. Records pre-quantization PSNR (what Exp 37 reported)
3. Clusters weights with KMeans K=50
4. Reconstructs quantized weights: codebook[indices]
5. Reloads quantized weights into model
6. Measures POST-quantization PSNR (the REAL number)
7. Compares against JPEG/WebP at matched byte budget

If the post-quantization PSNR drops significantly, the Exp 37 "victory"
over JPEG/WebP may disappear — and that would be the honest result.

ANTI-FABRICATION: same protocol. Output real, SHA-256, no tuning.
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
from experiment_37_real_photo_parity import (
    load_real_photos, train_siren_one_image, get_model_weights_flat,
    jpeg_compress_at_size, webp_compress_at_size,
    jpeg_decompress, webp_decompress, compute_psnr,
)


# ---------------------------------------------------------------------------
# THE KEY FUNCTION: dequantize and reload
# ---------------------------------------------------------------------------

def dequantize_and_reload(model: nn.Module, cluster_result: Dict,
                           image_index: int) -> float:
    """
    Reconstruct quantized weights from codebook[indices], reload into model,
    and return nothing (model is modified in-place).

    Steps:
    a. Get codebook and indices_per_image from cluster_result
    b. Reconstruct: quantized_weights = codebook[indices]
    c. Reload into model parameters (matching the original flat layout)
    """
    codebook = cluster_result['codebook']  # (K,) float32
    indices = cluster_result['indices_per_image'][image_index]  # (N,) int32

    # Reconstruct quantized weights
    quantized_weights = codebook[indices]  # (N,) float32

    # Reload into model parameters
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            shape = p.shape
            chunk = quantized_weights[offset:offset + n].reshape(shape)
            p.copy_(torch.from_numpy(chunk.astype(np.float32)))
            offset += n


def evaluate_model_psnr(model: nn.Module, img: np.ndarray) -> float:
    """Evaluate PSNR of model on image (forward pass + compare)."""
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


def run_experiment(seeds: List[int], image_size: int, hidden_features: int,
                    hidden_layers: int, omega: float, epochs: int, lr: float,
                    K: int, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[exp38] loading real photos at {image_size}x{image_size}...", flush=True)
    images, names = load_real_photos(image_size)
    print(f"[exp38] loaded {len(images)} real photos: {names}", flush=True)
    print(f"[exp38] CORRECTION: measuring PSNR AFTER KMeans quantization", flush=True)

    all_runs = []
    for seed in seeds:
        ckpt_path = os.path.join(output_dir, f'ckpt_seed{seed}.json')
        if os.path.exists(ckpt_path):
            print(f"\n[exp38] LOADING checkpoint seed={seed}", flush=True)
            with open(ckpt_path) as f:
                run_result = json.load(f)
            all_runs.append(run_result)
            print(f"  CACHED", flush=True)
            continue

        print(f"\n[exp38] === seed={seed} ===", flush=True)
        t0 = time.time()

        # 1. Train SIREN on each image
        models = []
        psnr_pre_quant = []
        for i, img in enumerate(images):
            model, psnr, t_train = train_siren_one_image(
                img, hidden_features=hidden_features, hidden_layers=hidden_layers,
                omega=omega, epochs=epochs, lr=lr, seed=seed,
            )
            models.append(model)
            psnr_pre_quant.append(psnr)
            print(f"  SIREN {names[i]}: pre-quant PSNR={psnr:.2f} dB", flush=True)

        # 2. Cluster weights with KMeans K=50
        weights_per_image = [get_model_weights_flat(m).astype(np.float32) for m in models]
        cluster_result = hierarchical_kmeans_cluster(weights_per_image, K=K, seed=seed)

        # 3. Entropy code the indices (for size calculation)
        entropy_result = entropy_code_indices(cluster_result)
        per_img_sizes = entropy_result['coded_sizes_per_image']
        codebook_share = entropy_result['total_bytes_codebook'] / len(per_img_sizes)
        siren_sizes = [s + codebook_share for s in per_img_sizes]

        # 4. THE CORRECTION: dequantize and reload, then measure REAL PSNR
        psnr_post_quant = []
        for i, (model, img) in enumerate(zip(models, images)):
            dequantize_and_reload(model, cluster_result, image_index=i)
            psnr_post = evaluate_model_psnr(model, img)
            psnr_post_quant.append(psnr_post)
            print(f"  {names[i]}: pre={psnr_pre_quant[i]:.2f} dB → post={psnr_post:.2f} dB "
                  f"(drop={psnr_pre_quant[i]-psnr_post:.2f} dB)", flush=True)

        # 5. JPEG and WebP at matched byte budget (using same siren_sizes as target)
        jpeg_psnrs = []
        jpeg_sizes = []
        webp_psnrs = []
        webp_sizes = []

        for i, img in enumerate(images):
            target_size = int(siren_sizes[i])

            jpeg_payload, jpeg_actual_size = jpeg_compress_at_size(img, target_size)
            jpeg_recon = jpeg_decompress(jpeg_payload, img.shape)
            jpeg_psnr = compute_psnr(img, jpeg_recon)
            jpeg_psnrs.append(jpeg_psnr)
            jpeg_sizes.append(jpeg_actual_size)

            webp_payload, webp_actual_size = webp_compress_at_size(img, target_size)
            webp_recon = webp_decompress(webp_payload, img.shape)
            webp_psnr = compute_psnr(img, webp_recon)
            webp_psnrs.append(webp_psnr)
            webp_sizes.append(webp_actual_size)

            print(f"  {names[i]} (target {target_size:.0f} B): "
                  f"SIREN_post={psnr_post_quant[i]:.2f} dB, "
                  f"JPEG={jpeg_psnr:.2f} dB, WebP={webp_psnr:.2f} dB", flush=True)

        # Save weights for SHA-256
        weights_payload = bytearray()
        for m in models:
            w = get_model_weights_flat(m).astype(np.float32)
            weights_payload += w.tobytes()
        weights_file = os.path.join(output_dir, f'exp38_weights_seed{seed}.bin')
        with open(weights_file, 'wb') as f:
            f.write(bytes(weights_payload))
        with open(weights_file, 'rb') as f:
            weights_sha = hashlib.sha256(f.read()).hexdigest()

        run_result = {
            'seed': seed,
            'per_image': [
                {
                    'name': names[i],
                    'psnr_pre_quant_db': psnr_pre_quant[i],
                    'psnr_post_quant_db': psnr_post_quant[i],
                    'psnr_drop_db': psnr_pre_quant[i] - psnr_post_quant[i],
                    'siren_size_bytes': siren_sizes[i],
                    'jpeg_psnr_db': jpeg_psnrs[i],
                    'jpeg_size_bytes': jpeg_sizes[i],
                    'webp_psnr_db': webp_psnrs[i],
                    'webp_size_bytes': webp_sizes[i],
                }
                for i in range(len(images))
            ],
            'mean_psnr_pre_quant': float(np.mean(psnr_pre_quant)),
            'mean_psnr_post_quant': float(np.mean(psnr_post_quant)),
            'mean_psnr_drop': float(np.mean(psnr_pre_quant) - np.mean(psnr_post_quant)),
            'mean_siren_size': float(np.mean(siren_sizes)),
            'mean_jpeg_psnr': float(np.mean(jpeg_psnrs)),
            'mean_jpeg_size': float(np.mean(jpeg_sizes)),
            'mean_webp_psnr': float(np.mean(webp_psnrs)),
            'mean_webp_size': float(np.mean(webp_sizes)),
            'total_time_s': time.time() - t0,
            'weights_file': weights_file,
            'weights_sha256': weights_sha,
        }

        with open(ckpt_path, 'w') as f:
            json.dump(run_result, f, indent=2, default=str)
        all_runs.append(run_result)

        print(f"\n  MEAN (seed={seed}):", flush=True)
        print(f"    SIREN pre-quant:  {run_result['mean_psnr_pre_quant']:.2f} dB", flush=True)
        print(f"    SIREN post-quant: {run_result['mean_psnr_post_quant']:.2f} dB", flush=True)
        print(f"    PSNR drop:        {run_result['mean_psnr_drop']:.2f} dB", flush=True)
        print(f"    JPEG:             {run_result['mean_jpeg_psnr']:.2f} dB, {run_result['mean_jpeg_size']:.0f} B", flush=True)
        print(f"    WebP:             {run_result['mean_webp_psnr']:.2f} dB, {run_result['mean_webp_size']:.0f} B", flush=True)

    # Aggregate across seeds
    pre_psnrs = np.array([r['mean_psnr_pre_quant'] for r in all_runs])
    post_psnrs = np.array([r['mean_psnr_post_quant'] for r in all_runs])
    drops = np.array([r['mean_psnr_drop'] for r in all_runs])
    siren_sizes = np.array([r['mean_siren_size'] for r in all_runs])
    jpeg_psnrs = np.array([r['mean_jpeg_psnr'] for r in all_runs])
    webp_psnrs = np.array([r['mean_webp_psnr'] for r in all_runs])

    aggregated = {
        'siren_pre_quant': {
            'mean_psnr_db': float(pre_psnrs.mean()), 'std_psnr_db': float(pre_psnrs.std()),
        },
        'siren_post_quant': {
            'mean_psnr_db': float(post_psnrs.mean()), 'std_psnr_db': float(post_psnrs.std()),
        },
        'psnr_drop': {
            'mean_db': float(drops.mean()), 'std_db': float(drops.std()),
        },
        'siren_size': {
            'mean_bytes': float(siren_sizes.mean()), 'std_bytes': float(siren_sizes.std()),
        },
        'jpeg': {
            'mean_psnr_db': float(jpeg_psnrs.mean()), 'std_psnr_db': float(jpeg_psnrs.std()),
        },
        'webp': {
            'mean_psnr_db': float(webp_psnrs.mean()), 'std_psnr_db': float(webp_psnrs.std()),
        },
    }

    print(f"\n[exp38] AGGREGATED across {len(all_runs)} seeds:", flush=True)
    print(f"  SIREN pre-quant:  {aggregated['siren_pre_quant']['mean_psnr_db']:.2f} ± {aggregated['siren_pre_quant']['std_psnr_db']:.2f} dB", flush=True)
    print(f"  SIREN post-quant: {aggregated['siren_post_quant']['mean_psnr_db']:.2f} ± {aggregated['siren_post_quant']['std_psnr_db']:.2f} dB", flush=True)
    print(f"  PSNR drop:        {aggregated['psnr_drop']['mean_db']:.2f} ± {aggregated['psnr_drop']['std_db']:.2f} dB", flush=True)
    print(f"  SIREN size:       {aggregated['siren_size']['mean_bytes']:.0f} B", flush=True)
    print(f"  JPEG:             {aggregated['jpeg']['mean_psnr_db']:.2f} ± {aggregated['jpeg']['std_psnr_db']:.2f} dB", flush=True)
    print(f"  WebP:             {aggregated['webp']['mean_psnr_db']:.2f} ± {aggregated['webp']['std_psnr_db']:.2f} dB", flush=True)

    # Determine if SIREN still wins
    siren_post = aggregated['siren_post_quant']['mean_psnr_db']
    jpeg_mean = aggregated['jpeg']['mean_psnr_db']
    webp_mean = aggregated['webp']['mean_psnr_db']

    diff_jpeg = siren_post - jpeg_mean
    diff_webp = siren_post - webp_mean

    if siren_post > jpeg_mean and siren_post > webp_mean:
        winner = "SIREN+entropy (post-quant) STILL beats JPEG and WebP"
        conclusion = "POSITIVE — the Exp 37 victory HOLDS after correction"
    elif siren_post > jpeg_mean or siren_post > webp_mean:
        winner = "mixed (beats one but not the other)"
        conclusion = "MIXED — partial victory after correction"
    else:
        winner = "JPEG/WebP (both beat SIREN+entropy post-quant)"
        conclusion = "NEGATIVE — the Exp 37 victory DISAPPEARS after correction"

    comparison = {
        'siren_post_minus_jpeg': float(diff_jpeg),
        'siren_post_minus_webp': float(diff_webp),
        'winner': winner,
        'conclusion': conclusion,
        'exp37_pre_quant_psnr': float(pre_psnrs.mean()),
        'exp38_post_quant_psnr': float(post_psnrs.mean()),
        'correction_drop_db': float(drops.mean()),
    }

    print(f"\n[exp38] CORRECTED PARITY COMPARISON:", flush=True)
    print(f"  SIREN post-quant - JPEG: {diff_jpeg:+.2f} dB", flush=True)
    print(f"  SIREN post-quant - WebP: {diff_webp:+.2f} dB", flush=True)
    print(f"  WINNER: {winner}", flush=True)
    print(f"  CONCLUSION: {conclusion}", flush=True)
    print(f"  Correction: pre-quant {pre_psnrs.mean():.2f} dB → post-quant {post_psnrs.mean():.2f} dB (drop {drops.mean():.2f} dB)", flush=True)

    output = {
        'experiment': 'experiment_38_quantized_psnr_correction',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'Measuring PSNR AFTER KMeans quantization (not before) will reveal '
                       'the true rate-distortion point. If the PSNR drop is large enough, '
                       'the Exp 37 victory over JPEG/WebP may disappear.',
        'bug_description': 'In Exp 30-37, PSNR was measured on float32 model BEFORE '
                           'KMeans clustering, while size was measured AFTER clustering. '
                           'This experiment measures PSNR on the quantized model (codebook[indices] '
                           'reloaded into model) for the first time.',
        'config': {
            'image_size': image_size,
            'hidden_features': hidden_features,
            'hidden_layers': hidden_layers,
            'omega': omega,
            'epochs': epochs,
            'lr': lr,
            'K': K,
            'seeds': seeds,
        },
        'all_runs': all_runs,
        'aggregated': aggregated,
        'corrected_comparison': comparison,
    }

    out_json_path = os.path.join(output_dir, 'experiment_38_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha

    print(f"\n[exp38] DONE", flush=True)
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
                        default='/home/z/my-project/research/universe/_exp38_out')
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
