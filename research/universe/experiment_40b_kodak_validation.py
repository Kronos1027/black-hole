"""
experiment_40b_kodak_validation.py
====================================
Experiment 40-B — Kodak Dataset Statistical Validation.

Exp 40 used only 3 images. This experiment validates on 12 Kodak images
(kodim01-12, the maximum available in the repo) at 256×256 grayscale.

For each image:
1. Compute Variance of Laplacian (complexity metric)
2. Train SIREN+QAT (same config as Exp 39/40)
3. Measure post-quant PSNR, SSIM, encoding time, decoding time
4. Compress JPEG/WebP/AVIF to matched ~6562 B
5. Measure their PSNR, SSIM

Output: correlation table between Laplacian variance and PSNR diff (SIREN - AVIF).

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
from coin_baseline_exp29 import normalize_to_pm1, denormalize_from_pm1, CoinMLP
from experiment_37_real_photo_parity import compute_psnr
from experiment_39_qat_ste import (
    train_qat_siren, evaluate_psnr_pre_quant, evaluate_psnr_post_quant,
    compute_quantized_size,
)
from experiment_40_byte_parity_battlefield import (
    compute_ssim, jpeg_at_exact_size, webp_at_exact_size,
    avif_at_exact_size, decompress_grayscale,
)


def load_kodak_images(size: int = 256) -> Tuple[np.ndarray, List[str]]:
    """Load 12 Kodak images, convert to grayscale, resize to 256×256."""
    kodak_dir = '/home/z/my-project/tests/kodak'
    images = []
    names = []
    for i in range(1, 13):
        path = os.path.join(kodak_dir, f'kodim{i:02d}.png')
        if not os.path.exists(path):
            continue
        img = Image.open(path)
        if img.mode != 'L':
            img = img.convert('L')
        img = img.resize((size, size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        images.append(arr)
        names.append(f'kodim{i:02d}')
    return np.stack(images), names


def variance_of_laplacian(img: np.ndarray) -> float:
    """Compute variance of Laplacian (image complexity/sharpness metric)."""
    # Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Apply convolution (simple 2D conv)
    from scipy.signal import convolve2d
    laplacian = convolve2d(img, kernel, mode='valid', boundary='fill', fillvalue=0)
    return float(laplacian.var())


def measure_decode_time(model: nn.Module, img: np.ndarray, repeats: int = 5) -> float:
    """Measure forward pass time (decode time) in seconds."""
    H, W = img.shape
    ys, xs = np.meshgrid(np.linspace(-1, 1, H, dtype=np.float32),
                          np.linspace(-1, 1, W, dtype=np.float32), indexing='ij')
    coords = np.stack([xs.flatten(), ys.flatten()], axis=1)
    coords_t = torch.tensor(coords)
    
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.time()
            _ = model(coords_t)
            times.append(time.time() - t0)
    return float(np.mean(times))


def run_experiment(seeds: List[int], image_size: int, hidden_features: int,
                    hidden_layers: int, omega: float, epochs: int, lr: float,
                    reg_weight: float, ste_start_epoch: int,
                    codebook_update_interval: int, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    k_config = {'layer_0': 256, 'layer_1': 64, 'layer_2': 64, 'layer_3': 128}

    print(f"[exp40b] loading Kodak images at {image_size}x{image_size}...", flush=True)
    images, names = load_kodak_images(image_size)
    print(f"[exp40b] loaded {len(images)} Kodak images: {names}", flush=True)
    
    # Try to import scipy for Laplacian
    try:
        from scipy.signal import convolve2d
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("[exp40b] WARNING: scipy not available, computing Laplacian manually", flush=True)

    all_runs = []
    for seed in seeds:
        ckpt_path = os.path.join(output_dir, f'ckpt_seed{seed}.json')
        if os.path.exists(ckpt_path):
            print(f"\n[exp40b] LOADING checkpoint seed={seed}", flush=True)
            with open(ckpt_path) as f:
                run_result = json.load(f)
            all_runs.append(run_result)
            continue

        print(f"\n[exp40b] === seed={seed} ===", flush=True)
        t_seed_start = time.time()

        per_image_results = []
        
        for idx, (img, name) in enumerate(zip(images, names)):
            print(f"\n  [{idx+1}/{len(images)}] {name}...", flush=True)
            
            # 1. Compute Laplacian variance
            if has_scipy:
                laplacian_var = variance_of_laplacian(img)
            else:
                # Manual Laplacian
                kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
                # Simple conv
                h, w = img.shape
                lap = np.zeros((h-2, w-2), dtype=np.float32)
                for i in range(h-2):
                    for j in range(w-2):
                        lap[i, j] = np.sum(img[i:i+3, j:j+3] * kernel)
                laplacian_var = float(lap.var())
            
            print(f"    Laplacian variance: {laplacian_var:.2f}", flush=True)
            
            # 2. Train SIREN+QAT
            t_enc_start = time.time()
            model, codebooks, train_info = train_qat_siren(
                img, hidden_features=hidden_features, hidden_layers=hidden_layers,
                omega=omega, epochs=epochs, lr=lr, seed=seed,
                k_config=k_config, reg_weight=reg_weight,
                codebook_update_interval=codebook_update_interval,
                ste_start_epoch=ste_start_epoch,
            )
            enc_time = time.time() - t_enc_start
            
            # 3. Measure post-quant PSNR and SSIM
            model_copy = CoinMLP(hidden_features=hidden_features,
                                  hidden_layers=hidden_layers, omega=omega)
            model_copy.load_state_dict(model.state_dict())
            siren_psnr = evaluate_psnr_post_quant(model_copy, codebooks, img)
            
            H, W = img.shape
            lo, hi = float(img.min()), float(img.max())
            target = normalize_to_pm1(img)
            ys, xs = np.meshgrid(np.linspace(-1, 1, H, dtype=np.float32),
                                  np.linspace(-1, 1, W, dtype=np.float32), indexing='ij')
            coords = np.stack([xs.flatten(), ys.flatten()], axis=1)
            coords_t = torch.tensor(coords)
            with torch.no_grad():
                pred = model_copy(coords_t).cpu().numpy().flatten()
            siren_recon = denormalize_from_pm1(pred, lo, hi).reshape(H, W)
            siren_ssim = compute_ssim(img, siren_recon)
            
            # 4. Decode time
            dec_time = measure_decode_time(model_copy, img)
            
            # 5. SIREN size
            siren_size = compute_quantized_size(codebooks, model)
            
            # 6. Compress codecs at matched size
            target_size = siren_size
            
            jpeg_payload, jpeg_actual, jpeg_q = jpeg_at_exact_size(img, target_size)
            jpeg_recon = decompress_grayscale(jpeg_payload)
            jpeg_psnr = compute_psnr(img, jpeg_recon)
            jpeg_ssim = compute_ssim(img, jpeg_recon)
            
            webp_payload, webp_actual, webp_q = webp_at_exact_size(img, target_size)
            webp_recon = decompress_grayscale(webp_payload)
            webp_psnr = compute_psnr(img, webp_recon)
            webp_ssim = compute_ssim(img, webp_recon)
            
            avif_payload, avif_actual, avif_q = avif_at_exact_size(img, target_size)
            avif_recon = decompress_grayscale(avif_payload)
            avif_psnr = compute_psnr(img, avif_recon)
            avif_ssim = compute_ssim(img, avif_recon)
            
            result = {
                'image_id': name,
                'laplacian_variance': laplacian_var,
                'siren': {
                    'psnr_db': siren_psnr, 'ssim': siren_ssim,
                    'size_bytes': siren_size,
                    'enc_time_s': enc_time, 'dec_time_s': dec_time,
                },
                'jpeg': {'psnr_db': jpeg_psnr, 'ssim': jpeg_ssim, 'size_bytes': jpeg_actual},
                'webp': {'psnr_db': webp_psnr, 'ssim': webp_ssim, 'size_bytes': webp_actual},
                'avif': {'psnr_db': avif_psnr, 'ssim': avif_ssim, 'size_bytes': avif_actual},
                'psnr_diff_siren_minus_avif': siren_psnr - avif_psnr,
                'ssim_diff_siren_minus_avif': siren_ssim - avif_ssim,
            }
            per_image_results.append(result)
            
            print(f"    SIREN: {siren_psnr:.2f} dB, SSIM={siren_ssim:.4f}, {siren_size}B, enc={enc_time:.1f}s, dec={dec_time:.4f}s", flush=True)
            print(f"    JPEG:  {jpeg_psnr:.2f} dB, SSIM={jpeg_ssim:.4f}, {jpeg_actual}B", flush=True)
            print(f"    WebP:  {webp_psnr:.2f} dB, SSIM={webp_ssim:.4f}, {webp_actual}B", flush=True)
            print(f"    AVIF:  {avif_psnr:.2f} dB, SSIM={avif_ssim:.4f}, {avif_actual}B", flush=True)
            print(f"    Δ(SIREN-AVIF): PSNR={siren_psnr-avif_psnr:+.2f} dB, SSIM={siren_ssim-avif_ssim:+.4f}", flush=True)
        
        # Aggregate
        siren_psnrs = [r['siren']['psnr_db'] for r in per_image_results]
        siren_ssims = [r['siren']['ssim'] for r in per_image_results]
        jpeg_psnrs = [r['jpeg']['psnr_db'] for r in per_image_results]
        webp_psnrs = [r['webp']['psnr_db'] for r in per_image_results]
        avif_psnrs = [r['avif']['psnr_db'] for r in per_image_results]
        avif_ssims = [r['avif']['ssim'] for r in per_image_results]
        enc_times = [r['siren']['enc_time_s'] for r in per_image_results]
        dec_times = [r['siren']['dec_time_s'] for r in per_image_results]
        laplacian_vars = [r['laplacian_variance'] for r in per_image_results]
        psnr_diffs = [r['psnr_diff_siren_minus_avif'] for r in per_image_results]
        ssim_diffs = [r['ssim_diff_siren_minus_avif'] for r in per_image_results]
        
        run_result = {
            'seed': seed,
            'num_images': len(per_image_results),
            'per_image': per_image_results,
            'mean_siren_psnr': float(np.mean(siren_psnrs)),
            'std_siren_psnr': float(np.std(siren_psnrs)),
            'mean_siren_ssim': float(np.mean(siren_ssims)),
            'std_siren_ssim': float(np.std(siren_ssims)),
            'mean_jpeg_psnr': float(np.mean(jpeg_psnrs)),
            'mean_webp_psnr': float(np.mean(webp_psnrs)),
            'mean_avif_psnr': float(np.mean(avif_psnrs)),
            'mean_avif_ssim': float(np.mean(avif_ssims)),
            'mean_enc_time_s': float(np.mean(enc_times)),
            'mean_dec_time_s': float(np.mean(dec_times)),
            'mean_psnr_diff_siren_avif': float(np.mean(psnr_diffs)),
            'mean_ssim_diff_siren_avif': float(np.mean(ssim_diffs)),
            'total_time_s': time.time() - t_seed_start,
        }
        
        with open(ckpt_path, 'w') as f:
            json.dump(run_result, f, indent=2, default=str)
        all_runs.append(run_result)
        
        print(f"\n  SEED {seed} SUMMARY:", flush=True)
        print(f"    SIREN: PSNR={run_result['mean_siren_psnr']:.2f}±{run_result['std_siren_psnr']:.2f} dB, SSIM={run_result['mean_siren_ssim']:.4f}±{run_result['std_siren_ssim']:.4f}", flush=True)
        print(f"    JPEG:  PSNR={run_result['mean_jpeg_psnr']:.2f} dB", flush=True)
        print(f"    WebP:  PSNR={run_result['mean_webp_psnr']:.2f} dB", flush=True)
        print(f"    AVIF:  PSNR={run_result['mean_avif_psnr']:.2f} dB, SSIM={run_result['mean_avif_ssim']:.4f}", flush=True)
        print(f"    Δ(SIREN-AVIF): PSNR={run_result['mean_psnr_diff_siren_avif']:+.2f} dB, SSIM={run_result['mean_ssim_diff_siren_avif']:+.4f}", flush=True)
        print(f"    Enc time: {run_result['mean_enc_time_s']:.1f}s, Dec time: {run_result['mean_dec_time_s']:.4f}s", flush=True)
    
    # Final aggregation across seeds
    final = all_runs[0]  # Use first seed for per-image detail
    if len(all_runs) > 1:
        siren_psnr_means = [r['mean_siren_psnr'] for r in all_runs]
        siren_ssim_means = [r['mean_siren_ssim'] for r in all_runs]
        avif_psnr_means = [r['mean_avif_psnr'] for r in all_runs]
        avif_ssim_means = [r['mean_avif_ssim'] for r in all_runs]
        psnr_diff_means = [r['mean_psnr_diff_siren_avif'] for r in all_runs]
        ssim_diff_means = [r['mean_ssim_diff_siren_avif'] for r in all_runs]
        
        aggregated = {
            'n_seeds': len(all_runs),
            'n_images_per_seed': all_runs[0]['num_images'],
            'siren_psnr': {'mean': float(np.mean(siren_psnr_means)), 'std': float(np.std(siren_psnr_means))},
            'siren_ssim': {'mean': float(np.mean(siren_ssim_means)), 'std': float(np.std(siren_ssim_means))},
            'avif_psnr': {'mean': float(np.mean(avif_psnr_means)), 'std': float(np.std(avif_psnr_means))},
            'avif_ssim': {'mean': float(np.mean(avif_ssim_means)), 'std': float(np.std(avif_ssim_means))},
            'psnr_diff_siren_avif': {'mean': float(np.mean(psnr_diff_means)), 'std': float(np.std(psnr_diff_means))},
            'ssim_diff_siren_avif': {'mean': float(np.mean(ssim_diff_means)), 'std': float(np.std(ssim_diff_means))},
        }
    else:
        aggregated = {
            'n_seeds': 1,
            'n_images_per_seed': all_runs[0]['num_images'],
            'siren_psnr': {'mean': all_runs[0]['mean_siren_psnr'], 'std': all_runs[0]['std_siren_psnr']},
            'siren_ssim': {'mean': all_runs[0]['mean_siren_ssim'], 'std': all_runs[0]['std_siren_ssim']},
            'avif_psnr': {'mean': all_runs[0]['mean_avif_psnr'], 'std': 0.0},
            'avif_ssim': {'mean': all_runs[0]['mean_avif_ssim'], 'std': 0.0},
            'psnr_diff_siren_avif': {'mean': all_runs[0]['mean_psnr_diff_siren_avif'], 'std': 0.0},
            'ssim_diff_siren_avif': {'mean': all_runs[0]['mean_ssim_diff_siren_avif'], 'std': 0.0},
        }
    
    # Correlation: Laplacian variance vs PSNR diff
    lap_vars = np.array([r['laplacian_variance'] for r in final['per_image']])
    psnr_diffs_arr = np.array([r['psnr_diff_siren_minus_avif'] for r in final['per_image']])
    ssim_diffs_arr = np.array([r['ssim_diff_siren_minus_avif'] for r in final['per_image']])
    
    if len(lap_vars) > 2:
        corr_psnr = float(np.corrcoef(lap_vars, psnr_diffs_arr)[0, 1])
        corr_ssim = float(np.corrcoef(lap_vars, ssim_diffs_arr)[0, 1])
    else:
        corr_psnr = 0.0
        corr_ssim = 0.0
    
    correlation = {
        'laplacian_vs_psnr_diff': corr_psnr,
        'laplacian_vs_ssim_diff': corr_ssim,
        'interpretation': 'Positive correlation means SIREN advantage grows with image complexity',
    }
    
    print(f"\n[exp40b] FINAL AGGREGATED ({aggregated['n_seeds']} seeds, {aggregated['n_images_per_seed']} images):", flush=True)
    print(f"  SIREN: PSNR={aggregated['siren_psnr']['mean']:.2f}±{aggregated['siren_psnr']['std']:.2f} dB, SSIM={aggregated['siren_ssim']['mean']:.4f}±{aggregated['siren_ssim']['std']:.4f}", flush=True)
    print(f"  AVIF:  PSNR={aggregated['avif_psnr']['mean']:.2f}±{aggregated['avif_psnr']['std']:.2f} dB, SSIM={aggregated['avif_ssim']['mean']:.4f}±{aggregated['avif_ssim']['std']:.4f}", flush=True)
    print(f"  Δ(SIREN-AVIF): PSNR={aggregated['psnr_diff_siren_avif']['mean']:+.2f}±{aggregated['psnr_diff_siren_avif']['std']:.2f} dB", flush=True)
    print(f"                  SSIM={aggregated['ssim_diff_siren_avif']['mean']:+.4f}±{aggregated['ssim_diff_siren_avif']['std']:.4f}", flush=True)
    print(f"  Correlation (Laplacian vs PSNR diff): {corr_psnr:.4f}", flush=True)
    print(f"  Correlation (Laplacian vs SSIM diff): {corr_ssim:.4f}", flush=True)
    
    # Print final table
    print(f"\n[exp40b] FINAL TABLE:", flush=True)
    print(f"  {'Image':10s} | {'Var_Lap':>8s} | {'SIREN_PSNR':>10s} | {'AVIF_PSNR':>10s} | {'SIREN_SSIM':>10s} | {'AVIF_SSIM':>10s} | {'Enc_Time':>8s}", flush=True)
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}", flush=True)
    for r in final['per_image']:
        print(f"  {r['image_id']:10s} | {r['laplacian_variance']:8.1f} | {r['siren']['psnr_db']:10.2f} | {r['avif']['psnr_db']:10.2f} | {r['siren']['ssim']:10.4f} | {r['avif']['ssim']:10.4f} | {r['siren']['enc_time_s']:8.1f}s", flush=True)
    
    output = {
        'experiment': 'experiment_40b_kodak_validation',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'SIREN+QAT beats JPEG/WebP on PSNR and all codecs on SSIM, '
                       'with advantage growing on complex images (high Laplacian variance).',
        'config': {
            'image_size': image_size,
            'hidden_features': hidden_features,
            'hidden_layers': hidden_layers,
            'omega': omega,
            'epochs': epochs,
            'lr': lr,
            'seeds': seeds,
            'k_config': k_config,
            'reg_weight': reg_weight,
            'ste_start_epoch': ste_start_epoch,
            'codebook_update_interval': codebook_update_interval,
            'dataset': 'Kodak kodim01-12, grayscale, 256×256',
        },
        'all_runs': all_runs,
        'aggregated': aggregated,
        'correlation': correlation,
    }
    
    out_json_path = os.path.join(output_dir, 'experiment_40b_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha
    
    print(f"\n[exp40b] DONE", flush=True)
    print(f"  JSON SHA-256: {json_sha}", flush=True)
    
    print("\n---JSON_BEGIN---")
    print(json.dumps(output, indent=2, default=str))
    print("---JSON_END---")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--hidden-features', type=int, default=64)
    parser.add_argument('--hidden-layers', type=int, default=2)
    parser.add_argument('--omega', type=float, default=30.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reg-weight', type=float, default=0.01)
    parser.add_argument('--ste-start-epoch', type=int, default=200)
    parser.add_argument('--codebook-update-interval', type=int, default=100)
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp40b_out')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    run_experiment(
        seeds=seeds, image_size=args.size,
        hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
        omega=args.omega, epochs=args.epochs, lr=args.lr,
        reg_weight=args.reg_weight, ste_start_epoch=args.ste_start_epoch,
        codebook_update_interval=args.codebook_update_interval,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
