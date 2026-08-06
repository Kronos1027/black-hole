"""
experiment_40_byte_parity_battlefield.py
==========================================
Experiment 40 — The Byte Parity Battlefield.

Exp 39 showed QAT reduces quantization loss to 1.19 dB, giving SIREN+QAT
30.75 dB at 6562 B. But AVIF was at 642 B — unfair comparison.

This experiment runs the DEFINITIVE test:
1. Train SIREN with QAT (same config as Exp 39)
2. Configure AVIF, WebP, JPEG to hit EXACTLY 6562 bytes (SIREN's size)
3. Compare PSNR AND SSIM at matched byte budget
4. Compute Shannon entropy of SIREN weight indices post-QAT
5. Estimate arithmetic coding savings

If SIREN loses at matched budget, next step is structured pruning (Exp 41).

ANTI-FABRICACTION: same protocol. Output real, SHA-256, no tuning.
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
from experiment_37_real_photo_parity import (
    load_real_photos, compute_psnr,
)
from experiment_39_qat_ste import (
    train_qat_siren, evaluate_psnr_pre_quant, evaluate_psnr_post_quant,
    compute_quantized_size, fit_per_layer_codebooks, update_codebooks,
    ste_quantize_weights, STEKMeansQuantize,
)


# ---------------------------------------------------------------------------
# SSIM computation
# ---------------------------------------------------------------------------

def compute_ssim(img: np.ndarray, recon: np.ndarray) -> float:
    """Compute SSIM between original and reconstruction."""
    from skimage.metrics import structural_similarity as ssim
    # Ensure same shape
    if recon.shape != img.shape:
        recon = recon[:img.shape[0], :img.shape[1]]
    # SSIM requires float64 and data_range
    img_f64 = img.astype(np.float64)
    recon_f64 = recon.astype(np.float64)
    data_range = float(img_f64.max() - img_f64.min())
    if data_range < 1e-6:
        data_range = 255.0
    return float(ssim(img_f64, recon_f64, data_range=data_range))


# ---------------------------------------------------------------------------
# Codec compression at EXACT target size
# ---------------------------------------------------------------------------

def jpeg_at_exact_size(img: np.ndarray, target_size: int) -> Tuple[bytes, int, int]:
    """Compress as JPEG at exact target size. Returns (payload, actual_size, quality)."""
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')

    best_payload = None
    best_size = None
    best_quality = None
    best_diff = float('inf')

    lo_q, hi_q = 1, 100
    for _ in range(30):
        mid_q = (lo_q + hi_q) // 2
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=mid_q, subsampling=0)
        size = buf.tell()

        diff = abs(size - target_size)
        if diff < best_diff:
            best_diff = diff
            best_payload = buf.getvalue()
            best_size = size
            best_quality = mid_q

        if size == target_size:
            return buf.getvalue(), size, mid_q
        if size < target_size:
            lo_q = mid_q + 1
        else:
            hi_q = mid_q - 1
        if lo_q > hi_q:
            break

    return best_payload, best_size, best_quality


def webp_at_exact_size(img: np.ndarray, target_size: int) -> Tuple[bytes, int, int]:
    """Compress as WebP at exact target size."""
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')

    best_payload = None
    best_size = None
    best_quality = None
    best_diff = float('inf')

    lo_q, hi_q = 1, 100
    for _ in range(30):
        mid_q = (lo_q + hi_q) // 2
        buf = io.BytesIO()
        pil_img.save(buf, format='WebP', quality=mid_q)
        size = buf.tell()

        diff = abs(size - target_size)
        if diff < best_diff:
            best_diff = diff
            best_payload = buf.getvalue()
            best_size = size
            best_quality = mid_q

        if size == target_size:
            return buf.getvalue(), size, mid_q
        if size < target_size:
            lo_q = mid_q + 1
        else:
            hi_q = mid_q - 1
        if lo_q > hi_q:
            break

    return best_payload, best_size, best_quality


def avif_at_exact_size(img: np.ndarray, target_size: int) -> Tuple[bytes, int, int]:
    """Compress as AVIF at exact target size."""
    import pillow_avif

    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')

    best_payload = None
    best_size = None
    best_quality = None
    best_diff = float('inf')

    lo_q, hi_q = 1, 100
    for _ in range(30):
        mid_q = (lo_q + hi_q) // 2
        buf = io.BytesIO()
        pil_img.save(buf, format='AVIF', quality=mid_q)
        size = buf.tell()

        diff = abs(size - target_size)
        if diff < best_diff:
            best_diff = diff
            best_payload = buf.getvalue()
            best_size = size
            best_quality = mid_q

        if size == target_size:
            return buf.getvalue(), size, mid_q
        if size < target_size:
            lo_q = mid_q + 1
        else:
            hi_q = mid_q - 1
        if lo_q > hi_q:
            break

    return best_payload, best_size, best_quality


def decompress_grayscale(payload: bytes) -> np.ndarray:
    """Decompress any format and return as float32 grayscale array."""
    pil_img = Image.open(io.BytesIO(payload))
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    return np.array(pil_img, dtype=np.float32)


# ---------------------------------------------------------------------------
# Shannon entropy analysis of SIREN weight indices
# ---------------------------------------------------------------------------

def compute_weight_indices_entropy(model: nn.Module, codebooks: Dict[str, torch.Tensor]) -> Dict:
    """
    Compute Shannon entropy of the KMeans indices for each layer.
    
    If entropy is significantly less than log2(K), arithmetic coding
    can compress the indices further.
    """
    results = {}
    params = list(model.parameters())
    total_entropy_bits = 0
    total_raw_bits = 0
    total_n_weights = 0

    for i, p in enumerate(params):
        layer_name = f'layer_{i}'
        codebook = codebooks[layer_name]
        k = len(codebook)

        # Get quantized indices
        flat = p.data.flatten()
        dists = torch.abs(flat.unsqueeze(-1) - codebook.unsqueeze(0))
        indices = dists.argmin(dim=-1).cpu().numpy()

        # Compute symbol frequencies
        n = len(indices)
        counts = np.bincount(indices, minlength=k)
        probs = counts[counts > 0] / n

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        raw_bits_per_index = math.ceil(math.log2(max(2, k)))
        raw_bits = raw_bits_per_index * n
        entropy_bits = entropy * n

        # Potential savings
        savings_bits = raw_bits - entropy_bits
        savings_bytes = savings_bits / 8
        savings_percent = (savings_bits / raw_bits * 100) if raw_bits > 0 else 0

        results[layer_name] = {
            'K': k,
            'n_weights': n,
            'entropy_bits_per_index': float(entropy),
            'raw_bits_per_index': raw_bits_per_index,
            'entropy_bits_total': float(entropy_bits),
            'raw_bits_total': int(raw_bits),
            'savings_bits': float(savings_bits),
            'savings_bytes': float(savings_bytes),
            'savings_percent': float(savings_percent),
        }

        total_entropy_bits += entropy_bits
        total_raw_bits += raw_bits
        total_n_weights += n

    total_savings_bits = total_raw_bits - total_entropy_bits
    results['_total'] = {
        'n_weights': total_n_weights,
        'raw_bits_total': int(total_raw_bits),
        'entropy_bits_total': float(total_entropy_bits),
        'savings_bits': float(total_savings_bits),
        'savings_bytes': float(total_savings_bits / 8),
        'savings_percent': float(total_savings_bits / total_raw_bits * 100) if total_raw_bits > 0 else 0,
        'estimated_arithmetic_coded_size_bytes': float(total_entropy_bits / 8),
    }

    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(seeds: List[int], image_size: int, hidden_features: int,
                    hidden_layers: int, omega: float, epochs: int, lr: float,
                    reg_weight: float, ste_start_epoch: int,
                    codebook_update_interval: int, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    k_config = {
        'layer_0': 256,
        'layer_1': 64,
        'layer_2': 64,
        'layer_3': 128,
    }

    print(f"[exp40] loading real photos at {image_size}x{image_size}...", flush=True)
    images, names = load_real_photos(image_size)
    print(f"[exp40] loaded {len(images)} real photos: {names}", flush=True)
    print(f"[exp40] BYTE PARITY BATTLEFIELD — all codecs at SIREN's exact size", flush=True)

    all_runs = []
    for seed in seeds:
        ckpt_path = os.path.join(output_dir, f'ckpt_seed{seed}.json')
        if os.path.exists(ckpt_path):
            print(f"\n[exp40] LOADING checkpoint seed={seed}", flush=True)
            with open(ckpt_path) as f:
                run_result = json.load(f)
            all_runs.append(run_result)
            print(f"  CACHED", flush=True)
            continue

        print(f"\n[exp40] === seed={seed} ===", flush=True)
        t0 = time.time()

        per_image_results = []
        siren_psnrs = []
        siren_ssims = []
        siren_sizes = []
        jpeg_psnrs = []
        jpeg_ssims = []
        jpeg_sizes = []
        webp_psnrs = []
        webp_ssims = []
        webp_sizes = []
        avif_psnrs = []
        avif_ssims = []
        avif_sizes = []

        models_for_entropy = []

        for i, img in enumerate(images):
            # 1. Train SIREN with QAT
            model, codebooks, train_info = train_qat_siren(
                img, hidden_features=hidden_features, hidden_layers=hidden_layers,
                omega=omega, epochs=epochs, lr=lr, seed=seed,
                k_config=k_config, reg_weight=reg_weight,
                codebook_update_interval=codebook_update_interval,
                ste_start_epoch=ste_start_epoch,
            )

            # 2. Measure SIREN post-quant PSNR and SSIM
            model_copy = CoinMLP(hidden_features=hidden_features,
                                  hidden_layers=hidden_layers, omega=omega)
            model_copy.load_state_dict(model.state_dict())
            siren_psnr = evaluate_psnr_post_quant(model_copy, codebooks, img)

            # Get SIREN reconstruction for SSIM
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

            # 3. Compute SIREN quantized size
            siren_size = compute_quantized_size(codebooks, model)

            # Save model for entropy analysis
            models_for_entropy.append((model, codebooks))

            # 4. Compress JPEG, WebP, AVIF at EXACT siren_size
            target = siren_size

            jpeg_payload, jpeg_actual_size, jpeg_q = jpeg_at_exact_size(img, target)
            jpeg_recon = decompress_grayscale(jpeg_payload)
            jpeg_psnr = compute_psnr(img, jpeg_recon)
            jpeg_ssim = compute_ssim(img, jpeg_recon)

            webp_payload, webp_actual_size, webp_q = webp_at_exact_size(img, target)
            webp_recon = decompress_grayscale(webp_payload)
            webp_psnr = compute_psnr(img, webp_recon)
            webp_ssim = compute_ssim(img, webp_recon)

            avif_payload, avif_actual_size, avif_q = avif_at_exact_size(img, target)
            avif_recon = decompress_grayscale(avif_payload)
            avif_psnr = compute_psnr(img, avif_recon)
            avif_ssim = compute_ssim(img, avif_recon)

            siren_psnrs.append(siren_psnr)
            siren_ssims.append(siren_ssim)
            siren_sizes.append(siren_size)
            jpeg_psnrs.append(jpeg_psnr)
            jpeg_ssims.append(jpeg_ssim)
            jpeg_sizes.append(jpeg_actual_size)
            webp_psnrs.append(webp_psnr)
            webp_ssims.append(webp_ssim)
            webp_sizes.append(webp_actual_size)
            avif_psnrs.append(avif_psnr)
            avif_ssims.append(avif_ssim)
            avif_sizes.append(avif_actual_size)

            per_image_results.append({
                'name': names[i],
                'siren': {'psnr_db': siren_psnr, 'ssim': siren_ssim, 'size_bytes': siren_size},
                'jpeg': {'psnr_db': jpeg_psnr, 'ssim': jpeg_ssim, 'size_bytes': jpeg_actual_size, 'quality': jpeg_q},
                'webp': {'psnr_db': webp_psnr, 'ssim': webp_ssim, 'size_bytes': webp_actual_size, 'quality': webp_q},
                'avif': {'psnr_db': avif_psnr, 'ssim': avif_ssim, 'size_bytes': avif_actual_size, 'quality': avif_q},
            })

            print(f"  {names[i]} (target {target} B):", flush=True)
            print(f"    SIREN:  {siren_psnr:.2f} dB, SSIM={siren_ssim:.4f}, {siren_size} B", flush=True)
            print(f"    JPEG:   {jpeg_psnr:.2f} dB, SSIM={jpeg_ssim:.4f}, {jpeg_actual_size} B (q={jpeg_q})", flush=True)
            print(f"    WebP:   {webp_psnr:.2f} dB, SSIM={webp_ssim:.4f}, {webp_actual_size} B (q={webp_q})", flush=True)
            print(f"    AVIF:   {avif_psnr:.2f} dB, SSIM={avif_ssim:.4f}, {avif_actual_size} B (q={avif_q})", flush=True)

        # 5. Shannon entropy analysis (on first image's model as representative)
        entropy_analysis = compute_weight_indices_entropy(
            models_for_entropy[0][0], models_for_entropy[0][1]
        )

        print(f"\n  SHANNON ENTROPY ANALYSIS (seed={seed}, first image):", flush=True)
        for layer, info in entropy_analysis.items():
            if layer == '_total':
                print(f"    TOTAL: raw={info['raw_bits_total']} bits, entropy={info['entropy_bits_total']:.0f} bits", flush=True)
                print(f"           savings={info['savings_bytes']:.0f} B ({info['savings_percent']:.1f}%)", flush=True)
                print(f"           estimated arithmetic coded size: {info['estimated_arithmetic_coded_size_bytes']:.0f} B", flush=True)
            else:
                print(f"    {layer}: K={info['K']}, entropy={info['entropy_bits_per_index']:.2f} bits (raw={info['raw_bits_per_index']})", flush=True)

        # Save weights for SHA-256
        weights_payload = bytearray()
        for m, _ in models_for_entropy:
            w = np.concatenate([p.detach().cpu().numpy().flatten() for p in m.parameters()])
            weights_payload += w.astype(np.float32).tobytes()
        weights_file = os.path.join(output_dir, f'exp40_weights_seed{seed}.bin')
        with open(weights_file, 'wb') as f:
            f.write(bytes(weights_payload))
        with open(weights_file, 'rb') as f:
            weights_sha = hashlib.sha256(f.read()).hexdigest()

        run_result = {
            'seed': seed,
            'per_image': per_image_results,
            'entropy_analysis': entropy_analysis,
            'mean_siren_psnr': float(np.mean(siren_psnrs)),
            'mean_siren_ssim': float(np.mean(siren_ssims)),
            'mean_siren_size': float(np.mean(siren_sizes)),
            'mean_jpeg_psnr': float(np.mean(jpeg_psnrs)),
            'mean_jpeg_ssim': float(np.mean(jpeg_ssims)),
            'mean_jpeg_size': float(np.mean(jpeg_sizes)),
            'mean_webp_psnr': float(np.mean(webp_psnrs)),
            'mean_webp_ssim': float(np.mean(webp_ssims)),
            'mean_webp_size': float(np.mean(webp_sizes)),
            'mean_avif_psnr': float(np.mean(avif_psnrs)),
            'mean_avif_ssim': float(np.mean(avif_ssims)),
            'mean_avif_size': float(np.mean(avif_sizes)),
            'total_time_s': time.time() - t0,
            'weights_sha256': weights_sha,
        }

        with open(ckpt_path, 'w') as f:
            json.dump(run_result, f, indent=2, default=str)
        all_runs.append(run_result)

        print(f"\n  MEAN (seed={seed}):", flush=True)
        print(f"    SIREN:  {run_result['mean_siren_psnr']:.2f} dB, SSIM={run_result['mean_siren_ssim']:.4f}, {run_result['mean_siren_size']:.0f} B", flush=True)
        print(f"    JPEG:   {run_result['mean_jpeg_psnr']:.2f} dB, SSIM={run_result['mean_jpeg_ssim']:.4f}, {run_result['mean_jpeg_size']:.0f} B", flush=True)
        print(f"    WebP:   {run_result['mean_webp_psnr']:.2f} dB, SSIM={run_result['mean_webp_ssim']:.4f}, {run_result['mean_webp_size']:.0f} B", flush=True)
        print(f"    AVIF:   {run_result['mean_avif_psnr']:.2f} dB, SSIM={run_result['mean_avif_ssim']:.4f}, {run_result['mean_avif_size']:.0f} B", flush=True)

    # Aggregate
    siren_psnr = np.array([r['mean_siren_psnr'] for r in all_runs])
    jpeg_psnr = np.array([r['mean_jpeg_psnr'] for r in all_runs])
    webp_psnr = np.array([r['mean_webp_psnr'] for r in all_runs])
    avif_psnr = np.array([r['mean_avif_psnr'] for r in all_runs])
    siren_ssim = np.array([r['mean_siren_ssim'] for r in all_runs])
    jpeg_ssim = np.array([r['mean_jpeg_ssim'] for r in all_runs])
    webp_ssim = np.array([r['mean_webp_ssim'] for r in all_runs])
    avif_ssim = np.array([r['mean_avif_ssim'] for r in all_runs])

    aggregated = {
        'siren': {
            'mean_psnr_db': float(siren_psnr.mean()), 'std_psnr_db': float(siren_psnr.std()),
            'mean_ssim': float(siren_ssim.mean()), 'std_ssim': float(siren_ssim.std()),
        },
        'jpeg': {
            'mean_psnr_db': float(jpeg_psnr.mean()), 'std_psnr_db': float(jpeg_psnr.std()),
            'mean_ssim': float(jpeg_ssim.mean()), 'std_ssim': float(jpeg_ssim.std()),
        },
        'webp': {
            'mean_psnr_db': float(webp_psnr.mean()), 'std_psnr_db': float(webp_psnr.std()),
            'mean_ssim': float(webp_ssim.mean()), 'std_ssim': float(webp_ssim.std()),
        },
        'avif': {
            'mean_psnr_db': float(avif_psnr.mean()), 'std_psnr_db': float(avif_psnr.std()),
            'mean_ssim': float(avif_ssim.mean()), 'std_ssim': float(avif_ssim.std()),
        },
    }

    print(f"\n[exp40] AGGREGATED across {len(all_runs)} seeds:", flush=True)
    for name, agg in aggregated.items():
        print(f"  {name}: PSNR={agg['mean_psnr_db']:.2f}±{agg['std_psnr_db']:.2f} dB, "
              f"SSIM={agg['mean_ssim']:.4f}±{agg['std_ssim']:.4f}", flush=True)

    # Comparison
    s = aggregated['siren']['mean_psnr_db']
    j = aggregated['jpeg']['mean_psnr_db']
    w = aggregated['webp']['mean_psnr_db']
    a = aggregated['avif']['mean_psnr_db']

    diffs = {
        'siren_minus_jpeg': float(s - j),
        'siren_minus_webp': float(s - w),
        'siren_minus_avif': float(s - a),
    }

    if s > j and s > w and s > a:
        winner = "SIREN+QAT (beats ALL codecs at matched byte budget)"
        conclusion = "POSITIVE — SIREN+QAT is the winner on the byte parity battlefield"
    elif s > max(j, w, a):
        winner = "SIREN+QAT (beats some but not all)"
        conclusion = "MIXED — SIREN beats some codecs but not all"
    else:
        winner = f"{max(j, w, a):.0f} dB codec wins"
        conclusion = "NEGATIVE — SIREN+QAT loses at matched byte budget"

    # Entropy analysis summary
    entropy_summary = all_runs[0].get('entropy_analysis', {}).get('_total', {})

    comparison = {
        'psnr_diffs': diffs,
        'winner': winner,
        'conclusion': conclusion,
        'entropy_analysis_summary': entropy_summary,
    }

    print(f"\n[exp40] BYTE PARITY BATTLEFIELD RESULTS:", flush=True)
    print(f"  SIREN - JPEG: {diffs['siren_minus_jpeg']:+.2f} dB", flush=True)
    print(f"  SIREN - WebP: {diffs['siren_minus_webp']:+.2f} dB", flush=True)
    print(f"  SIREN - AVIF: {diffs['siren_minus_avif']:+.2f} dB", flush=True)
    print(f"  WINNER: {winner}", flush=True)
    print(f"  CONCLUSION: {conclusion}", flush=True)
    if entropy_summary:
        print(f"  Entropy: raw={entropy_summary.get('raw_bits_total', 0)} bits, "
              f"entropy={entropy_summary.get('entropy_bits_total', 0):.0f} bits, "
              f"savings={entropy_summary.get('savings_percent', 0):.1f}%", flush=True)

    output = {
        'experiment': 'experiment_40_byte_parity_battlefield',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'SIREN+QAT at 6562 B beats JPEG, WebP, AND AVIF at the same byte budget.',
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
        },
        'all_runs': all_runs,
        'aggregated': aggregated,
        'comparison': comparison,
    }

    out_json_path = os.path.join(output_dir, 'experiment_40_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha

    print(f"\n[exp40] DONE", flush=True)
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
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reg-weight', type=float, default=0.01)
    parser.add_argument('--ste-start-epoch', type=int, default=200)
    parser.add_argument('--codebook-update-interval', type=int, default=100)
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp40_out')
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
