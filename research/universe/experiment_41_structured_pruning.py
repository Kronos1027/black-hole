"""
experiment_41_structured_pruning.py
====================================
Experiment 41 — Structured Neuron Pruning + Recovery QAT.

Exp 40-B showed SIREN loses to AVIF by 21 dB on Kodak. The protocol
says: try structured pruning to reduce size, then re-test.

Strategy:
1. Train SIREN normally (500 epochs)
2. Remove 50% of neurons in hidden layer 2 (lowest L2 norm)
3. Recovery QAT: 300 epochs of fine-tuning with QAT (K=50) on pruned model
4. Compare against AVIF at the NEW reduced byte budget (~3.3 KB)

If SIREN can maintain SSIM at half the size, it might find a niche.
If not, the program ends with a negative conclusion.

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
from experiment_37_real_photo_parity import load_real_photos, compute_psnr
from experiment_39_qat_ste import (
    evaluate_psnr_pre_quant, evaluate_psnr_post_quant,
    compute_quantized_size, fit_per_layer_codebooks, update_codebooks,
)
from experiment_40_byte_parity_battlefield import (
    compute_ssim, avif_at_exact_size, decompress_grayscale,
)


class PrunedCoinMLP(nn.Module):
    """CoinMLP with prunable hidden layer neurons."""
    def __init__(self, hidden_features: int, hidden_layers: int, omega: float,
                 pruned_hidden_features: int = None):
        super().__init__()
        self.omega = omega
        self.pruned_hidden = pruned_hidden_features or hidden_features
        
        # Layer 1: input -> hidden (full size)
        self.layers = nn.ModuleList()
        # SIREN input layer
        bound = 1.0 / 2  # in_features=2
        layer0 = nn.Linear(2, hidden_features)
        with torch.no_grad():
            layer0.weight.uniform_(-bound, bound)
        self.layers.append(layer0)
        
        # Hidden layers (first one pruned)
        for i in range(hidden_layers):
            in_f = self.pruned_hidden if i == 0 else hidden_features
            bound_h = math.sqrt(6.0 / in_f) / omega
            layer = nn.Linear(in_f, hidden_features)
            with torch.no_grad():
                layer.weight.uniform_(-bound_h, bound_h)
            self.layers.append(layer)
        
        # Output layer
        self.output = nn.Linear(hidden_features, 1)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.sin(self.omega * layer(x))
        return self.output(x)
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def compute_neuron_l2_norms(model: CoinMLP, layer_idx: int = 1) -> np.ndarray:
    """Compute L2 norm of each neuron's weight vector in the specified layer."""
    # Layer 1 weights: (hidden_features, 2) — each row is a neuron
    # We want the L2 norm of each neuron's incoming weights
    w = model.net[layer_idx].linear.weight.detach().cpu().numpy()
    # Each neuron is a row: norm along axis=1
    norms = np.linalg.norm(w, axis=1)
    return norms


def prune_neurons(model: CoinMLP, layer_idx: int = 1, 
                    prune_ratio: float = 0.5) -> Tuple[PrunedCoinMLP, Dict]:
    """
    Remove the lowest-L2-norm neurons from the specified layer.
    Returns (pruned_model, pruning_stats).
    """
    norms = compute_neuron_l2_norms(model, layer_idx)
    n_neurons = len(norms)
    n_keep = int(n_neurons * (1 - prune_ratio))
    
    # Sort by norm (descending), keep top n_keep
    sorted_indices = np.argsort(norms)[::-1]
    keep_indices = sorted(sorted_indices[:n_keep])
    prune_indices = sorted(sorted_indices[n_keep:])
    
    # Create pruned model
    pruned = PrunedCoinMLP(
        hidden_features=n_neurons, hidden_layers=2, omega=model.net[0].omega,
        pruned_hidden_features=n_keep
    )
    
    # Copy weights for kept neurons
    with torch.no_grad():
        # Layer 0 (input -> hidden): keep only the rows for kept neurons
        w0 = model.net[0].linear.weight  # (hidden, 2)
        b0 = model.net[0].linear.bias    # (hidden,)
        pruned.layers[0].weight.data = w0[keep_indices].clone()
        pruned.layers[0].bias.data = b0[keep_indices].clone()
        
        # Layer 1 (hidden -> hidden): input is pruned, output is full
        w1 = model.net[1].linear.weight  # (hidden, hidden)
        b1 = model.net[1].linear.bias    # (hidden,)
        pruned.layers[1].weight.data = w1[:, keep_indices].clone()
        pruned.layers[1].bias.data = b1.clone()
        
        # Layer 2+ : copy as-is
        for i in range(2, len(model.net) - 1):
            pruned.layers[i].weight.data = model.net[i].linear.weight.data.clone()
            pruned.layers[i].bias.data = model.net[i].linear.bias.data.clone()
        
        # Output layer
        pruned.output.weight.data = model.net[-1].weight.data.clone()
        pruned.output.bias.data = model.net[-1].bias.data.clone()
    
    stats = {
        'original_neurons': n_neurons,
        'pruned_neurons': n_neurons - n_keep,
        'kept_neurons': n_keep,
        'prune_ratio': prune_ratio,
        'keep_indices': [int(x) for x in keep_indices],
        'prune_indices': [int(x) for x in prune_indices],
        'original_params': model.num_params(),
        'pruned_params': pruned.num_params(),
        'param_reduction': model.num_params() - pruned.num_params(),
    }
    return pruned, stats


def recovery_qat(pruned_model: PrunedCoinMLP, img: np.ndarray,
                  omega: float, epochs: int, lr: float, seed: int,
                  K: int, reg_weight: float, ste_start_epoch: int) -> Tuple[PrunedCoinMLP, Dict, Dict]:
    """
    Recovery QAT: fine-tune the pruned model with QAT.
    Returns (model, codebooks, info).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cpu')
    
    H, W = img.shape
    lo, hi = float(img.min()), float(img.max())
    target = normalize_to_pm1(img)
    ys, xs = np.meshgrid(np.linspace(-1, 1, H, dtype=np.float32),
                          np.linspace(-1, 1, W, dtype=np.float32), indexing='ij')
    coords = np.stack([xs.flatten(), ys.flatten()], axis=1)
    coords_t = torch.tensor(coords).to(device)
    targets_t = torch.tensor(target.flatten()).unsqueeze(1).to(device)
    
    opt = torch.optim.Adam(pruned_model.parameters(), lr=lr)
    
    # K config for pruned model
    k_config = {}
    for i, p in enumerate(pruned_model.parameters()):
        k_config[f'layer_{i}'] = K  # uniform K=50 for pruned model
    
    codebooks = fit_per_layer_codebooks(pruned_model, k_config, seed)
    
    from experiment_39_qat_ste import ste_quantize_weights, quantization_reg_loss
    
    for epoch in range(epochs):
        opt.zero_grad()
        
        if epoch >= ste_start_epoch:
            original_weights = [p.data.clone() for p in pruned_model.parameters()]
            params = list(pruned_model.parameters())
            for i, p in enumerate(params):
                layer_name = f'layer_{i}'
                codebook = codebooks[layer_name].to(device)
                p.data = ste_quantize_weights(p.data, codebook)
            
            pred = pruned_model(coords_t)
            recon_loss = F.mse_loss(pred, targets_t)
            
            for i, p in enumerate(params):
                p.data = original_weights[i]
            
            pred_orig = pruned_model(coords_t)
            recon_loss = F.mse_loss(pred_orig, targets_t)
            reg_loss = quantization_reg_loss(pruned_model, codebooks, device)
            total_loss = recon_loss + reg_weight * reg_loss
        else:
            pred = pruned_model(coords_t)
            recon_loss = F.mse_loss(pred, targets_t)
            reg_loss = torch.tensor(0.0)
            total_loss = recon_loss
        
        total_loss.backward()
        opt.step()
        
        if epoch >= ste_start_epoch and epoch % 100 == 0 and epoch > 0:
            codebooks = update_codebooks(pruned_model, codebooks, seed)
    
    codebooks = update_codebooks(pruned_model, codebooks, seed)
    
    info = {'k_config': k_config, 'reg_weight': reg_weight, 'ste_start_epoch': ste_start_epoch}
    return pruned_model, codebooks, info


def evaluate_pruned_model_psnr(model: PrunedCoinMLP, img: np.ndarray) -> float:
    """Evaluate PSNR of pruned model on image."""
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
                    reg_weight: float, ste_start_epoch: int,
                    prune_ratio: float, K: int, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[exp41] loading real photos at {image_size}x{image_size}...", flush=True)
    images, names = load_real_photos(image_size)
    print(f"[exp41] loaded {len(images)} real photos: {names}", flush=True)
    print(f"[exp41] Pruning {prune_ratio*100:.0f}% of hidden layer neurons, then {epochs}ep recovery QAT", flush=True)

    all_runs = []
    for seed in seeds:
        ckpt_path = os.path.join(output_dir, f'ckpt_seed{seed}.json')
        if os.path.exists(ckpt_path):
            print(f"\n[exp41] LOADING checkpoint seed={seed}", flush=True)
            with open(ckpt_path) as f:
                run_result = json.load(f)
            all_runs.append(run_result)
            continue

        print(f"\n[exp41] === seed={seed} ===", flush=True)
        t0 = time.time()

        per_image = []
        
        for i, img in enumerate(images):
            print(f"\n  [{names[i]}] Training original SIREN...", flush=True)
            # 1. Train original SIREN
            from experiment_39_qat_ste import train_qat_siren
            orig_model, _, _ = train_qat_siren(
                img, hidden_features=hidden_features, hidden_layers=hidden_layers,
                omega=omega, epochs=epochs, lr=lr, seed=seed,
                k_config={'layer_0': 256, 'layer_1': 64, 'layer_2': 64, 'layer_3': 128},
                reg_weight=reg_weight, codebook_update_interval=100,
                ste_start_epoch=ste_start_epoch,
            )
            
            # 2. Prune 50% of neurons
            print(f"  Pruning {prune_ratio*100:.0f}% neurons...", flush=True)
            pruned_model, prune_stats = prune_neurons(orig_model, layer_idx=1, prune_ratio=prune_ratio)
            print(f"    Original: {prune_stats['original_params']} params → Pruned: {prune_stats['pruned_params']} params", flush=True)
            
            # 3. Recovery QAT
            print(f"  Recovery QAT ({epochs} epochs)...", flush=True)
            t_rec_start = time.time()
            recovered_model, codebooks, qat_info = recovery_qat(
                pruned_model, img, omega=omega, epochs=epochs, lr=lr, seed=seed,
                K=K, reg_weight=reg_weight, ste_start_epoch=ste_start_epoch,
            )
            rec_time = time.time() - t_rec_start
            
            # 4. Evaluate
            siren_psnr = evaluate_pruned_model_psnr(recovered_model, img)
            
            # SSIM
            H, W = img.shape
            lo, hi = float(img.min()), float(img.max())
            target = normalize_to_pm1(img)
            ys, xs = np.meshgrid(np.linspace(-1, 1, H, dtype=np.float32),
                                  np.linspace(-1, 1, W, dtype=np.float32), indexing='ij')
            coords = np.stack([xs.flatten(), ys.flatten()], axis=1)
            coords_t = torch.tensor(coords)
            with torch.no_grad():
                pred = recovered_model(coords_t).cpu().numpy().flatten()
            siren_recon = denormalize_from_pm1(pred, lo, hi).reshape(H, W)
            siren_ssim = compute_ssim(img, siren_recon)
            
            # 5. Compute pruned size
            pruned_size = compute_quantized_size(codebooks, recovered_model)
            
            # 6. AVIF at matched size
            avif_payload, avif_actual, avif_q = avif_at_exact_size(img, pruned_size)
            avif_recon = decompress_grayscale(avif_payload)
            avif_psnr = compute_psnr(img, avif_recon)
            avif_ssim = compute_ssim(img, avif_recon)
            
            result = {
                'name': names[i],
                'prune_stats': prune_stats,
                'siren_psnr_db': siren_psnr,
                'siren_ssim': siren_ssim,
                'siren_size_bytes': pruned_size,
                'avif_psnr_db': avif_psnr,
                'avif_ssim': avif_ssim,
                'avif_size_bytes': avif_actual,
                'recovery_time_s': rec_time,
                'psnr_diff_siren_minus_avif': siren_psnr - avif_psnr,
                'ssim_diff_siren_minus_avif': siren_ssim - avif_ssim,
            }
            per_image.append(result)
            
            print(f"  RESULT: SIREN={siren_psnr:.2f} dB, SSIM={siren_ssim:.4f}, {pruned_size}B", flush=True)
            print(f"          AVIF={avif_psnr:.2f} dB, SSIM={avif_ssim:.4f}, {avif_actual}B", flush=True)
            print(f"          Δ: PSNR={siren_psnr-avif_psnr:+.2f}, SSIM={siren_ssim-avif_ssim:+.4f}", flush=True)
        
        # Aggregate
        siren_psnrs = [r['siren_psnr_db'] for r in per_image]
        siren_ssims = [r['siren_ssim'] for r in per_image]
        avif_psnrs = [r['avif_psnr_db'] for r in per_image]
        avif_ssims = [r['avif_ssim'] for r in per_image]
        siren_sizes = [r['siren_size_bytes'] for r in per_image]
        
        run_result = {
            'seed': seed,
            'per_image': per_image,
            'mean_siren_psnr': float(np.mean(siren_psnrs)),
            'mean_siren_ssim': float(np.mean(siren_ssims)),
            'mean_avif_psnr': float(np.mean(avif_psnrs)),
            'mean_avif_ssim': float(np.mean(avif_ssims)),
            'mean_siren_size': float(np.mean(siren_sizes)),
            'mean_psnr_diff': float(np.mean(siren_psnrs) - np.mean(avif_psnrs)),
            'mean_ssim_diff': float(np.mean(siren_ssims) - np.mean(avif_ssims)),
            'total_time_s': time.time() - t0,
        }
        
        with open(ckpt_path, 'w') as f:
            json.dump(run_result, f, indent=2, default=str)
        all_runs.append(run_result)
        
        print(f"\n  SEED {seed} SUMMARY:", flush=True)
        print(f"    SIREN (pruned): PSNR={run_result['mean_siren_psnr']:.2f} dB, SSIM={run_result['mean_siren_ssim']:.4f}, {run_result['mean_siren_size']:.0f} B", flush=True)
        print(f"    AVIF:           PSNR={run_result['mean_avif_psnr']:.2f} dB, SSIM={run_result['mean_avif_ssim']:.4f}", flush=True)
        print(f"    Δ:              PSNR={run_result['mean_psnr_diff']:+.2f} dB, SSIM={run_result['mean_ssim_diff']:+.4f}", flush=True)
    
    # Final aggregation
    siren_psnr_means = [r['mean_siren_psnr'] for r in all_runs]
    siren_ssim_means = [r['mean_siren_ssim'] for r in all_runs]
    avif_psnr_means = [r['mean_avif_psnr'] for r in all_runs]
    avif_ssim_means = [r['mean_avif_ssim'] for r in all_runs]
    
    aggregated = {
        'n_seeds': len(all_runs),
        'siren_psnr': {'mean': float(np.mean(siren_psnr_means)), 'std': float(np.std(siren_psnr_means))},
        'siren_ssim': {'mean': float(np.mean(siren_ssim_means)), 'std': float(np.std(siren_ssim_means))},
        'avif_psnr': {'mean': float(np.mean(avif_psnr_means)), 'std': float(np.std(avif_psnr_means))},
        'avif_ssim': {'mean': float(np.mean(avif_ssim_means)), 'std': float(np.std(avif_ssim_means))},
    }
    
    siren_psnr = aggregated['siren_psnr']['mean']
    avif_psnr = aggregated['avif_psnr']['mean']
    siren_ssim = aggregated['siren_ssim']['mean']
    avif_ssim = aggregated['avif_ssim']['mean']
    
    if siren_ssim > avif_ssim:
        winner = "SIREN (pruned) maintains SSIM advantage"
        conclusion = "POSITIVE — structured pruning preserves SSIM at half the size"
    else:
        winner = "AVIF"
        conclusion = "NEGATIVE — structured pruning does not help; AVIF still wins"
    
    comparison = {
        'psnr_diff': float(siren_psnr - avif_psnr),
        'ssim_diff': float(siren_ssim - avif_ssim),
        'winner': winner,
        'conclusion': conclusion,
    }
    
    print(f"\n[exp41] FINAL:", flush=True)
    print(f"  SIREN (pruned): PSNR={siren_psnr:.2f} dB, SSIM={siren_ssim:.4f}", flush=True)
    print(f"  AVIF:           PSNR={avif_psnr:.2f} dB, SSIM={avif_ssim:.4f}", flush=True)
    print(f"  WINNER: {winner}", flush=True)
    print(f"  CONCLUSION: {conclusion}", flush=True)
    
    output = {
        'experiment': 'experiment_41_structured_pruning',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'Structured pruning (50% neurons) + recovery QAT maintains SSIM '
                       'while reducing size by ~50%, making SIREN competitive.',
        'config': {
            'image_size': image_size,
            'hidden_features': hidden_features,
            'hidden_layers': hidden_layers,
            'omega': omega,
            'epochs': epochs,
            'lr': lr,
            'seeds': seeds,
            'prune_ratio': prune_ratio,
            'K': K,
            'reg_weight': reg_weight,
            'ste_start_epoch': ste_start_epoch,
        },
        'all_runs': all_runs,
        'aggregated': aggregated,
        'comparison': comparison,
    }
    
    out_json_path = os.path.join(output_dir, 'experiment_41_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha
    
    print(f"\n[exp41] DONE", flush=True)
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
    parser.add_argument('--reg-weight', type=float, default=0.01)
    parser.add_argument('--ste-start-epoch', type=int, default=100)
    parser.add_argument('--prune-ratio', type=float, default=0.5)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp41_out')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    run_experiment(
        seeds=seeds, image_size=args.size,
        hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
        omega=args.omega, epochs=args.epochs, lr=args.lr,
        reg_weight=args.reg_weight, ste_start_epoch=args.ste_start_epoch,
        prune_ratio=args.prune_ratio, K=args.K, output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
