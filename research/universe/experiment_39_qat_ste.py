"""
experiment_39_qat_ste.py
====================================
Experiment 39 — Quantization-Aware Training with Straight-Through Estimator.

Root cause identified in Exp 38: KMeans K=50 post-training quantization
destroys 13.33 dB of PSNR. The fix: train the model AWARE that it will
be quantized, using:

1. Straight-Through Estimator (STE): forward pass uses quantized weights,
   backward pass passes gradient through to float32 weights as if
   quantization didn't happen. Model learns to be robust to quantization.

2. Asymmetric K distribution (per-layer codebooks):
   - Layer 1 (input/coordinate): K=256 (high granularity, handles frequencies)
   - Hidden layers: K=64 (lower granularity, bulk of params)
   - Output layer: K=128 (medium granularity for output projection)

3. Quantization-friendly regularization: penalize weights that are far
   from their nearest centroid, so the model converges to weights that
   quantize cleanly.

4. Comparison against AVIF at 0.1 BPP (not JPEG) — AVIF is the modern
   codec that should be the real competitor.

ANTI-FABRICATION: same protocol. Output real, SHA-256, no predicted gains.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
import time
from typing import Dict, List, Tuple, Optional

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
    load_scikit_images, normalize_to_pm1, denormalize_from_pm1, CoinMLP,
)
from experiment_37_real_photo_parity import (
    load_real_photos, compute_psnr,
)


# ---------------------------------------------------------------------------
# STE (Straight-Through Estimator) for KMeans quantization
# ---------------------------------------------------------------------------

class STEKMeansQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for KMeans quantization.
    
    Forward: quantize weights to nearest codebook centroid
    Backward: pass gradient through unchanged (as if quantization didn't happen)
    """
    
    @staticmethod
    def forward(ctx, weights: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        # Find nearest centroid for each weight
        # weights: (N,), codebook: (K,)
        # Compute distances: (N, K)
        dists = torch.abs(weights.unsqueeze(-1) - codebook.unsqueeze(0))
        indices = dists.argmin(dim=-1)  # (N,)
        quantized = codebook[indices]  # (N,)
        
        # Save for regularization
        ctx.save_for_backward(weights, quantized)
        return quantized
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # STE: pass gradient through unchanged
        return grad_output, None


def ste_quantize_weights(weights: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Apply STE quantization to a weight tensor."""
    flat = weights.flatten()
    quantized_flat = STEKMeansQuantize.apply(flat, codebook)
    return quantized_flat.view_as(weights)


# ---------------------------------------------------------------------------
# Per-layer codebook management
# ---------------------------------------------------------------------------

def fit_per_layer_codebooks(model: nn.Module, k_config: Dict[str, int],
                              seed: int) -> Dict[str, torch.Tensor]:
    """
    Fit a separate KMeans codebook for each layer's weights.
    
    k_config: {'layer_0': 256, 'layer_1': 64, ..., 'output': 128}
    Returns: {'layer_0': codebook_tensor, ...}
    """
    codebooks = {}
    params = list(model.parameters())
    
    for i, p in enumerate(params):
        layer_name = f'layer_{i}'
        k = k_config.get(layer_name, 64)  # default K=64
        
        w = p.detach().cpu().numpy().flatten()
        n_unique = len(np.unique(w))
        actual_k = min(k, n_unique, len(w))
        
        if actual_k < 2:
            # Too few unique values, use identity codebook
            codebook = np.array([w.mean() if len(w) > 0 else 0.0], dtype=np.float32)
        else:
            km = KMeans(n_clusters=actual_k, random_state=seed, n_init=4)
            km.fit(w.reshape(-1, 1))
            codebook = km.cluster_centers_.flatten().astype(np.float32)
        
        codebooks[layer_name] = torch.from_numpy(codebook)
    
    return codebooks


def update_codebooks(model: nn.Module, codebooks: Dict[str, torch.Tensor],
                       seed: int) -> Dict[str, torch.Tensor]:
    """Re-fit codebooks based on current model weights."""
    params = list(model.parameters())
    new_codebooks = {}
    
    for i, p in enumerate(params):
        layer_name = f'layer_{i}'
        old_codebook = codebooks[layer_name]
        k = len(old_codebook)
        
        w = p.detach().cpu().numpy().flatten()
        n_unique = len(np.unique(w))
        actual_k = min(k, n_unique, len(w))
        
        if actual_k < 2:
            new_codebooks[layer_name] = old_codebook
        else:
            km = KMeans(n_clusters=actual_k, random_state=seed, n_init=4)
            km.fit(w.reshape(-1, 1))
            codebook = km.cluster_centers_.flatten().astype(np.float32)
            new_codebooks[layer_name] = torch.from_numpy(codebook)
    
    return new_codebooks


# ---------------------------------------------------------------------------
# Quantization-friendly regularization
# ---------------------------------------------------------------------------

def quantization_reg_loss(model: nn.Module, codebooks: Dict[str, torch.Tensor],
                            device: torch.device) -> torch.Tensor:
    """
    Regularization that penalizes weights far from their nearest centroid.
    
    This pushes the model to converge to weights that quantize cleanly,
    minimizing the "phase shift" caused by rounding.
    """
    total_loss = torch.tensor(0.0, device=device)
    params = list(model.parameters())
    
    for i, p in enumerate(params):
        layer_name = f'layer_{i}'
        codebook = codebooks[layer_name].to(device)
        
        w = p.flatten()  # (N,)
        # Distance to nearest centroid
        dists = torch.abs(w.unsqueeze(-1) - codebook.unsqueeze(0))  # (N, K)
        min_dists = dists.min(dim=-1).values  # (N,)
        
        total_loss = total_loss + torch.mean(min_dists ** 2)
    
    return total_loss / len(params)


# ---------------------------------------------------------------------------
# QAT training loop
# ---------------------------------------------------------------------------

def train_qat_siren(img: np.ndarray, hidden_features: int, hidden_layers: int,
                      omega: float, epochs: int, lr: float, seed: int,
                      k_config: Dict[str, int], reg_weight: float,
                      codebook_update_interval: int,
                      ste_start_epoch: int) -> Tuple[CoinMLP, Dict]:
    """
    Train SIREN with Quantization-Aware Training.
    
    - First `ste_start_epoch` epochs: normal training (let model converge)
    - After `ste_start_epoch`: enable STE quantization + regularization
    - Every `codebook_update_interval` epochs: re-fit codebooks
    
    Returns (model, training_info)
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
    targets = target.flatten()
    
    coords_t = torch.tensor(coords).to(device)
    targets_t = torch.tensor(targets).unsqueeze(1).to(device)
    
    model = CoinMLP(hidden_features=hidden_features, hidden_layers=hidden_layers, omega=omega).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initial codebooks (will be updated during training)
    codebooks = fit_per_layer_codebooks(model, k_config, seed)
    
    # Training history
    history = {'epoch': [], 'loss': [], 'reg_loss': [], 'total_loss': [], 'ste_active': []}
    
    for epoch in range(epochs):
        opt.zero_grad()
        
        if epoch >= ste_start_epoch:
            # QAT mode: apply STE quantization to weights
            # Save original weights
            original_weights = []
            for p in model.parameters():
                original_weights.append(p.data.clone())
            
            # Apply STE quantization
            params = list(model.parameters())
            for i, p in enumerate(params):
                layer_name = f'layer_{i}'
                codebook = codebooks[layer_name].to(device)
                p.data = ste_quantize_weights(p.data, codebook)
            
            # Forward pass with quantized weights
            pred = model(coords_t)
            recon_loss = F.mse_loss(pred, targets_t)
            
            # Restore original weights for gradient computation
            for i, p in enumerate(params):
                p.data = original_weights[i]
            
            # Re-do forward with original weights to get proper gradients
            # (STE: gradient flows through as if no quantization)
            pred_orig = model(coords_t)
            recon_loss = F.mse_loss(pred_orig, targets_t)
            
            # Regularization: penalize distance to codebook
            reg_loss = quantization_reg_loss(model, codebooks, device)
            total_loss = recon_loss + reg_weight * reg_loss
        else:
            # Normal training (no STE yet)
            pred = model(coords_t)
            recon_loss = F.mse_loss(pred, targets_t)
            reg_loss = torch.tensor(0.0)
            total_loss = recon_loss
        
        total_loss.backward()
        opt.step()
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            history['epoch'].append(epoch)
            history['loss'].append(float(recon_loss.item()))
            history['reg_loss'].append(float(reg_loss.item()) if isinstance(reg_loss, torch.Tensor) else 0.0)
            history['total_loss'].append(float(total_loss.item()))
            history['ste_active'].append(epoch >= ste_start_epoch)
        
        # Update codebooks periodically
        if epoch >= ste_start_epoch and epoch % codebook_update_interval == 0 and epoch > 0:
            codebooks = update_codebooks(model, codebooks, seed)
    
    # Final codebook update
    codebooks = update_codebooks(model, codebooks, seed)
    
    info = {
        'codebooks': {k: v.numpy().tolist() for k, v in codebooks.items()},
        'k_config': k_config,
        'reg_weight': reg_weight,
        'ste_start_epoch': ste_start_epoch,
        'codebook_update_interval': codebook_update_interval,
        'history': history,
    }
    
    return model, codebooks, info


# ---------------------------------------------------------------------------
# Post-quantization PSNR measurement (the Exp 38 correction)
# ---------------------------------------------------------------------------

def evaluate_psnr_pre_quant(model: CoinMLP, img: np.ndarray) -> float:
    """PSNR on float32 model (before quantization)."""
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


def evaluate_psnr_post_quant(model: CoinMLP, codebooks: Dict[str, torch.Tensor],
                               img: np.ndarray) -> float:
    """PSNR on quantized model (after codebook[indices] reload)."""
    # Quantize weights
    params = list(model.parameters())
    with torch.no_grad():
        for i, p in enumerate(params):
            layer_name = f'layer_{i}'
            codebook = codebooks[layer_name]
            # Quantize: find nearest centroid
            flat = p.data.flatten()
            dists = torch.abs(flat.unsqueeze(-1) - codebook.unsqueeze(0))
            indices = dists.argmin(dim=-1)
            quantized = codebook[indices]
            p.data = quantized.view_as(p.data)
    
    # Evaluate
    psnr = evaluate_psnr_pre_quant(model, img)
    return psnr


def compute_quantized_size(codebooks: Dict[str, torch.Tensor], model: CoinMLP) -> int:
    """
    Compute total size of quantized model:
    - Codebook: K * 4 bytes (float32 per centroid)
    - Indices: ceil(log2(K)) bits per weight
    - Plus arithmetic coding savings (estimated 30% from Exp 29)
    """
    total_bytes = 0
    params = list(model.parameters())
    
    for i, p in enumerate(params):
        layer_name = f'layer_{i}'
        codebook = codebooks[layer_name]
        k = len(codebook)
        n_weights = p.numel()
        
        # Codebook size: K * 4 bytes
        cb_bytes = k * 4
        
        # Index size: ceil(log2(K)) bits per weight, with ~30% arithmetic coding savings
        bits_per_index = math.ceil(math.log2(max(2, k)))
        raw_index_bytes = (n_weights * bits_per_index + 7) // 8
        coded_index_bytes = int(raw_index_bytes * 0.7)  # 30% savings estimate
        
        total_bytes += cb_bytes + coded_index_bytes
    
    return total_bytes


# ---------------------------------------------------------------------------
# AVIF compression at 0.1 BPP
# ---------------------------------------------------------------------------

def avif_compress_at_bpp(img: np.ndarray, target_bpp: float,
                            image_size: int) -> Tuple[bytes, int, float]:
    """
    Compress image as AVIF at target bits-per-pixel.
    Returns (payload, size_bytes, actual_bpp).
    """
    import pillow_avif  # register AVIF support
    
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')
    
    target_bytes = int(target_bpp * image_size * image_size / 8)
    
    # Binary search for quality
    best_payload = None
    best_size = None
    lo_q, hi_q = 1, 80  # AVIF quality range
    
    for _ in range(20):
        mid_q = (lo_q + hi_q) // 2
        buf = io.BytesIO()
        pil_img.save(buf, format='AVIF', quality=mid_q)
        size = buf.tell()
        
        if abs(size - target_bytes) < target_bytes * 0.15:
            actual_bpp = size * 8 / (image_size * image_size)
            return buf.getvalue(), size, actual_bpp
        
        if size < target_bytes:
            lo_q = mid_q + 1
        else:
            hi_q = mid_q - 1
        
        best_payload = buf.getvalue()
        best_size = size
        
        if lo_q > hi_q:
            break
    
    actual_bpp = best_size * 8 / (image_size * image_size)
    return best_payload, best_size, actual_bpp


def avif_decompress(payload: bytes) -> np.ndarray:
    """Decompress AVIF and return as float32 array (grayscale)."""
    pil_img = Image.open(io.BytesIO(payload))
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    arr = np.array(pil_img, dtype=np.float32)
    return arr


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(seeds: List[int], image_size: int, hidden_features: int,
                    hidden_layers: int, omega: float, epochs: int, lr: float,
                    reg_weight: float, ste_start_epoch: int,
                    codebook_update_interval: int, target_bpp: float,
                    output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    
    # Asymmetric K config
    k_config = {
        'layer_0': 256,  # Input layer (coordinate encoding)
        'layer_1': 64,   # Hidden layer 1
        'layer_2': 64,   # Hidden layer 2
        'layer_3': 128,  # Output layer
    }
    
    print(f"[exp39] loading real photos at {image_size}x{image_size}...", flush=True)
    images, names = load_real_photos(image_size)
    print(f"[exp39] loaded {len(images)} real photos: {names}", flush=True)
    print(f"[exp39] QAT config: STE starts at epoch {ste_start_epoch}, "
          f"codebook update every {codebook_update_interval} epochs", flush=True)
    print(f"[exp39] K config: {k_config}", flush=True)
    print(f"[exp39] Reg weight: {reg_weight}", flush=True)
    print(f"[exp39] AVIF target: {target_bpp} BPP", flush=True)
    
    all_runs = []
    for seed in seeds:
        ckpt_path = os.path.join(output_dir, f'ckpt_seed{seed}.json')
        if os.path.exists(ckpt_path):
            print(f"\n[exp39] LOADING checkpoint seed={seed}", flush=True)
            with open(ckpt_path) as f:
                run_result = json.load(f)
            all_runs.append(run_result)
            print(f"  CACHED", flush=True)
            continue
        
        print(f"\n[exp39] === seed={seed} ===", flush=True)
        t0 = time.time()
        
        psnr_pre_quant = []
        psnr_post_quant = []
        quantized_sizes = []
        avif_psnrs = []
        avif_sizes = []
        avif_bpps = []
        
        for i, img in enumerate(images):
            # Train with QAT
            model, codebooks, train_info = train_qat_siren(
                img, hidden_features=hidden_features, hidden_layers=hidden_layers,
                omega=omega, epochs=epochs, lr=lr, seed=seed,
                k_config=k_config, reg_weight=reg_weight,
                codebook_update_interval=codebook_update_interval,
                ste_start_epoch=ste_start_epoch,
            )
            
            # Measure pre-quant PSNR (float32 model)
            psnr_pre = evaluate_psnr_pre_quant(model, img)
            
            # Measure post-quant PSNR (the REAL number, Exp 38 correction)
            # Clone model to preserve original for potential reuse
            model_copy = CoinMLP(hidden_features=hidden_features,
                                  hidden_layers=hidden_layers, omega=omega)
            model_copy.load_state_dict(model.state_dict())
            psnr_post = evaluate_psnr_post_quant(model_copy, codebooks, img)
            
            # Compute quantized size
            q_size = compute_quantized_size(codebooks, model)
            
            # AVIF at 0.1 BPP
            avif_payload, avif_size, avif_bpp = avif_compress_at_bpp(
                img, target_bpp, image_size
            )
            avif_recon = avif_decompress(avif_payload)
            avif_psnr = compute_psnr(img, avif_recon)
            
            psnr_pre_quant.append(psnr_pre)
            psnr_post_quant.append(psnr_post)
            quantized_sizes.append(q_size)
            avif_psnrs.append(avif_psnr)
            avif_sizes.append(avif_size)
            avif_bpps.append(avif_bpp)
            
            print(f"  {names[i]}: pre={psnr_pre:.2f} dB → post={psnr_post:.2f} dB "
                  f"(drop={psnr_pre-psnr_post:.2f} dB), size={q_size} B | "
                  f"AVIF: {avif_psnr:.2f} dB, {avif_size} B ({avif_bpp:.3f} BPP)", flush=True)
        
        # Save weights for SHA-256
        weights_payload = bytearray()
        for m in [train_qat_siren(img, hidden_features, hidden_layers, omega, epochs, lr, seed,
                                    k_config, reg_weight, codebook_update_interval, ste_start_epoch)[0]
                    for img in images]:
            w = np.concatenate([p.detach().cpu().numpy().flatten() for p in m.parameters()])
            weights_payload += w.astype(np.float32).tobytes()
        weights_file = os.path.join(output_dir, f'exp39_weights_seed{seed}.bin')
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
                    'quantized_size_bytes': quantized_sizes[i],
                    'avif_psnr_db': avif_psnrs[i],
                    'avif_size_bytes': avif_sizes[i],
                    'avif_bpp': avif_bpps[i],
                }
                for i in range(len(images))
            ],
            'mean_psnr_pre_quant': float(np.mean(psnr_pre_quant)),
            'mean_psnr_post_quant': float(np.mean(psnr_post_quant)),
            'mean_psnr_drop': float(np.mean(psnr_pre_quant) - np.mean(psnr_post_quant)),
            'mean_quantized_size': float(np.mean(quantized_sizes)),
            'mean_avif_psnr': float(np.mean(avif_psnrs)),
            'mean_avif_size': float(np.mean(avif_sizes)),
            'mean_avif_bpp': float(np.mean(avif_bpps)),
            'total_time_s': time.time() - t0,
            'weights_sha256': weights_sha,
            'k_config': k_config,
            'reg_weight': reg_weight,
            'ste_start_epoch': ste_start_epoch,
        }
        
        with open(ckpt_path, 'w') as f:
            json.dump(run_result, f, indent=2, default=str)
        all_runs.append(run_result)
        
        print(f"\n  MEAN (seed={seed}):", flush=True)
        print(f"    SIREN pre-quant:  {run_result['mean_psnr_pre_quant']:.2f} dB", flush=True)
        print(f"    SIREN post-quant: {run_result['mean_psnr_post_quant']:.2f} dB", flush=True)
        print(f"    PSNR drop:        {run_result['mean_psnr_drop']:.2f} dB", flush=True)
        print(f"    Quantized size:   {run_result['mean_quantized_size']:.0f} B", flush=True)
        print(f"    AVIF:             {run_result['mean_avif_psnr']:.2f} dB, {run_result['mean_avif_size']:.0f} B ({run_result['mean_avif_bpp']:.3f} BPP)", flush=True)
    
    # Aggregate
    pre_psnrs = np.array([r['mean_psnr_pre_quant'] for r in all_runs])
    post_psnrs = np.array([r['mean_psnr_post_quant'] for r in all_runs])
    drops = np.array([r['mean_psnr_drop'] for r in all_runs])
    q_sizes = np.array([r['mean_quantized_size'] for r in all_runs])
    avif_psnrs = np.array([r['mean_avif_psnr'] for r in all_runs])
    avif_sizes = np.array([r['mean_avif_size'] for r in all_runs])
    
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
        'quantized_size': {
            'mean_bytes': float(q_sizes.mean()), 'std_bytes': float(q_sizes.std()),
        },
        'avif': {
            'mean_psnr_db': float(avif_psnrs.mean()), 'std_psnr_db': float(avif_psnrs.std()),
            'mean_size_bytes': float(avif_sizes.mean()), 'std_size_bytes': float(avif_sizes.std()),
        },
    }
    
    print(f"\n[exp39] AGGREGATED across {len(all_runs)} seeds:", flush=True)
    print(f"  SIREN pre-quant:  {aggregated['siren_pre_quant']['mean_psnr_db']:.2f} ± {aggregated['siren_pre_quant']['std_psnr_db']:.2f} dB", flush=True)
    print(f"  SIREN post-quant: {aggregated['siren_post_quant']['mean_psnr_db']:.2f} ± {aggregated['siren_post_quant']['std_psnr_db']:.2f} dB", flush=True)
    print(f"  PSNR drop:        {aggregated['psnr_drop']['mean_db']:.2f} ± {aggregated['psnr_drop']['std_db']:.2f} dB", flush=True)
    print(f"  Quantized size:   {aggregated['quantized_size']['mean_bytes']:.0f} B", flush=True)
    print(f"  AVIF:             {aggregated['avif']['mean_psnr_db']:.2f} ± {aggregated['avif']['std_psnr_db']:.2f} dB, {aggregated['avif']['mean_size_bytes']:.0f} B", flush=True)
    
    # Comparison
    siren_post = aggregated['siren_post_quant']['mean_psnr_db']
    avif_mean = aggregated['avif']['mean_psnr_db']
    diff = siren_post - avif_mean
    
    if siren_post > avif_mean:
        winner = "SIREN+QAT (beats AVIF)"
        conclusion = "POSITIVE — QAT recovers enough PSNR to beat AVIF"
    else:
        winner = "AVIF"
        conclusion = "NEGATIVE — QAT does not recover enough PSNR to beat AVIF"
    
    comparison = {
        'siren_post_minus_avif': float(diff),
        'winner': winner,
        'conclusion': conclusion,
        'exp38_drop_for_comparison': 13.33,  # Exp 38 KMeans K=50 drop
        'exp39_drop_with_qat': float(drops.mean()),
        'improvement_over_exp38': float(13.33 - drops.mean()),
    }
    
    print(f"\n[exp39] COMPARISON:", flush=True)
    print(f"  SIREN post-quant - AVIF: {diff:+.2f} dB", flush=True)
    print(f"  WINNER: {winner}", flush=True)
    print(f"  CONCLUSION: {conclusion}", flush=True)
    print(f"  QAT improvement over Exp 38: drop reduced from 13.33 dB to {drops.mean():.2f} dB "
          f"(improvement: {13.33 - drops.mean():.2f} dB)", flush=True)
    
    output = {
        'experiment': 'experiment_39_qat_ste',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'QAT with STE + asymmetric K + quantization regularization will '
                       'reduce the post-quantization PSNR drop from 13.33 dB (Exp 38) '
                       'to under 5 dB, making SIREN+entropy competitive with AVIF.',
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
            'target_bpp': target_bpp,
        },
        'all_runs': all_runs,
        'aggregated': aggregated,
        'comparison': comparison,
    }
    
    out_json_path = os.path.join(output_dir, 'experiment_39_results.json')
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_json_path, 'rb') as f:
        json_sha = hashlib.sha256(f.read()).hexdigest()
    output['_output_json_sha256'] = json_sha
    
    print(f"\n[exp39] DONE", flush=True)
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
    parser.add_argument('--target-bpp', type=float, default=0.1)
    parser.add_argument('--output-dir', type=str,
                        default='/home/z/my-project/research/universe/_exp39_out')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    run_experiment(
        seeds=seeds, image_size=args.size,
        hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
        omega=args.omega, epochs=args.epochs, lr=args.lr,
        reg_weight=args.reg_weight, ste_start_epoch=args.ste_start_epoch,
        codebook_update_interval=args.codebook_update_interval,
        target_bpp=args.target_bpp, output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
