# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 1 Experiment: Bit-Perfect Multi-File SIREN
==================================================
Extends phase1_multi_file_siren.py with bit-perfect residual coding.

The basic Multi-File SIREN gives lossy compression (SIREN approximation
+ float32 weights). This experiment adds:

1. INT8 quantization of weights (like BLKH production)
2. PNG residual for bit-perfect reconstruction
3. Measures total bit-perfect size vs lossy size

HYPOTHESIS:
  Bit-perfect Multi-File SIREN will still achieve 3-10x improvement
  over separate bit-perfect SIRENs, because the shared base network
  dominates the size even with residual overhead.

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import io
import struct
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
from phase1_multi_file_siren import SIREN, ModulatedSIREN, generate_satellite_images, get_coordinates


def quantize_int8(weights_dict):
    """Quantize weights to INT8 with per-tensor scale."""
    all_vals = np.concatenate([v.detach().cpu().numpy().ravel() for v in weights_dict.values()])
    max_abs = max(np.abs(all_vals).max(), 1e-8)
    scale = max_abs / 127.0

    quantized = {}
    for k, v in weights_dict.items():
        w = v.detach().cpu().numpy()
        q = np.round(w / scale).astype(np.int8)
        quantized[k] = q
    return quantized, scale


def dequantize_int8(quantized, scale):
    """Dequantize INT8 weights."""
    return {k: torch.from_numpy(v.astype(np.float32) * scale) for k, v in quantized.items()}


def compress_weights_zlib(quantized):
    """Compress quantized weights with zlib."""
    buf = bytearray()
    for k in sorted(quantized.keys()):
        buf.extend(quantized[k].tobytes())
    return zlib.compress(bytes(buf), 9)


def compute_residual(original, predicted):
    """Compute residual image for bit-perfect reconstruction."""
    # residual = (original - predicted) mod 256
    residual = (original.astype(np.int16) - predicted.astype(np.int16)) % 256
    return residual.astype(np.uint8)


def compress_residual_png(residual):
    """Compress residual with PNG lossless."""
    buf = io.BytesIO()
    Image.fromarray(residual).save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def reconstruct_with_residual(predicted, residual):
    """Reconstruct original from predicted + residual."""
    return ((predicted.astype(np.int16) + residual.astype(np.int16)) % 256).astype(np.uint8)


def train_and_compress_bitperfect(images, epochs=100, device='cpu', verbose=False):
    """Train multi-file SIREN and create bit-perfect compression."""
    n_files = len(images)
    size = images[0].shape[0]
    coords = get_coordinates(size, device)

    # Train model
    from phase1_multi_file_siren import train_multi_file_siren
    model, loss = train_multi_file_siren(images, epochs=epochs, device=device, verbose=verbose)

    # Quantize weights to INT8
    weights_dict = dict(model.named_parameters())
    quantized, scale = quantize_int8(weights_dict)

    # Compress quantized weights
    weights_compressed = compress_weights_zlib(quantized)

    # Dequantize and compute predictions for each image
    dequant_weights = dequantize_int8(quantized, scale)
    model.load_state_dict(dequant_weights)
    model.eval()

    # Compute residuals for each image
    total_residual_size = 0
    all_sha_ok = True
    with torch.no_grad():
        for i, img in enumerate(images):
            pred = model(coords, i)
            pred_img = (pred.cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)
            residual = compute_residual(img, pred_img)

            # Verify reconstruction
            rec = reconstruct_with_residual(pred_img, residual)
            sha_orig = hashlib.sha256(img.tobytes()).hexdigest()
            sha_rec = hashlib.sha256(rec.tobytes()).hexdigest()
            if sha_orig != sha_rec:
                all_sha_ok = False
                if verbose:
                    print(f"  ⚠️ SHA mismatch on image {i}")

            # Compress residual
            res_png = compress_residual_png(residual)
            total_residual_size += len(res_png)

    # Total bit-perfect size
    total_size = len(weights_compressed) + total_residual_size + 8  # +8 for scale

    return {
        'weights_size': len(weights_compressed),
        'residual_size': total_residual_size,
        'total_size': total_size,
        'bit_perfect': all_sha_ok,
        'training_loss': loss,
    }


def run_bitperfect_experiment(n_images=20, size=128, epochs=100, verbose=True):
    """Run bit-perfect experiment."""
    print("=" * 80)
    print("🧪 Phase 1: Bit-Perfect Multi-File SIREN")
    print("=" * 80)
    print(f"\nConfig: {n_images} images @ {size}x{size}, {epochs} epochs")

    # Generate images
    images = generate_satellite_images(n_images, size)
    print(f"Generated {n_images} satellite-like images")

    # ZIP baseline
    zip_total = sum(len(zlib.compress(img.tobytes(), 9)) for img in images)

    # Bit-perfect BHUH
    print(f"\nTraining bit-perfect Multi-File SIREN...")
    t0 = time.time()
    result = train_and_compress_bitperfect(images, epochs=epochs, verbose=verbose)
    dt = time.time() - t0

    print(f"\n{'='*60}")
    print("📊 BIT-PERFECT RESULTS")
    print(f"{'='*60}")
    print(f"  Weights (INT8 + zlib):  {result['weights_size']:>10,}B")
    print(f"  Residuals (PNG):        {result['residual_size']:>10,}B")
    print(f"  Total bit-perfect:      {result['total_size']:>10,}B")
    print(f"  Bit-perfect verified:   {'✅ YES' if result['bit_perfect'] else '❌ NO'}")
    print(f"  Training time:          {dt:.1f}s")
    print(f"\n  ZIP baseline:           {zip_total:>10,}B")
    print(f"  BHUH vs ZIP:            {zip_total/result['total_size']:.2f}x smaller")
    print(f"  Per-image overhead:     {result['total_size']/n_images:.0f}B/img")

    # Compare with lossy
    from phase1_multi_file_siren import run_experiment
    lossy_result = run_experiment(
        n_images=n_images, size=size,
        epochs_single=epochs, epochs_multi=epochs,
        verbose=False
    )

    print(f"\n  Lossy BHUH:             {lossy_result['bhuh_size']:>10,}B")
    print(f"  Bit-perfect overhead:   {result['total_size'] - lossy_result['bhuh_size']:>10,}B "
          f"({(result['total_size']/lossy_result['bhuh_size'] - 1)*100:.1f}%)")

    return result


if __name__ == '__main__':
    result = run_bitperfect_experiment(n_images=20, size=128, epochs=80, verbose=True)
