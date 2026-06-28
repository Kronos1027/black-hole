# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 41: Seed Quantization Study (INT4/INT2/Binary)
======================================================
Tests extreme quantization of SIREN weights.

CONCEPT:
  INT8 quantization is standard. But what about:
  - INT4 (4-bit): 50% smaller, may lose quality
  - INT2 (2-bit): 75% smaller, likely degraded
  - Binary (1-bit): 87.5% smaller, extreme compression

  If INT4 works, we halve seed size with acceptable quality loss.

HYPOTHESIS:
  INT4 will maintain >25dB PSNR (usable quality).
  INT2 will degrade significantly (<15dB).
  Binary will produce garbage.

METHOD:
  1. Train SIREN on image
  2. Quantize to INT8, INT4, INT2, Binary
  3. Measure PSNR and seed size for each
  4. Find the "sweet spot" (best size/quality trade-off)

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, zlib, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren


def quantize_weights(model, bits=8, device='cpu'):
    """Quantize model weights to specified bit depth."""
    quantized = copy.deepcopy(model)
    n_levels = 2 ** bits

    with torch.no_grad():
        for param in quantized.parameters():
            w = param.detach().cpu().numpy()
            w_min, w_max = w.min(), w.max()
            if w_max - w_min < 1e-10:
                continue
            # Quantize
            scale = (w_max - w_min) / (n_levels - 1)
            q = np.round((w - w_min) / scale).astype(np.int32)
            # Dequantize
            dq = q * scale + w_min
            param.data = torch.from_numpy(dq.astype(np.float32)).to(device)

    return quantized


def measure_compressed_size(model, bits=8):
    """Measure compressed size of quantized weights."""
    weights_buf = bytearray()
    for param in model.parameters():
        w = param.detach().cpu().numpy()
        if bits <= 8:
            # Pack to uint8 or smaller
            n_levels = 2 ** bits
            w_min, w_max = w.min(), w.max()
            if w_max - w_min < 1e-10:
                q = np.zeros_like(w, dtype=np.uint8)
            else:
                scale = (w_max - w_min) / (n_levels - 1)
                q = np.round((w - w_min) / scale).astype(np.uint8)
            weights_buf.extend(q.tobytes())
        else:
            weights_buf.extend(w.tobytes())

    return len(zlib.compress(bytes(weights_buf), 9))


def query_model(model, size, device='cpu'):
    """Get model output as image."""
    coords = get_coordinates(size, device)
    with torch.no_grad():
        pred = model(coords)
    return (pred.cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)


def run_phase41_experiment(verbose=True):
    """Run Phase 41 Quantization Study."""
    print("=" * 80)
    print("🧪 Phase 41: Seed Quantization Study (INT4/INT2/Binary)")
    print("=" * 80)

    device = 'cpu'
    size = 128

    # Generate and train
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(2, 5, 2)
            img[:, :, c] += 50 * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    model, _ = train_single_siren(img, epochs=100, device=device, verbose=False)

    # Reference output (float32, no quantization)
    ref_img = query_model(model, size, device)

    # Test different bit depths
    bit_depths = [32, 16, 8, 4, 2, 1]
    results = []

    print(f"\n{'Bits':>6} {'Levels':>8} {'Comp Size':>10} {'PSNR':>8} {'vs FP32':>8} {'Visual':>10}")
    print("-" * 55)

    for bits in bit_depths:
        if bits == 32:
            # Original float32
            comp_size = measure_compressed_size(model, bits=32)
            output = ref_img
            psnr = 99.0
        else:
            quant_model = quantize_weights(model, bits=bits, device=device)
            comp_size = measure_compressed_size(quant_model, bits=bits)
            output = query_model(quant_model, size, device)
            mse = np.mean((ref_img.astype(float) - output.astype(float))**2)
            psnr = 10 * np.log10(255**2 / max(mse, 1e-10)) if mse > 0 else 99

        n_levels = 2 ** bits
        if psnr > 40:
            visual = "✅ clean"
        elif psnr > 25:
            visual = "⚠️ usable"
        elif psnr > 15:
            visual = "❌ degraded"
        else:
            visual = "💥 garbage"

        size_ratio = measure_compressed_size(model, bits=32) / max(comp_size, 1)
        print(f"{bits:>5} {n_levels:>7} {comp_size:>9,}B {psnr:>6.1f}dB {size_ratio:>6.2f}x {visual:>10}")

        results.append({
            'bits': bits,
            'levels': n_levels,
            'size': comp_size,
            'psnr': psnr,
            'size_ratio': size_ratio,
        })

    # Find sweet spot
    print(f"\n{'='*80}")
    print("📊 PHASE 41 SUMMARY — QUANTIZATION STUDY")
    print(f"{'='*80}")

    # INT8 is baseline (current production)
    int8 = next(r for r in results if r['bits'] == 8)
    int4 = next(r for r in results if r['bits'] == 4)
    int2 = next(r for r in results if r['bits'] == 2)

    print(f"\n  📋 Key findings:")
    print(f"  - FP32 (baseline): {results[0]['size']:,}B, {results[0]['psnr']:.1f}dB")
    print(f"  - INT8 (production): {int8['size']:,}B ({int8['size_ratio']:.2f}x smaller), {int8['psnr']:.1f}dB")
    print(f"  - INT4 (extreme): {int4['size']:,}B ({int4['size_ratio']:.2f}x smaller), {int4['psnr']:.1f}dB")

    if int4['psnr'] > 25:
        print(f"\n  ✅ INT4 is USABLE! {int4['size_ratio']:.2f}x smaller with {int4['psnr']:.1f}dB")
        print(f"     This could HALVE seed size in production!")
    else:
        print(f"\n  ⚠️  INT4 degrades quality ({int4['psnr']:.1f}dB < 25dB)")

    if int2['psnr'] > 15:
        print(f"  ✅ INT2 survives: {int2['psnr']:.1f}dB (degraded but recognizable)")
    else:
        print(f"  ❌ INT2 broken: {int2['psnr']:.1f}dB")

    print(f"\n  📋 Recommendations:")
    print(f"  - Production: INT8 (current, proven)")
    print(f"  - Aggressive: INT4 (if >{int4['psnr']:.0f}dB acceptable)")
    print(f"  - Extreme: INT2 (only for thumbnails/previews)")
    print(f"  - Binary: not viable for SIREN")

    return results


if __name__ == '__main__':
    results = run_phase41_experiment(verbose=True)
