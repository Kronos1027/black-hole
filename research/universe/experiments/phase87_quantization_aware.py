# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 87: Quantization-Aware Training for SIREN
=================================================
BHUH Phase II Wave 5

CONTEXT
-------
Phase 85 (distillation, 32× reduction) and Phase 86 (multi-resolution, 8.3×
reduction) both reduce PARAMETER COUNT. But another axis is BITLENGTH:
- Standard SIREN: float32 (4 bytes/param)
- INT8 quantized: 1 byte/param (4× reduction)
- INT4 quantized: 0.5 bytes/param (8× reduction)
- Ternary {-1, 0, 1}: ~1.58 bits/param (20× reduction vs float32)

The production BLKH already uses INT8. This phase tests AGGRESSIVE
quantization: INT4 and ternary, with quantization-aware training (QAT).

HYPOTHESIS
----------
Quantization-aware training can recover most of the quality lost by
post-training quantization:
- Standard SIREN float32 → INT8 PTQ: typically loses 2-5 dB
- QAT INT8: recovers ~80% of that loss
- QAT INT4: should achieve >25 dB on smooth images
- QAT ternary: should achieve >20 dB (extreme compression)

EXPERIMENT
----------
1. Train SIREN with quantization-aware forward pass (STE for gradients)
2. Quantization levels: float32, INT8, INT4, ternary
3. Compare PSNR and effective seed size
4. Combined with Phase 85 distillation: distilled INT4 SIREN

PREDICTION
----------
- INT4 QAT: >28 dB PSNR, 8× size reduction vs float32
- Ternary QAT: >22 dB PSNR, 20× reduction
- Combined with distillation: 32× (distill) × 8× (INT4) = 256× total reduction

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import time
import json
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from phase85_knowledge_distillation import (make_smooth_image, psnr,
                                             siren_param_count)


def quantize_ste(x, bits, symmetric=True):
    """Straight-through estimator quantization.
    Forward: quantize to bits
    Backward: identity (pass gradient through)
    """
    import torch
    if bits >= 32:
        return x  # no quantization
    levels = 2 ** bits - 1
    if symmetric:
        # Symmetric quantization: [-max, max] -> {-levels/2, ..., 0, ..., levels/2}
        max_val = x.abs().max().detach()
        scale = max_val / (levels / 2)
        x_scaled = x / (scale + 1e-9)
        x_rounded = torch.round(x_scaled)
        x_clipped = torch.clamp(x_rounded, -levels/2, levels/2)
        x_dequant = x_clipped * scale
    else:
        # Asymmetric: [min, max] -> {0, 1, ..., levels}
        min_val = x.min().detach()
        max_val = x.max().detach()
        scale = (max_val - min_val) / levels
        x_scaled = (x - min_val) / (scale + 1e-9)
        x_rounded = torch.round(x_scaled)
        x_clipped = torch.clamp(x_rounded, 0, levels)
        x_dequant = x_clipped * scale + min_val
    # STE: forward = quantized, backward = identity
    return x + (x_dequant - x).detach()


def quantize_ternary(x, threshold=0.05):
    """Ternary quantization: {-1, 0, 1} with STE.
    Values below threshold -> 0
    Values above threshold -> sign(x) * mean(|x|)
    """
    import torch
    max_val = x.abs().max().detach()
    threshold_val = threshold * max_val
    # Ternary: 0 if |x| < threshold, else sign(x)
    x_ternary = torch.where(
        x.abs() < threshold_val,
        torch.zeros_like(x),
        torch.sign(x) * max_val  # use max as scale
    )
    return x + (x_ternary - x).detach()


def train_siren_qat(coords, target, hidden=32, n_layers=3, omega=15.0,
                    epochs=800, lr=1e-3, quant_bits=32, ternary=False):
    """Train SIREN with quantization-aware training."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    class SirenQAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            d = 2
            for k in range(n_layers - 1):
                self.layers.append(nn.Linear(d, hidden))
                d = hidden
            self.head = nn.Linear(hidden, 1)
            self.omega = omega
            self.quant_bits = quant_bits
            self.ternary = ternary
            # SIREN init
            for i, layer in enumerate(self.layers):
                bound = 1.0 / layer.in_features if i == 0 else np.sqrt(6.0 / layer.in_features) / omega
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.uniform_(-bound, bound)
            bound = np.sqrt(6.0 / self.head.in_features) / omega
            self.head.weight.data.uniform_(-bound, bound)
            self.head.bias.data.uniform_(-bound, bound)

        def quantize_params(self):
            """Apply quantization to all parameters."""
            if self.ternary:
                quants = []
                for layer in self.layers:
                    quants.append((quantize_ternary(layer.weight), quantize_ternary(layer.bias)))
                quants.append((quantize_ternary(self.head.weight), quantize_ternary(self.head.bias)))
                return quants
            elif self.quant_bits >= 32:
                return None  # no quantization
            else:
                quants = []
                for layer in self.layers:
                    quants.append((quantize_ste(layer.weight, self.quant_bits),
                                   quantize_ste(layer.bias, self.quant_bits)))
                quants.append((quantize_ste(self.head.weight, self.quant_bits),
                               quantize_ste(self.head.bias, self.quant_bits)))
                return quants

        def forward(self, x):
            quants = self.quantize_params()
            h = x
            for i, layer in enumerate(self.layers):
                w, b = (quants[i] if quants else (layer.weight, layer.bias))
                h = torch.sin(self.omega * torch.nn.functional.linear(h, w, b))
            w_h, b_h = (quants[-1] if quants else (self.head.weight, self.head.bias))
            return torch.nn.functional.linear(h, w_h, b_h)

    model = SirenQAT()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    yt = torch.tensor(target.flatten(), dtype=torch.float32)

    for ep in range(epochs):
        opt.zero_grad()
        pred = model(xt).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()

    # Final eval (with quantization applied)
    model.eval()
    with torch.no_grad():
        pred = model(xt).squeeze(-1).numpy()
    return model, float(loss.detach()), pred


def effective_seed_size(n_params, quant_bits, ternary=False):
    """Compute effective seed size in bytes."""
    if ternary:
        # Ternary: 2 values per byte (log2(3) ≈ 1.58 bits)
        bits_per_param = np.log2(3)
    else:
        bits_per_param = min(quant_bits, 32)
    total_bits = n_params * bits_per_param
    return int(np.ceil(total_bits / 8))


def run_phase87():
    print("=" * 72)
    print("PHASE 87: Quantization-Aware Training for SIREN")
    print("=" * 72)
    print()

    import torch  # noqa

    N_PIX = 32
    coords = np.stack(np.meshgrid(np.linspace(0, 1, N_PIX),
                                  np.linspace(0, 1, N_PIX)), axis=-1).reshape(-1, 2)

    targets = {
        'gaussian': make_smooth_image(N_PIX, 'gaussian'),
        'sin':      make_smooth_image(N_PIX, 'sin'),
        'plane':    make_smooth_image(N_PIX, 'plane'),
    }

    HIDDEN = 32
    n_params = siren_param_count(HIDDEN)
    print(f"SIREN: hidden={HIDDEN}, params={n_params}")
    print()

    # Quantization configurations
    configs = [
        ('float32', 32, False),
        ('INT8_QAT', 8, False),
        ('INT4_QAT', 4, False),
        ('Ternary_QAT', None, True),
    ]

    all_results = {}

    for tname, target in targets.items():
        print(f"\n--- Target: {tname} ---")
        target_results = []
        for config_name, bits, ternary in configs:
            t0 = time.time()
            model, loss, pred = train_siren_qat(coords, target, hidden=HIDDEN,
                                                  epochs=800, lr=1e-3,
                                                  quant_bits=bits, ternary=ternary)
            t_train = time.time() - t0
            p = psnr(target, pred)
            seed_size = effective_seed_size(n_params, bits if bits else 32, ternary)
            target_results.append({
                'config': config_name,
                'bits_per_param': (np.log2(3) if ternary else bits),
                'seed_size_bytes': seed_size,
                'psnr_db': p,
                'train_time_s': t_train,
                'loss': loss,
            })
            print(f"  {config_name:<15}: PSNR={p:.1f}dB, seed={seed_size}B, "
                  f"bits/param={np.log2(3) if ternary else bits:.1f}, time={t_train:.2f}s")
        all_results[tname] = target_results

    # ============================================================
    # SUMMARY
    # ============================================================
    print()
    print("=" * 72)
    print("QUANTIZATION RESULTS")
    print("=" * 72)
    print(f"{'Target':<10}", end='')
    for config_name, _, _ in configs:
        print(f" {config_name:>14}", end='')
    print()
    for tname, results in all_results.items():
        print(f"{tname:<10}", end='')
        for r in results:
            print(f" {r['psnr_db']:>5.1f}dB/{r['seed_size_bytes']:>5d}B", end='')
        print()

    # ============================================================
    # ANALYSIS
    # ============================================================
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()

    # Compare to float32 baseline
    for tname, results in all_results.items():
        float32_psnr = results[0]['psnr_db']
        float32_size = results[0]['seed_size_bytes']
        print(f"  {tname} (float32 baseline: {float32_psnr:.1f}dB, {float32_size}B):")
        for r in results[1:]:
            psnr_loss = float32_psnr - r['psnr_db']
            size_reduction = float32_size / r['seed_size_bytes']
            print(f"    {r['config']:<15}: -{psnr_loss:.1f}dB PSNR, {size_reduction:.1f}x smaller")
        print()

    # Check if INT4 achieves >25 dB
    int4_psnrs = [next(r['psnr_db'] for r in results if r['config'] == 'INT4_QAT')
                   for results in all_results.values()]
    ternary_psnrs = [next(r['psnr_db'] for r in results if r['config'] == 'Ternary_QAT')
                      for results in all_results.values()]

    int4_success = sum(1 for p in int4_psnrs if p > 25)
    ternary_success = sum(1 for p in ternary_psnrs if p > 20)

    print(f"  INT4 QAT PSNR > 25 dB: {int4_success}/{len(int4_psnrs)}")
    print(f"  Ternary QAT PSNR > 20 dB: {ternary_success}/{len(ternary_psnrs)}")
    print(f"  INT4 average PSNR: {np.mean(int4_psnrs):.1f} dB")
    print(f"  Ternary average PSNR: {np.mean(ternary_psnrs):.1f} dB")

    # Compute combined reduction potential
    int4_size = next(r['seed_size_bytes'] for r in all_results['gaussian'] if r['config'] == 'INT4_QAT')
    float32_size = next(r['seed_size_bytes'] for r in all_results['gaussian'] if r['config'] == 'float32')
    int4_reduction = float32_size / int4_size
    distill_reduction = 32.0  # from Phase 85
    combined = int4_reduction * distill_reduction

    print()
    print(f"  Combined reduction potential:")
    print(f"    Phase 85 distillation: {distill_reduction:.1f}x")
    print(f"    INT4 QAT: {int4_reduction:.1f}x")
    print(f"    Combined (distill + INT4): {combined:.0f}x")

    if int4_success >= 2:
        verdict = (f"VALIDATED — Quantization-aware training works. INT4 QAT achieves "
                   f"{np.mean(int4_psnrs):.1f} dB average PSNR with {int4_reduction:.1f}x size "
                   f"reduction. Combined with Phase 85 distillation ({distill_reduction:.0f}x), "
                   f"total reduction potential is {combined:.0f}x. "
                   "Axiom 16 (Quantization Compression) accepted.")
        print()
        print("NEW AXIOM (Axiom 16 — Quantization Compression):")
        print("  SIREN weights can be quantized to INT4 (or ternary) via QAT with")
        print("  acceptable PSNR loss. Combined with distillation, enables extreme")
        print("  compression ratios (>200x vs float32 monolithic).")
    elif int4_success >= 1:
        verdict = "PARTIAL — INT4 works for some images but not all."
    else:
        verdict = "INVALID — INT4 quantization destroys SIREN quality."

    print(f"\nVerdict: {verdict}")

    return {
        'phase': 87,
        'name': 'Quantization-Aware Training',
        'verdict': verdict,
        'n_params': n_params,
        'int4_avg_psnr_db': float(np.mean(int4_psnrs)),
        'ternary_avg_psnr_db': float(np.mean(ternary_psnrs)),
        'int4_reduction_x': float(int4_reduction),
        'combined_reduction_x': float(combined),
        'all_results': all_results,
    }


if __name__ == '__main__':
    result = run_phase87()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
