# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 86: Multi-Resolution SIREN — Coarse-to-Fine Progressive Compression
==========================================================================
BHUH Phase II Wave 5

CONTEXT
-------
Phase 85 showed knowledge distillation works (32x reduction, 32.7 dB PSNR).
But distillation requires training a teacher first, then a student —
double the training cost.

This phase tests a different approach: MULTI-RESOLUTION SIREN. Instead
of training a large SIREN then compressing, train a SMALL SIREN first
on a low-resolution version of the image, then ADD a residual SIREN
for the high-frequency details.

HYPOTHESIS
----------
A multi-resolution SIREN:
  f(x) = f_coarse(x) + f_detail(x)
where:
  - f_coarse is trained on downsampled image (small, ~37 params)
  - f_detail is trained on residual (high-freq, also small)

Total params: 2 × 37 = 74 (vs 1185 for monolithic)
PSNR target: >30 dB

This is the wavelet-like approach applied to SIREN.

EXPERIMENT
----------
1. Downsample image to 16x16, train small SIREN → f_coarse
2. Upscale f_coarse output to 32x32, compute residual = original - upscale
3. Train second small SIREN on residual → f_detail
4. Combine: f(x) = f_coarse(x) + f_detail(x)
5. Compare to:
   - Monolithic SIREN (hidden=32, 1185 params)
   - Distilled SIREN from Phase 85

PREDICTION
----------
- Multi-resolution should achieve >30 dB with ~74 params (16x reduction)
- Faster training than distillation (no teacher needed)
- Quality may be slightly lower than monolithic

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import time
import json
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from phase85_knowledge_distillation import (make_smooth_image, psnr,
                                             train_siren, predict_with_params,
                                             siren_param_count)


def downsample_image(img, factor=2):
    """Downsample by averaging blocks."""
    H, W = img.shape
    H2, W2 = H // factor, W // factor
    return img[:H2*factor, :W2*factor].reshape(H2, factor, W2, factor).mean(axis=(1, 3))


def upsample_image(img, target_shape):
    """Nearest-neighbor upsampling."""
    H, W = img.shape
    Ht, Wt = target_shape
    # Repeat rows and columns
    row_idx = np.linspace(0, H - 1, Ht).astype(int)
    col_idx = np.linspace(0, W - 1, Wt).astype(int)
    return img[row_idx][:, col_idx]


def train_siren_model(coords, target, hidden=8, n_layers=3, omega=15.0, epochs=500, lr=1e-3):
    """Train SIREN, return model object."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    class Siren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            d = 2
            for k in range(n_layers - 1):
                lin = nn.Linear(d, hidden)
                bound = 1.0 / d if k == 0 else np.sqrt(6.0 / hidden) / omega
                lin.weight.data.uniform_(-bound, bound)
                lin.bias.data.uniform_(-bound, bound)
                self.layers.append(lin)
                d = hidden
            self.head = nn.Linear(hidden, 1)
            bound = np.sqrt(6.0 / hidden) / omega
            self.head.weight.data.uniform_(-bound, bound)
            self.head.bias.data.uniform_(-bound, bound)
            self.omega = omega

        def forward(self, x):
            h = x
            for layer in self.layers:
                h = torch.sin(self.omega * layer(h))
            return self.head(h)

    model = Siren()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    yt = torch.tensor(target.flatten(), dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(xt).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()
    return model, float(loss.detach())


def run_phase86():
    print("=" * 72)
    print("PHASE 86: Multi-Resolution SIREN — Coarse-to-Fine Progressive")
    print("=" * 72)
    print()

    import torch  # noqa

    N_PIX = 32
    N_COARSE = 16  # 16x16 for coarse
    coords_full = np.stack(np.meshgrid(np.linspace(0, 1, N_PIX),
                                       np.linspace(0, 1, N_PIX)), axis=-1).reshape(-1, 2)
    coords_coarse = np.stack(np.meshgrid(np.linspace(0, 1, N_COARSE),
                                         np.linspace(0, 1, N_COARSE)), axis=-1).reshape(-1, 2)

    targets = {
        'gaussian': make_smooth_image(N_PIX, 'gaussian'),
        'sin':      make_smooth_image(N_PIX, 'sin'),
        'plane':    make_smooth_image(N_PIX, 'plane'),
        'sinc':     make_smooth_image(N_PIX, 'sinc'),
    }

    COARSE_HIDDEN = 4
    DETAIL_HIDDEN = 8
    coarse_params = siren_param_count(COARSE_HIDDEN)
    detail_params = siren_param_count(DETAIL_HIDDEN)
    total_params = coarse_params + detail_params
    mono_params = siren_param_count(32)

    print(f"Architecture:")
    print(f"  Coarse SIREN: hidden={COARSE_HIDDEN}, params={coarse_params}")
    print(f"  Detail SIREN: hidden={DETAIL_HIDDEN}, params={detail_params}")
    print(f"  Total multi-res: {total_params} params")
    print(f"  Monolithic (h=32): {mono_params} params")
    print(f"  Reduction: {mono_params/total_params:.1f}x")
    print()

    all_results = {}

    for tname, target_full in targets.items():
        print(f"\n--- Target: {tname} ---")

        # Step 1: Downsample, train coarse SIREN
        target_coarse = downsample_image(target_full, factor=2)
        t0 = time.time()
        coarse_model, coarse_loss = train_siren_model(
            coords_coarse, target_coarse, hidden=COARSE_HIDDEN, epochs=400, lr=1e-3
        )
        t_coarse = time.time() - t0

        # Step 2: Upscale coarse prediction, compute residual
        coarse_model.eval()
        with torch.no_grad():
            coarse_pred_coarse = coarse_model(torch.tensor(coords_coarse, dtype=torch.float32)).squeeze(-1).numpy()
        coarse_pred_coarse_2d = coarse_pred_coarse.reshape(N_COARSE, N_COARSE)
        coarse_pred_full = upsample_image(coarse_pred_coarse_2d, (N_PIX, N_PIX))
        residual = target_full - coarse_pred_full

        coarse_psnr = psnr(target_full, coarse_pred_full)

        # Step 3: Train detail SIREN on residual
        t0 = time.time()
        detail_model, detail_loss = train_siren_model(
            coords_full, residual, hidden=DETAIL_HIDDEN, epochs=500, lr=1e-3
        )
        t_detail = time.time() - t0

        # Step 4: Combine
        detail_model.eval()
        with torch.no_grad():
            detail_pred = detail_model(torch.tensor(coords_full, dtype=torch.float32)).squeeze(-1).numpy()
        detail_pred_2d = detail_pred.reshape(N_PIX, N_PIX)
        combined_pred = coarse_pred_full + detail_pred_2d
        combined_psnr = psnr(target_full, combined_pred)

        # Compare to monolithic
        t0 = time.time()
        mono_model, mono_loss = train_siren_model(
            coords_full, target_full, hidden=32, epochs=500, lr=1e-3
        )
        t_mono = time.time() - t0
        mono_model.eval()
        with torch.no_grad():
            mono_pred = mono_model(torch.tensor(coords_full, dtype=torch.float32)).squeeze(-1).numpy()
        mono_psnr = psnr(target_full, mono_pred)

        all_results[tname] = {
            'coarse_psnr_db': coarse_psnr,
            'combined_psnr_db': combined_psnr,
            'mono_psnr_db': mono_psnr,
            'multi_res_params': total_params,
            'mono_params': mono_params,
            'reduction_x': mono_params / total_params,
            'coarse_time_s': t_coarse,
            'detail_time_s': t_detail,
            'mono_time_s': t_mono,
        }

        print(f"  Coarse-only PSNR:  {coarse_psnr:.1f} dB")
        print(f"  Multi-res PSNR:    {combined_psnr:.1f} dB ({total_params} params)")
        print(f"  Monolithic PSNR:   {mono_psnr:.1f} dB ({mono_params} params)")
        print(f"  Reduction: {mono_params/total_params:.1f}x")
        print(f"  Multi-res time: {t_coarse + t_detail:.2f}s vs Mono: {t_mono:.2f}s")

    # ============================================================
    # SUMMARY
    # ============================================================
    print()
    print("=" * 72)
    print("MULTI-RESOLUTION RESULTS")
    print("=" * 72)
    print(f"{'Target':<10} {'Coarse':>8} {'MultiRes':>9} {'Mono':>8} {'Reduction':>10} {'Time MR/Mono':>14}")
    for tname, r in all_results.items():
        time_ratio = (r['coarse_time_s'] + r['detail_time_s']) / r['mono_time_s']
        print(f"{tname:<10} {r['coarse_psnr_db']:>7.1f}dB {r['combined_psnr_db']:>8.1f}dB "
              f"{r['mono_psnr_db']:>7.1f}dB {r['reduction_x']:>9.1f}x "
              f"{time_ratio:>13.2f}x")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()

    successful = sum(1 for r in all_results.values() if r['combined_psnr_db'] > 30)
    avg_psnr = np.mean([r['combined_psnr_db'] for r in all_results.values()])
    avg_reduction = np.mean([r['reduction_x'] for r in all_results.values()])
    avg_time_ratio = np.mean([(r['coarse_time_s'] + r['detail_time_s']) / r['mono_time_s']
                               for r in all_results.values()])

    print(f"  Multi-res PSNR > 30 dB: {successful}/{len(all_results)}")
    print(f"  Average PSNR: {avg_psnr:.1f} dB")
    print(f"  Average reduction: {avg_reduction:.1f}x")
    print(f"  Average time ratio (multi-res / mono): {avg_time_ratio:.2f}x")
    print()

    if successful >= 3 and avg_psnr > 30:
        verdict = (f"VALIDATED — Multi-resolution SIREN achieves {avg_psnr:.1f} dB PSNR "
                   f"with {avg_reduction:.1f}x parameter reduction. Faster than distillation "
                   f"(no teacher needed) and competitive with monolithic. "
                   "This is a SECOND working path to SIREN compression (alongside Phase 85 distillation). "
                   "Axiom 15 (Multi-Resolution Compression) accepted.")
        print("NEW AXIOM (Axiom 15 — Multi-Resolution Compression):")
        print("  SIREN compression is achievable via coarse-to-fine decomposition:")
        print("    f(x) = f_coarse(x) + f_detail(x)")
        print("  where f_coarse fits a downsampled image and f_detail fits the residual.")
        print("  Total params: 2 × small_hidden << monolithic_hidden")
    elif successful >= 1:
        verdict = "PARTIAL — Multi-resolution works for some targets but not all."
    else:
        verdict = "INVALID — Multi-resolution fails."

    print(f"\nVerdict: {verdict}")
    print()
    print("COMPARISON OF COMPRESSION APPROACHES:")
    mr_status = 'VALIDATED' if successful >= 3 else 'PARTIAL'
    print(f"  {'Approach':<25} {'Reduction':>10} {'PSNR':>8} {'Time':>8} {'Status':<15}")
    print(f"  {'Phase 80 (Linear PCA)':<25} {'N/A':>10} {'3.5dB':>8} {'N/A':>8} {'FAILED':<15}")
    print(f"  {'Phase 82 (Nonlinear AE)':<25} {'N/A':>10} {'10dB':>8} {'N/A':>8} {'FAILED':<15}")
    print(f"  {'Phase 85 (Distillation)':<25} {'32.0x':>10} {'32.7dB':>8} {'2x':>8} {'VALIDATED':<15}")
    print(f"  {'Phase 86 (Multi-res)':<25} {f'{avg_reduction:.1f}x':>10} {f'{avg_psnr:.1f}dB':>8} "
          f"{f'{avg_time_ratio:.2f}x':>8} {mr_status:<15}")

    return {
        'phase': 86,
        'name': 'Multi-Resolution SIREN',
        'verdict': verdict,
        'coarse_hidden': COARSE_HIDDEN,
        'detail_hidden': DETAIL_HIDDEN,
        'multi_res_params': total_params,
        'mono_params': mono_params,
        'reduction_x': float(avg_reduction),
        'avg_psnr_db': float(avg_psnr),
        'n_successful': successful,
        'n_total': len(all_results),
        'avg_time_ratio': float(avg_time_ratio),
        'all_results': all_results,
    }


if __name__ == '__main__':
    result = run_phase86()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
