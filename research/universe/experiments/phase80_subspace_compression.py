# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 80: Subspace Compression — Projecting to the Effective Rank-k Manifold
============================================================================
BHUH Phase II Wave 3

CONTEXT
-------
Phase 76 showed SIREN with P=337 parameters has Fisher effective rank only
22.4 (6.7% of nominal). This means ~93% of parameters are in "nuisance"
directions that barely affect output.

HYPOTHESIS (Axiom 11 — Subspace Compression)
--------------------------------------------
If we PROJECT trained SIREN parameters onto the top-k eigenvectors of the
Fisher Information Matrix, we can:
  1. Reduce stored seed size from P to k (15× reduction for k=22, P=337)
  2. Maintain acceptable PSNR (>25 dB) for smooth images
  3. The "true" BHUH seed lives in this k-dim subspace

This is the BHUH analog of PCA for neural networks.

EXPERIMENT
----------
1. Train SIREN on smooth image → θ (P=337 params)
2. Compute Fisher Information Matrix F(θ) → eigenvalues λ_i, eigenvectors v_i
3. Project: θ_k = Σ_{i=1}^k (θ·v_i) v_i  (top-k projection)
4. Reconstruct image from θ_k
5. Measure PSNR vs k
6. Find minimum k that maintains >25 dB PSNR

PREDICTION
----------
- k=22 (Fisher effective rank): PSNR should be >30 dB
- k=10: PSNR should be >25 dB (effective rank is conservative)
- k=5: PSNR should drop significantly
- This validates that the "true" seed is k-dimensional, not P-dimensional

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import time
import json
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from phase78_universal_ancestry import fit_siren
from phase76_information_geometry import compute_fisher_diagonal


def make_smooth_image(n, kind='gaussian', cx=0.5, cy=0.5, sigma=0.15):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    if kind == 'gaussian':
        return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    if kind == 'sin':
        return (0.5 + 0.3 * np.sin(2 * np.pi * (X + Y))).astype(np.float32)
    if kind == 'plane':
        return (0.3 + 0.4 * X + 0.2 * Y).astype(np.float32)
    raise ValueError(kind)


def psnr(orig, pred):
    err = orig.flatten() - pred.flatten()
    mse = float(np.mean(err ** 2))
    pmax = float(orig.max())
    return 10 * np.log10(pmax ** 2 / max(mse, 1e-12))


def load_siren_with_params(coords, theta, hidden=32, n_layers=3, omega=15.0):
    """Reconstruct SIREN model from flattened params."""
    import torch
    import torch.nn as nn

    class Siren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            d = 2
            for k in range(n_layers - 1):
                self.layers.append(nn.Linear(d, hidden))
                d = hidden
            self.head = nn.Linear(hidden, 1)
            self.omega = omega

        def forward(self, x):
            h = x
            for layer in self.layers:
                h = torch.sin(self.omega * layer(h))
            return self.head(h)

    model = Siren()
    idx = 0
    for p in model.parameters():
        n = int(np.prod(p.shape))
        with torch.no_grad():
            p.copy_(torch.tensor(theta[idx:idx+n].reshape(p.shape), dtype=torch.float32))
        idx += n
    return model


def predict_with_params(coords, theta, hidden=32, n_layers=3, omega=15.0):
    import torch
    model = load_siren_with_params(coords, theta, hidden, n_layers, omega)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(coords, dtype=torch.float32)).squeeze(-1).numpy()


def project_to_subspace(theta, eigvecs, k):
    """Project θ onto top-k eigenvectors."""
    V_k = eigvecs[:, :k]  # (P, k)
    coords = V_k.T @ theta  # (k,)
    return V_k @ coords  # (P,) — back to full space


def run_phase80():
    print("=" * 72)
    print("PHASE 80: Subspace Compression — Projecting to Effective Rank-k Manifold")
    print("=" * 72)
    print()

    import torch  # noqa
    torch.manual_seed(42)

    N_PIX = 16
    coords = np.stack(np.meshgrid(np.linspace(0, 1, N_PIX),
                                  np.linspace(0, 1, N_PIX)), axis=-1).reshape(-1, 2)

    # ============================================================
    # Step 1: Train SIREN on 3 different smooth images
    # ============================================================
    targets = {
        'gaussian': make_smooth_image(N_PIX, 'gaussian', cx=0.5, cy=0.5, sigma=0.15),
        'sin':      make_smooth_image(N_PIX, 'sin'),
        'plane':    make_smooth_image(N_PIX, 'plane'),
    }

    print("--- Step 1: Train SIREN on each target ---")
    trained = {}
    for name, target in targets.items():
        theta, loss = fit_siren(coords, target, hidden=32, n_layers=3, omega=15.0,
                                epochs=500, lr=1e-3)
        pred_full = predict_with_params(coords, theta)
        trained[name] = {
            'theta': theta,
            'target': target,
            'psnr_full': psnr(target, pred_full),
        }
        print(f"  {name}: full PSNR = {trained[name]['psnr_full']:.1f} dB ({len(theta)} params)")

    # ============================================================
    # Step 2: For each target, compute Fisher at THAT target's θ
    # (Fisher is local — different θ values have different Fishers)
    # ============================================================
    print()
    print("--- Step 2: Compute Fisher Information Matrix at each θ ---")
    fisher_per_target = {}
    for name, data in trained.items():
        model = load_siren_with_params(coords, data['theta'], hidden=32, n_layers=3, omega=15.0)
        _, F_mat, n_params = compute_fisher_diagonal(model, coords, data['target'], n_samples=64)
        eigvals, eigvecs = np.linalg.eigh(F_mat)
        idx_sort = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]
        fisher_per_target[name] = {
            'eigvals': eigvals,
            'eigvecs': eigvecs,
            'n_params': n_params,
        }
        print(f"  {name}: top eigenvalue={eigvals[0]:.3f}, "
              f"bottom={eigvals[-1]:.3e}, cond#={eigvals[0]/max(eigvals[-1],1e-30):.3e}")

    # ============================================================
    # Step 3: Project each θ to ITS OWN top-k Fisher subspace, measure PSNR
    # ============================================================
    print()
    print("--- Step 3: Project to OWN top-k Fisher subspace, measure PSNR ---")
    k_values = [5, 10, 15, 20, 25, 30, 50, 100, 200, 337]

    results = {name: [] for name in targets}
    for name in targets:
        theta = trained[name]['theta']
        target = trained[name]['target']
        psnr_full = trained[name]['psnr_full']
        eigvecs = fisher_per_target[name]['eigvecs']
        n_params_local = fisher_per_target[name]['n_params']
        print(f"\n  Target: {name} (full PSNR: {psnr_full:.1f} dB, {n_params_local} params)")
        print(f"    {'k':>5} {'PSNR':>10} {'ΔPSNR':>10} {'Reduction':>10}")
        for k in k_values:
            if k > len(theta):
                continue
            theta_proj = project_to_subspace(theta, eigvecs, k)
            pred_proj = predict_with_params(coords, theta_proj)
            p = psnr(target, pred_proj)
            delta = p - psnr_full
            reduction = len(theta) / k
            results[name].append({
                'k': k,
                'psnr_proj': p,
                'delta_psnr': delta,
                'reduction_x': reduction,
            })
            print(f"    {k:>5} {p:>9.1f}dB {delta:>+9.1f}dB {reduction:>9.1f}x")

    # ============================================================
    # Step 4: Find minimum k that maintains PSNR > 25 dB
    # ============================================================
    print()
    print("--- Step 4: Minimum k for PSNR > 25 dB ---")
    for name, res_list in results.items():
        min_k = None
        for r in res_list:
            if r['psnr_proj'] > 25:
                min_k = r['k']
                break
        if min_k:
            reduction = len(trained[name]['theta']) / min_k
            print(f"  {name}: min k = {min_k} (reduction {reduction:.1f}x)")
        else:
            print(f"  {name}: no k maintains PSNR > 25 dB")

    # ============================================================
    # Step 5: Effective rank comparison
    # ============================================================
    print()
    print("--- Step 5: Effective rank comparison ---")
    # Shannon effective rank (averaged across targets)
    eff_ranks = []
    for name, data in fisher_per_target.items():
        ev = np.maximum(data['eigvals'], 1e-30)
        p = ev / ev.sum()
        H = -np.sum(p * np.log(p + 1e-30))
        eff_ranks.append(np.exp(H))
        print(f"  {name}: eff_rank = {np.exp(H):.2f}")
    eff_rank = float(np.mean(eff_ranks))
    print(f"  Mean effective rank: {eff_rank:.2f}")
    print(f"  Intrinsic dim fraction: {eff_rank/n_params:.1%}")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()
    print("SUBSPACE COMPRESSION RESULTS:")
    print(f"  Original parameter count: {n_params}")
    print(f"  Fisher effective rank:    {eff_rank:.1f}")
    print(f"  Compression ratio at eff_rank: {n_params/eff_rank:.1f}x")
    print()

    # Find k where all 3 targets still get >25 dB
    min_k_all = None
    for k in k_values:
        if k > n_params:
            continue
        all_ok = all(
            next(r['psnr_proj'] for r in results[name] if r['k'] == k) > 25
            for name in results
        )
        if all_ok:
            min_k_all = k
            break

    if min_k_all:
        reduction = n_params / min_k_all
        print(f"  Minimum k maintaining >25 dB on ALL targets: k={min_k_all}")
        print(f"  Achievable reduction: {reduction:.1f}x")
        verdict = (f"VALIDATED — Subspace compression works. Projecting to top-{min_k_all} "
                   f"Fisher eigenvectors maintains >25 dB PSNR on all 3 smooth image types. "
                   f"Compression ratio: {reduction:.1f}x reduction in seed size "
                   f"({n_params} → {min_k_all} params). "
                   "Axiom 11 (Subspace Compression) accepted.")
    else:
        # Find best k
        best_k = None
        best_min_psnr = -np.inf
        for k in k_values:
            if k > n_params:
                continue
            min_psnr = min(
                next(r['psnr_proj'] for r in results[name] if r['k'] == k)
                for name in results
            )
            if min_psnr > best_min_psnr:
                best_min_psnr = min_psnr
                best_k = k
        verdict = (f"INVALID — Fisher subspace projection FAILS. Best k={best_k} achieves only "
                   f"{best_min_psnr:.1f} dB min PSNR (target was >25 dB). SIREN is too nonlinear "
                   "for linear Fisher subspace projection. The Fisher effective rank measures "
                   "LOCAL sensitivity (for small perturbations), but projection onto top-k "
                   "Fisher eigenvectors involves LARGE perturbations that exceed the linear "
                   "regime. The 'effective rank' theorem (Phase 76) is correct as a LOCAL "
                   "property, but CANNOT be exploited for global compression via linear projection.")

    print(f"\nVerdict: {verdict}")
    print()
    if 'VALIDATED' in verdict:
        print("NEW AXIOM (Axiom 11 — Subspace Compression):")
        print("  The 'true' BHUH seed lives in a k-dimensional subspace of parameter space,")
        print("  where k = effective_rank(Fisher(θ)) << |θ|. Compression by projection onto")
        print("  the top-k Fisher eigenvectors preserves output quality with")
        print("  |θ|/k parameter reduction.")
        print()
        print("  Formal: ∃ V_k ∈ ℝ^(P×k) such that PSNR(Genesis(V_k V_k^T θ), Genesis(θ)) > 25 dB")
        print("  where k = O(eff_rank(F(θ))) << P")
    else:
        print("NEGATIVE FINDING — Axiom 11 (Subspace Compression) REJECTED:")
        print("  Linear projection onto Fisher top-k eigenvectors does NOT preserve output.")
        print("  The Fisher effective rank (Phase 76) is a LOCAL property: it describes how")
        print("  small perturbations affect output, NOT how large projections behave.")
        print()
        print("  IMPLICATIONS:")
        print("  1. Phase 76's theorem stands (effective rank is real, just LOCAL)")
        print("  2. Global compression requires NONLINEAR methods (e.g., distillation,")
        print("     pruning with retraining, or hypernetwork conditioning)")
        print("  3. Linear PCA-style projection is insufficient for SIREN")
        print("  4. This is consistent with the nonlinear nature of SIREN (sin activations)")

    return {
        'phase': 80,
        'name': 'Subspace Compression',
        'verdict': verdict,
        'n_params': int(n_params),
        'fisher_effective_rank': float(eff_rank),
        'min_k_for_25dB': int(min_k_all) if min_k_all else None,
        'reduction_x': float(n_params / min_k_all) if min_k_all else None,
        'results': {name: res_list for name, res_list in results.items()},
    }


if __name__ == '__main__':
    result = run_phase80()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
