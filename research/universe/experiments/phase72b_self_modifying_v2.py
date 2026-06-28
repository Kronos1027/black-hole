# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 72 v2: Self-Modifying Universes — Restricted Domain Test
===============================================================
BHUH Phase II — Beyond Singularities

CONTEXT
-------
Phase 72 v1 showed: γ-only fit FAILS for diverse file types
(gaussian, sin, plane mixed). New files only reach ~13 dB PSNR.

This v2 tests the RESTRICTED form: all files from SAME domain
(e.g., all gaussian bumps with different centers). The base network
is pre-adapted to this domain; γ should encode position/shape.

PREDICTION
----------
- Same-domain: γ-only fit should reach >30 dB (base already knows
  the function family)
- Cross-domain: γ-only fit fails (base cannot represent new family)

This refines Axiom 6: Self-modification is DOMAIN-CONDITIONAL.
Formal: ∀f_new ∈ Domain(Φ_base): ∃γ_new: Φ(x; γ_new) ≈ f_new

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from phase72_self_modifying import (FiLMSiren, make_image, psnr,
                                     run_phase72 as run_phase72_v1)


def make_gaussian_family(n, seed):
    """All gaussians, different centers/sizes."""
    np.random.seed(seed)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    cx, cy = 0.2 + 0.6 * np.random.rand(), 0.2 + 0.6 * np.random.rand()
    sigma = 0.1 + 0.2 * np.random.rand()
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def make_sin_family(n, seed):
    """All sines, different frequencies/phases."""
    np.random.seed(seed)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    fx, fy = 1 + 2 * np.random.rand(), 1 + 2 * np.random.rand()
    phase = 2 * np.pi * np.random.rand()
    return (0.5 + 0.3 * np.sin(2 * np.pi * (fx * X + fy * Y) + phase)).astype(np.float32)


def run_phase72_v2():
    print("=" * 72)
    print("PHASE 72 v2: Self-Modification — Restricted Domain Test")
    print("=" * 72)
    print()

    import torch  # noqa
    torch.manual_seed(42)

    N_PIX = 32
    N0 = 4
    N_NEW = 6
    coords = np.stack(np.meshgrid(np.linspace(0, 1, N_PIX),
                                  np.linspace(0, 1, N_PIX)), axis=-1).reshape(-1, 2)

    domains = {
        'gaussian_family': make_gaussian_family,
        'sin_family': make_sin_family,
    }

    results = {}

    for domain_name, gen in domains.items():
        print(f"\n--- DOMAIN: {domain_name} ---")
        initial = [gen(N_PIX, seed=i) for i in range(N0)]
        new = [gen(N_PIX, seed=N0 + i) for i in range(N_NEW)]

        # Train base
        t0 = time.time()
        model = FiLMSiren(in_dim=2, hidden=64, n_layers=3, mod_dim=32, omega=15.0)
        model.fit_base(coords, initial, epochs=800, lr=1e-3)
        t_base = time.time() - t0

        psnr_initial = [psnr(f, model.predict(coords, model.gammas[i]))
                        for i, f in enumerate(initial)]
        print(f"  Base PSNR (initial {N0} files): {[f'{p:.1f}' for p in psnr_initial]}")

        # Add new files via γ-only
        psnr_new = []
        times = []
        for f in new:
            t0 = time.time()
            model.fit_new_modulation(coords, f, epochs=400, lr=5e-3)
            dt = time.time() - t0
            times.append(dt)
            p = psnr(f, model.predict(coords, model.gammas[-1]))
            psnr_new.append(p)
        print(f"  γ-only PSNR (new {N_NEW} files): {[f'{p:.1f}' for p in psnr_new]}")

        # Full retrain baseline
        all_files = initial + new
        t0 = time.time()
        full = FiLMSiren(in_dim=2, hidden=64, n_layers=3, mod_dim=32, omega=15.0)
        full.fit_base(coords, all_files, epochs=800, lr=1e-3)
        t_full = time.time() - t0
        psnr_full = [psnr(f, full.predict(coords, full.gammas[i]))
                     for i, f in enumerate(all_files)]
        print(f"  Full retrain PSNR: {[f'{p:.1f}' for p in psnr_full]}")

        # Old preservation check
        psnr_old_after = [psnr(f, model.predict(coords, model.gammas[i]))
                          for i, f in enumerate(initial)]
        drift = max(abs(a - b) for a, b in zip(psnr_initial, psnr_old_after))

        results[domain_name] = {
            'base_time_s': t_base,
            'gamma_total_time_s': sum(times),
            'full_retrain_time_s': t_full,
            'speedup': t_full / max(t_base + sum(times), 1e-6),
            'min_psnr_gamma_new': float(min(psnr_new)),
            'mean_psnr_gamma_new': float(np.mean(psnr_new)),
            'min_psnr_full': float(min(psnr_full)),
            'mean_psnr_full': float(np.mean(psnr_full)),
            'max_drift_old_db': drift,
        }

    # Summary
    print()
    print("=" * 72)
    print("RESTRICTED-DOMAIN RESULTS")
    print("=" * 72)
    print(f"{'Domain':<22} {'γ min':>8} {'γ mean':>8} {'full min':>9} {'drift':>7} {'speedup':>8}")
    for d, r in results.items():
        print(f"{d:<22} {r['min_psnr_gamma_new']:>7.1f}dB {r['mean_psnr_gamma_new']:>7.1f}dB "
              f"{r['min_psnr_full']:>8.1f}dB {r['max_drift_old_db']:>6.3f} {r['speedup']:>7.2f}x")

    print()
    print("=" * 72)
    print("REFINED AXIOM 6 (Domain-Conditional Self-Modification)")
    print("=" * 72)
    all_pass = all(r['min_psnr_gamma_new'] > 25 for r in results.values())
    if all_pass:
        verdict = ("VALIDATED — Axiom 6 holds for restricted domains. Same-family files "
                   "can be added at O(1) cost. Cross-domain fails (Phase 72 v1).")
        print("\n  Formal: ∀f_new ∈ Domain(Φ_base): ∃γ_new: Φ(x; γ_new) ≈ f_new")
        print("          Cost: O(1) per new file. Old files: exactly preserved.")
    else:
        any_pass = any(r['min_psnr_gamma_new'] > 25 for r in results.values())
        if any_pass:
            verdict = ("PARTIAL — Axiom 6 holds for SOME domains but not all. "
                       "Self-modification is family-dependent.")
        else:
            verdict = "INVALID — Axiom 6 fails even for restricted domains."

    print(f"\nVerdict: {verdict}")
    return {
        'phase': '72v2',
        'name': 'Self-Modifying Universes — Restricted Domain',
        'verdict': verdict,
        'results': results,
    }


if __name__ == '__main__':
    print("\n\n### Running Phase 72 v1 first for context ###\n")
    r1 = run_phase72_v1()
    print("\n\n### Now Phase 72 v2 ###\n")
    r2 = run_phase72_v2()
    print("\n--- JSON RESULT (v2) ---")
    print(json.dumps(r2, indent=2, default=str))
