# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 72: Self-Modifying Universes
==================================
BHUH Phase II — Beyond Singularities

MOTIVATION
----------
A real "universe" of files is not static — new files arrive over time.
BHUH Phase I assumed the corpus was fixed (train once, compress all).
A TRULY universe-like system should:

  1. Accept new files WITHOUT retraining the base network
  2. Preserve compression quality of old files
  3. Pay only O(1) cost per new file (modulation update)

This is the difference between a "fixed snapshot" and a "living universe".

HYPOTHESIS
----------
For new file f_{N+1}, we can fit ONLY a new modulation vector
γ_{N+1} while keeping the base SIREN Φ frozen. The cost is O(1) per
new file (one modulation vector) instead of O(N) retraining.

EXPERIMENT
----------
1. Train base Φ on files 1..N0
2. Add files N0+1, N0+2, ... incrementally
3. For each new file, fit ONLY γ (modulation vector, ~64 params)
4. Measure: PSNR of old files (should be unchanged) + PSNR of new file
5. Compare total time vs full retrain

PREDICTION
----------
- Old files: PSNR unchanged (Φ frozen)
- New files: PSNR slightly lower (γ-only fit), but ≥20 dB
- Time: γ-only fit ~10x faster than full retrain
- Compression ratio preserved

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import time
import json
import io
import zlib


def make_image(n, kind, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    if kind == 'gaussian':
        cx, cy = 0.3 + 0.4 * np.random.rand(), 0.3 + 0.4 * np.random.rand()
        return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) * 8).astype(np.float32)
    if kind == 'sin':
        fx, fy = 1 + 3 * np.random.rand(), 1 + 3 * np.random.rand()
        return (0.5 + 0.3 * np.sin(2 * np.pi * (fx * X + fy * Y))).astype(np.float32)
    if kind == 'plane':
        a, b, c = np.random.rand(3)
        return (a * X + b * Y + c * 0.3).astype(np.float32)
    raise ValueError(kind)


class FiLMSiren:
    """SIREN with FiLM modulation: output = Φ(x; θ_base, γ) where γ is per-file."""

    def __init__(self, in_dim=2, hidden=64, n_layers=3, mod_dim=32, omega=15.0):
        import torch
        import torch.nn as nn
        self.torch = torch

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                d = in_dim
                for k in range(n_layers - 1):
                    self.layers.append(nn.Linear(d, hidden))
                    d = hidden
                self.head = nn.Linear(hidden, 1)
                # FiLM: per-layer scale/bias from modulation vector
                self.film_proj = nn.ModuleList([
                    nn.Linear(mod_dim, 2 * hidden) for _ in range(n_layers - 1)
                ])
                self.omega = omega
                # SIREN init
                for i, layer in enumerate(self.layers):
                    bound = 1.0 / layer.in_features if i == 0 else np.sqrt(6.0 / layer.in_features) / omega
                    layer.weight.data.uniform_(-bound, bound)
                    layer.bias.data.uniform_(-bound, bound)

            def forward(self, x, gamma):
                h = x
                for i, layer in enumerate(self.layers):
                    film = self.film_proj[i](gamma)
                    scale, bias = film[:, :h.shape[-1] if i == 0 else self.layers[i-1].out_features], \
                                  film[:, self.layers[i-1].out_features if i > 0 else 0:]
                    # Simpler: project to 2*hidden, take first/second half
                    out_dim = layer.out_features
                    scale = film[:, :out_dim]
                    bias = film[:, out_dim:2*out_dim]
                    h = torch.sin(self.omega * layer(h)) * (1 + 0.1 * scale) + 0.1 * bias
                return self.head(h)

        self.net = Net()
        self.mod_dim = mod_dim

    def fit_base(self, X, files, epochs=800, lr=1e-3):
        """Train base + per-file modulations jointly on all files."""
        torch = self.torch
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        xt = torch.tensor(X, dtype=torch.float32)
        # Per-file modulation vectors
        gammas = [torch.zeros(self.mod_dim, requires_grad=True) for _ in files]
        # Add gammas to optimizer
        opt = torch.optim.Adam(list(self.net.parameters()) + gammas, lr=lr)
        targets = [torch.tensor(f.flatten(), dtype=torch.float32) for f in files]
        for ep in range(epochs):
            opt.zero_grad()
            loss = 0
            for g, yt in zip(gammas, targets):
                # Broadcast gamma to batch
                g_batch = g.unsqueeze(0).expand(xt.shape[0], -1)
                pred = self.net(xt, g_batch).squeeze(-1)
                loss = loss + ((pred - yt) ** 2).mean()
            loss.backward()
            opt.step()
        self.gammas = [g.detach().numpy() for g in gammas]
        return float(loss.detach())

    def fit_new_modulation(self, X, target, epochs=300, lr=5e-3):
        """Add new file: fit ONLY γ, freeze base net."""
        torch = self.torch
        for p in self.net.parameters():
            p.requires_grad = False
        gamma_new = torch.zeros(self.mod_dim, requires_grad=True)
        opt = torch.optim.Adam([gamma_new], lr=lr)
        xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(target.flatten(), dtype=torch.float32)
        for ep in range(epochs):
            opt.zero_grad()
            g_batch = gamma_new.unsqueeze(0).expand(xt.shape[0], -1)
            pred = self.net(xt, g_batch).squeeze(-1)
            loss = ((pred - yt) ** 2).mean()
            loss.backward()
            opt.step()
        # Restore gradients
        for p in self.net.parameters():
            p.requires_grad = True
        self.gammas.append(gamma_new.detach().numpy())
        return float(loss.detach())

    def predict(self, X, gamma):
        torch = self.torch
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32)
            g = torch.tensor(gamma, dtype=torch.float32)
            g_batch = g.unsqueeze(0).expand(xt.shape[0], -1)
            return self.net(xt, g_batch).squeeze(-1).numpy()


def psnr(orig, pred):
    err = orig.flatten() - pred.flatten()
    mse = float(np.mean(err ** 2))
    pmax = float(orig.max())
    return 10 * np.log10(pmax ** 2 / max(mse, 1e-12))


def run_phase72():
    print("=" * 72)
    print("PHASE 72: Self-Modifying Universes (Incremental Learning)")
    print("=" * 72)
    print()

    import torch  # noqa
    torch.manual_seed(0)

    N_PIX = 32
    N0 = 4  # initial universe size
    N_NEW = 6  # files to add incrementally
    coords = np.stack(np.meshgrid(np.linspace(0, 1, N_PIX),
                                  np.linspace(0, 1, N_PIX)), axis=-1).reshape(-1, 2)

    # Generate files
    all_files = [make_image(N_PIX, k, seed=i)
                 for i, k in enumerate(['gaussian', 'sin', 'plane',
                                        'gaussian', 'sin', 'plane',
                                        'gaussian', 'sin', 'plane', 'gaussian'])]
    initial_files = all_files[:N0]
    new_files = all_files[N0:N0 + N_NEW]

    # === STAGE 1: Train base universe on N0 initial files ===
    print(f"--- Stage 1: Train base on {N0} initial files ---")
    t0 = time.time()
    model = FiLMSiren(in_dim=2, hidden=64, n_layers=3, mod_dim=32, omega=15.0)
    loss = model.fit_base(coords, initial_files, epochs=800, lr=1e-3)
    t_base = time.time() - t0
    print(f"  Base training: {t_base:.1f}s, final loss={loss:.6f}")

    # Measure initial PSNR
    psnr_initial = []
    for i, f in enumerate(initial_files):
        pred = model.predict(coords, model.gammas[i])
        psnr_initial.append(psnr(f, pred))
    print(f"  Initial PSNR: {[f'{p:.1f}dB' for p in psnr_initial]}")

    # === STAGE 2: Add new files incrementally (γ-only fit) ===
    print()
    print(f"--- Stage 2: Add {N_NEW} files incrementally (γ-only fit) ---")
    psnr_new_gamma = []
    times_gamma = []
    for i, f in enumerate(new_files):
        t0 = time.time()
        model.fit_new_modulation(coords, f, epochs=300, lr=5e-3)
        dt = time.time() - t0
        times_gamma.append(dt)
        pred = model.predict(coords, model.gammas[-1])
        p = psnr(f, pred)
        psnr_new_gamma.append(p)
        print(f"  New file {i+1}: PSNR={p:.1f}dB, time={dt:.2f}s")

    # === STAGE 3: Verify old files still work (Φ frozen) ===
    print()
    print("--- Stage 3: Verify old files unchanged ---")
    psnr_old_after = []
    for i, f in enumerate(initial_files):
        pred = model.predict(coords, model.gammas[i])
        psnr_old_after.append(psnr(f, pred))
    drift = [abs(a - b) for a, b in zip(psnr_initial, psnr_old_after)]
    print(f"  Old PSNR before: {[f'{p:.1f}' for p in psnr_initial]}")
    print(f"  Old PSNR after:  {[f'{p:.1f}' for p in psnr_old_after]}")
    print(f"  Max drift: {max(drift):.4f} dB")

    # === STAGE 4: Compare to full retrain ===
    print()
    print(f"--- Stage 4: Full retrain baseline (all {N0+N_NEW} files from scratch) ---")
    t0 = time.time()
    full_model = FiLMSiren(in_dim=2, hidden=64, n_layers=3, mod_dim=32, omega=15.0)
    full_model.fit_base(coords, all_files, epochs=800, lr=1e-3)
    t_full = time.time() - t0
    print(f"  Full retrain: {t_full:.1f}s")
    psnr_full = []
    for i, f in enumerate(all_files):
        pred = full_model.predict(coords, full_model.gammas[i])
        psnr_full.append(psnr(f, pred))
    print(f"  PSNR (full): min={min(psnr_full):.1f}dB, mean={np.mean(psnr_full):.1f}dB")

    # === SUMMARY ===
    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    total_gamma_time = sum(times_gamma)
    incremental_total = t_base + total_gamma_time
    speedup = t_full / max(incremental_total, 1e-6)

    print(f"Base training ({N0} files):          {t_base:.2f}s")
    print(f"Incremental addition ({N_NEW} files): {total_gamma_time:.2f}s")
    print(f"  (avg per file: {np.mean(times_gamma):.2f}s)")
    print(f"Incremental total:                    {incremental_total:.2f}s")
    print(f"Full retrain ({N0+N_NEW} files):       {t_full:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    print()
    print(f"PSNR old files (before/after γ-fit): {np.mean(psnr_initial):.1f} → {np.mean(psnr_old_after):.1f} dB")
    print(f"Max drift on old files:              {max(drift):.4f} dB (target: <0.01)")
    print(f"PSNR new files (γ-only):             min={min(psnr_new_gamma):.1f}, mean={np.mean(psnr_new_gamma):.1f} dB")
    print(f"PSNR full retrain:                   min={min(psnr_full):.1f}, mean={np.mean(psnr_full):.1f} dB")

    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    drift_ok = max(drift) < 0.1
    new_ok = min(psnr_new_gamma) > 20
    speedup_ok = speedup > 2

    print(f"Old files preserved: {'✅' if drift_ok else '❌'} (max drift {max(drift):.4f} dB)")
    print(f"New files learnable: {'✅' if new_ok else '❌'} (min PSNR {min(psnr_new_gamma):.1f} dB)")
    print(f"Speedup vs retrain:  {'✅' if speedup_ok else '❌'} ({speedup:.2f}x)")

    if drift_ok and new_ok and speedup_ok:
        verdict = ("VALIDATED — Universe is self-modifying: new files added at O(1) cost via "
                   "γ-only fit, old files preserved bit-for-bit, faster than full retrain. "
                   "BHUH gains a 6th axiom: SELF-MODIFICATION.")
    elif drift_ok and new_ok:
        verdict = ("PARTIAL — Self-modification works but no speedup. γ-only fit is correct but "
                   "not faster than full retrain for small N.")
    else:
        verdict = "INVALID — Self-modification fails."

    print(f"\nVerdict: {verdict}")
    print()
    print("PROPOSED NEW AXIOM:")
    print("  Axiom 6 (Self-Modification): A BHUH universe can accept new files")
    print("  without retraining the base network. The cost per new file is O(1)")
    print("  (one modulation vector), and existing files are preserved exactly.")
    print()
    print("  Formal: ∀f_new: ∃γ_new: Φ(x; γ_new) ≈ f_new ∧ Φ unchanged ∧ |γ_new| = O(1)")

    return {
        'phase': 72,
        'name': 'Self-Modifying Universes',
        'verdict': verdict,
        'n_initial': N0,
        'n_new': N_NEW,
        'base_train_time_s': t_base,
        'incremental_total_time_s': total_gamma_time,
        'full_retrain_time_s': t_full,
        'speedup_x': speedup,
        'max_drift_db': max(drift),
        'min_psnr_new_db': float(min(psnr_new_gamma)),
        'mean_psnr_new_db': float(np.mean(psnr_new_gamma)),
        'min_psnr_full_db': float(min(psnr_full)),
    }


if __name__ == '__main__':
    result = run_phase72()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
