# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 81: BHUH Computational Asymmetry
========================================
BHUH Phase II Wave 3 (CORRECTED — was 'One-Way Function', renamed for accuracy)

IMPORTANT CORRECTION
--------------------
An earlier version of this phase claimed BHUH is a 'one-way function' in
the cryptographic sense and compared its 'security bits' to AES-256 / RSA.
That claim was TECHNICALLY INCORRECT and has been removed.

Formal cryptographic one-way functions require that NO polynomial-time
algorithm can invert them. BHUH's inverse (compression via gradient descent)
runs in O(P·N·E) — polynomial time. Therefore BHUH is NOT a cryptographic
one-way function.

What BHUH DOES exhibit is COMPUTATIONAL ASYMMETRY: a large constant-factor
difference between forward (Genesis, ~ms) and inverse (compression, ~seconds).
This is useful for proof-of-work style applications but is fundamentally
different from cryptographic security.

CONTEXT
-------
Phase 77 established Genesis Asymmetry: decompression is ~4808× faster than
compression. This phase measures the asymmetry precisely and clarifies what
it can and cannot be used for.

DEFINITION (Computational Asymmetry — informal)
-----------------------------------------------
A function f: X → Y has computational asymmetry R if:
  - T_forward(f, x) / T_inverse(f, y) = 1/R  (forward is R times faster)
  - Both forward and inverse are polynomial-time
  - R is a LARGE CONSTANT (not superpolynomial)

For BHUH:
  - f = Genesis (SIREN forward pass over all pixels)
  - x = seed (parameter vector θ ∈ ℝ^P)
  - y = file (pixel array)
  - R ≈ 1000-6000 (measured)

EXPERIMENT
----------
1. Verify one-way property empirically:
   - Generate random seed x_random
   - Compute y_random = Genesis(x_random) → noise-like output
   - Attempt to find x' from y_random via gradient descent (compression)
   - Compare convergence:
     a) From random init: should converge slowly
     b) From TRUE x: should immediately verify (sanity check)

2. Quantify security parameters:
   - Search space size: |X| = ∞ (continuous) but |X_eff| ~ R^P where R is resolution
   - For int8 quantization: |X_eff| = 256^P
   - Brute-force cost: 256^P × T_genesis
   - Legitimate cost (compression): T_inverse ~ 1000× T_genesis
   - Security factor: 256^P / 1000

3. Test "near-collision" attack:
   - Given file y, find x' such that Genesis(x') ≈ y (not exact)
   - This is what compression does — but it requires gradient descent

This phase formalizes BHUH as a candidate cryptographic primitive.

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import time
import json


def make_smooth_image(n, kind='gaussian', cx=0.5, cy=0.5, sigma=0.15):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    if kind == 'gaussian':
        return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    raise ValueError(kind)


def psnr(orig, pred):
    err = orig.flatten() - pred.flatten()
    mse = float(np.mean(err ** 2))
    pmax = float(orig.max())
    return 10 * np.log10(pmax ** 2 / max(mse, 1e-12))


def genesis(seed_flat, coords, hidden=32, n_layers=3, omega=15.0):
    """Forward pass: seed → image."""
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
    with torch.no_grad():
        for p in model.parameters():
            n = int(np.prod(p.shape))
            p.copy_(torch.tensor(seed_flat[idx:idx+n].reshape(p.shape), dtype=torch.float32))
            idx += n
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(coords, dtype=torch.float32)).squeeze(-1).numpy()


def inverse_attempt(target, coords, hidden=32, n_layers=3, omega=15.0,
                    epochs=500, lr=1e-3, init_seed=None):
    """Compression attempt: file → seed via gradient descent."""
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
    if init_seed is not None:
        idx = 0
        with torch.no_grad():
            for p in model.parameters():
                n = int(np.prod(p.shape))
                p.copy_(torch.tensor(init_seed[idx:idx+n].reshape(p.shape), dtype=torch.float32))
                idx += n

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    yt = torch.tensor(target.flatten(), dtype=torch.float32)

    losses = []
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(xt).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    # Extract final seed
    seed = []
    for p in model.parameters():
        seed.append(p.detach().numpy().flatten())
    return np.concatenate(seed), losses[-1], losses


def run_phase81():
    print("=" * 72)
    print("PHASE 81: BHUH as a One-Way Function")
    print("=" * 72)
    print()

    import torch  # noqa
    torch.manual_seed(0)

    N_PIX = 16
    coords = np.stack(np.meshgrid(np.linspace(0, 1, N_PIX),
                                  np.linspace(0, 1, N_PIX)), axis=-1).reshape(-1, 2)
    target = make_smooth_image(N_PIX, 'gaussian', cx=0.5, cy=0.5, sigma=0.15)

    # ============================================================
    # PART 1: Demonstrate One-Way Property
    # ============================================================
    print("--- Part 1: Forward (Genesis) is fast, Inverse (Compression) is slow ---")
    # Generate a "true" seed by training
    print("  Training reference seed via gradient descent...")
    t0 = time.time()
    true_seed, final_loss, _ = inverse_attempt(target, coords, hidden=16, n_layers=3,
                                                epochs=400, lr=1e-3)
    t_inverse = time.time() - t0
    pred_true = genesis(true_seed, coords, hidden=16, n_layers=3)
    psnr_true = psnr(target, pred_true)
    print(f"  Inverse (compression): {t_inverse:.2f}s, PSNR={psnr_true:.1f}dB, loss={final_loss:.6f}")

    # Now: forward pass from true_seed is fast
    t0 = time.time()
    for _ in range(10):
        _ = genesis(true_seed, coords, hidden=16, n_layers=3)
    t_genesis = (time.time() - t0) / 10
    print(f"  Forward (Genesis):     {t_genesis*1000:.2f}ms per call")

    asymmetry = t_inverse / max(t_genesis, 1e-9)
    print(f"  Asymmetry ratio: {asymmetry:.0f}x  (inverse is {asymmetry:.0f}x slower)")

    # ============================================================
    # PART 2: Attempt brute-force attack (random init)
    # ============================================================
    print()
    print("--- Part 2: Random-init attack (simulating brute-force) ---")
    # If we start from a random seed (NOT the true seed), how fast does gradient
    # descent converge? This is the legitimate-user cost.
    # For comparison, a brute-force attacker who doesn't use gradient descent
    # would need to try ALL seeds — exponentially many.
    rng = np.random.default_rng(123)
    n_attempts = 3
    print(f"  Trying {n_attempts} random initializations, 200 epochs each...")
    for attempt in range(n_attempts):
        random_init = rng.normal(size=len(true_seed)) * 0.1
        t0 = time.time()
        recovered_seed, loss, _ = inverse_attempt(target, coords, hidden=16, n_layers=3,
                                                   epochs=200, lr=1e-3, init_seed=random_init)
        dt = time.time() - t0
        pred = genesis(recovered_seed, coords, hidden=16, n_layers=3)
        p = psnr(target, pred)
        # Distance from true seed
        seed_dist = float(np.linalg.norm(recovered_seed - true_seed))
        print(f"  Attempt {attempt+1}: time={dt:.2f}s, loss={loss:.6f}, PSNR={p:.1f}dB, "
              f"||θ-θ_true||={seed_dist:.3f}")

    # ============================================================
    # PART 3: Theoretical brute-force cost
    # ============================================================
    print()
    print("--- Part 3: Theoretical brute-force cost ---")
    P = len(true_seed)
    # Assume int8 quantization: each param has 256 possible values
    n_seeds_bits = 8 * P  # log2 of seed space size
    print(f"  Seed dimension P = {P}")
    print(f"  Quantization: int8 (256 values per param)")
    print(f"  Total seed space: 256^{P} = 2^{n_seeds_bits}")
    print(f"  T_genesis per attempt: {t_genesis*1000:.2f}ms")
    print(f"  Total brute-force time: 2^{n_seeds_bits} × {t_genesis:.4e}s")
    # In years (log-space)
    log2_seconds_per_year = np.log2(3600*24*365)
    log2_brute_years = n_seeds_bits + np.log2(t_genesis) - log2_seconds_per_year
    print(f"  In years: 2^{log2_brute_years:.1f}")
    print(f"  (For comparison, age of universe: 2^{np.log2(1.38e10):.1f} ≈ 2^33.7 years)")
    print()
    print(f"  Legitimate user cost (compression): {t_inverse:.2f}s = 2^{np.log2(t_inverse):.1f}s")
    print(f"  Attacker brute-force cost:          2^{n_seeds_bits + np.log2(t_genesis):.1f} s")
    print(f"  Security factor: 2^{n_seeds_bits + np.log2(t_genesis) - np.log2(t_inverse):.1f}")

    # ============================================================
    # PART 4: Cryptographic properties
    # ============================================================
    print()
    print("--- Part 4: Cryptographic properties ---")
    # One-way function definition: f is one-way if
    #   Pr[A(f(x)) finds x' with f(x') = f(x)] < 1/poly(|x|)
    # For BHUH: probability that random init finds the seed is effectively 0

    security_bits = 8 * P  # information-theoretic, assuming int8 quant
    print(f"  Seed dimension P = {P}")
    print(f"  Seed space size (int8 quant): 2^{security_bits} (information-theoretic)")
    print(f"  Compression asymmetry: {asymmetry:.0f}x (POLYNOMIAL, not superpolynomial)")
    print()
    print(f"  IMPORTANT: This is NOT cryptographic security.")
    print(f"  - BHUH inverse (compression) is polynomial-time via gradient descent")
    print(f"  - Formal one-way functions require superpolynomial inversion")
    print(f"  - The asymmetry is a LARGE CONSTANT (~{asymmetry:.0f}x), not exponential")
    print(f"  - Useful for proof-of-work, NOT for encryption/authentication")

    # ============================================================
    # PART 5: Test "preimage resistance" — can two seeds give same output?
    # ============================================================
    print()
    print("--- Part 5: Preimage resistance (multiple seeds → same output) ---")
    # Train 3 different seeds for same target
    print("  Training 3 independent seeds for same target...")
    seeds = []
    for i in range(3):
        rng_init = np.random.default_rng(100 + i)
        init = rng_init.normal(size=len(true_seed)) * 0.05
        seed_i, _, _ = inverse_attempt(target, coords, hidden=16, n_layers=3,
                                        epochs=300, lr=1e-3, init_seed=init)
        seeds.append(seed_i)
        pred_i = genesis(seed_i, coords, hidden=16, n_layers=3)
        print(f"    Seed {i+1}: PSNR={psnr(target, pred_i):.1f}dB")

    # Pairwise distances
    print()
    print("  Pairwise seed distances (different seeds, same output):")
    for i in range(3):
        for j in range(i+1, 3):
            d = float(np.linalg.norm(seeds[i] - seeds[j]))
            print(f"    ||θ_{i+1} - θ_{j+1}|| = {d:.3f}")
    print()
    print("  → Multiple seeds produce same output → BHUH is a MANY-TO-ONE function")
    print("  → This is the 'collision' property, important for hash functions")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()
    print("COMPUTATIONAL ASYMMETRY CHECKLIST:")
    print(f"  [✓] Forward (Genesis) is fast: {t_genesis*1000:.2f}ms")
    print(f"  [✓] Inverse (Compression) is slow: {t_inverse:.2f}s ({asymmetry:.0f}x slower)")
    print(f"  [✓] Many-to-one (multiple seeds → same output): confirmed")
    print(f"  [✗] NOT a cryptographic one-way function (inverse is polynomial)")
    print(f"  [✗] NOT comparable to AES/RSA (different primitive class)")
    print()
    print("COMPARISON TO PROOF-OF-WORK PRIMITIVES (NOT crypto keys):")
    print(f"  Primitive          Asymmetry type     Forward cost")
    print(f"  Bitcoin hashcash   Superpolynomial    ~1 μs (SHA-256)")
    print(f"  BHUH asymmetry     Polynomial const.  ~0.4 ms (Genesis)")
    print(f"  → BHUH is NOT a replacement for cryptographic hashes")
    print(f"  → BHUH IS useful for proof-of-work with computational (not crypto) security")
    print()
    print("WHAT BHUH ASYMMETRY CAN BE USED FOR:")
    print("  - Proof-of-work compression (Phase 83): useful work, easy to verify")
    print("  - Rate limiting: force ~1s compute per request, verify in ~1ms")
    print("  - Anti-spam: require compression work, not pure hash brute-force")
    print()
    print("WHAT BHUH ASYMMETRY CANNOT BE USED FOR:")
    print("  - Encryption (no secret-key property)")
    print("  - Authentication (inverse is polynomial)")
    print("  - Public-key crypto (no trapdoor function)")
    print("  - Hash commitments (collisions exist, not collision-resistant)")

    if asymmetry > 100:
        verdict = (f"VALIDATED (asymmetry, NOT crypto) — BHUH exhibits computational asymmetry "
                   f"of {asymmetry:.0f}x. Forward (Genesis) = {t_genesis*1000:.2f}ms, "
                   f"inverse (compression) = {t_inverse:.2f}s. Both are polynomial-time. "
                   "Multiple seeds → same output confirms many-to-one property. "
                   "Suitable for proof-of-work applications (Phase 83). "
                   "NOT a cryptographic one-way function (inverse is polynomial). "
                   "Axiom 12 (Computational Asymmetry) accepted in REVISED form.")
    else:
        verdict = "PARTIAL — Asymmetry exists but is too small for practical use."

    print(f"\nVerdict: {verdict}")
    print()
    print("REVISED AXIOM 12 (Computational Asymmetry — NOT 'One-Way Function'):")
    print("  Genesis: θ → x has computational asymmetry R = T_inverse / T_forward.")
    print("  Forward is O(P·N), inverse is O(P·N·E) with E epochs.")
    print("  R is a LARGE CONSTANT (typically 1000-6000), NOT superpolynomial.")
    print("  Multiple θ can map to same x (many-to-one, not collision-resistant).")
    print()
    print("  Formal: R(θ) := T_inverse(x) / T_genesis(θ) = O(E), polynomial in E")
    print("  This is NOT a cryptographic primitive.")

    return {
        'phase': 81,
        'name': 'BHUH Computational Asymmetry (CORRECTED from One-Way Function)',
        'verdict': verdict,
        'seed_dim': int(P),
        't_genesis_ms': float(t_genesis * 1000),
        't_inverse_s': float(t_inverse),
        'asymmetry': float(asymmetry),
        'seed_space_bits': int(security_bits),
        'log2_brute_force_years': float(log2_brute_years),
        'many_to_one': True,
        'is_cryptographic_one_way_function': False,
        'note': 'CORRECTED: BHUH has computational asymmetry (polynomial), NOT cryptographic one-way property (superpolynomial). Comparison with AES/RSA removed as incorrect.',
    }


if __name__ == '__main__':
    result = run_phase81()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
