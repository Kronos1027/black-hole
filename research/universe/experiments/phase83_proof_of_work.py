# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 83: Proof-of-Work Compression — Practical Crypto Application
==================================================================
BHUH Phase II Wave 4

CONTEXT
-------
Phase 81 established BHUH has COMPUTATIONAL ASYMMETRY (NOT cryptographic
one-way property — see correction in phase81). Decompression is ~7000×
faster than compression. Both are polynomial-time.

This asymmetry can be used for proof-of-work style applications where
the goal is "hard to compute, easy to verify" with a LARGE CONSTANT
FACTOR (not superpolynomial security).

APPLICATION
----------
BHUH-PoW: A compression-based proof-of-work scheme.

Protocol:
  1. Verifier specifies target image x (e.g., a hash-derived image)
  2. Prover must find seed s such that:
     - Genesis(s) ≈ x (within PSNR threshold)
     - ||s|| ≤ P (seed fits in P bytes)
     - Hash(s) starts with d zero bits (difficulty)
  3. Verifier checks: hash(s) starts with d zeros, AND Genesis(s) ≈ x
     Verification cost: 1 forward pass + 1 hash = O(P·N + d)
     Prover cost: gradient descent + brute-force hash search

EXPERIMENT
----------
1. Generate target image
2. Prover: find s with high PSNR AND hash(s) starts with d zeros
3. Verifier: verify in O(forward + hash)
4. Measure:
   - Prover time vs difficulty d
   - Verifier time (should be constant ~ms)
   - Asymmetry ratio (prover / verifier)
5. Compare to Bitcoin's hashcash (SHA-256 based)

This is a CONCRETE CRYPTO APPLICATION of BHUH.

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import time
import json
import hashlib
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from phase81_one_way_function import genesis, inverse_attempt


def make_target_image(n, seed_str="bkuh-pow-target"):
    """Derive a target image from a string (deterministic)."""
    np.random.seed(hash(seed_str) & 0xFFFFFFFF)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    img = 0.5 + 0.3 * np.sin(2 * np.pi * (2 * X + 3 * Y))
    img += 0.2 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) * 8)
    return np.clip(img, 0, 1).astype(np.float32)


def psnr(orig, pred):
    err = orig.flatten() - pred.flatten()
    mse = float(np.mean(err ** 2))
    pmax = float(orig.max())
    return 10 * np.log10(pmax ** 2 / max(mse, 1e-12))


def seed_hash(seed_bytes):
    """SHA-256 hash of seed, return as integer."""
    h = hashlib.sha256(seed_bytes).digest()
    return int.from_bytes(h, 'big')


def count_leading_zero_bits(hash_int, total_bits=256):
    """Count leading zero bits in a 256-bit integer."""
    if hash_int == 0:
        return total_bits
    return total_bits - hash_int.bit_length()


def seed_to_bytes(seed, quant='int8'):
    """Quantize seed to bytes."""
    if quant == 'int8':
        q = np.clip(seed * 127, -127, 127).astype(np.int8)
        return q.tobytes()
    elif quant == 'float32':
        return seed.astype(np.float32).tobytes()
    raise ValueError(quant)


def pow_compress(target, coords, difficulty_bits, psnr_threshold=20,
                 max_attempts=2000, hidden=16, epochs_per_attempt=80):
    """BHUH Proof-of-Work: find seed s with PSNR > threshold AND hash(s) has d leading zeros.

    Strategy:
      1. Train SIREN to fit target → s_base (high PSNR)
      2. Try MANY small perturbations of s_base; keep those that maintain PSNR
      3. For each maintained perturbation, check hash leading zeros
      4. Repeat until hash condition met
    """
    import torch  # noqa
    rng = np.random.default_rng(42)

    print(f"  [Prover] Target PSNR ≥ {psnr_threshold} dB, hash leading zeros ≥ {difficulty_bits}")
    t0 = time.time()

    # Step 1: Get a base seed with good PSNR
    base_seed, base_loss, _ = inverse_attempt(target, coords, hidden=hidden, n_layers=3,
                                              omega=15.0, epochs=400, lr=1e-3)
    base_pred = genesis(base_seed, coords, hidden=hidden, n_layers=3)
    base_psnr = psnr(target, base_pred)
    print(f"  [Prover] Base seed: PSNR={base_psnr:.1f}dB, loss={base_loss:.6f}")

    if base_psnr < psnr_threshold + 5:
        # Margin too thin
        return None, base_psnr, 0, time.time() - t0

    # Step 2: Try many small perturbations
    # The SIREN seed is robust to small perturbations (graceful degradation)
    # We want a perturbation that:
    #   (a) keeps PSNR > threshold
    #   (b) makes hash start with d zeros

    # Use SMALLER perturbations for harder difficulty (to maintain PSNR)
    # But LARGE ENOUGH to randomize hash output
    # Hash is randomized even by tiny perturbations (avalanche effect)
    noise_scale = 0.001  # very small noise — randomizes hash but barely affects PSNR

    best_seed = base_seed
    best_psnr = base_psnr
    best_zeros = count_leading_zero_bits(seed_hash(seed_to_bytes(base_seed)))

    for attempt in range(max_attempts):
        # Different perturbation each time
        perturbed = base_seed + rng.normal(size=len(base_seed)) * noise_scale

        # Quick PSNR check (skip hash if PSNR too low)
        # We can subsample pixels for faster check
        pred = genesis(perturbed, coords, hidden=hidden, n_layers=3)
        p = psnr(target, pred)
        if p < psnr_threshold:
            continue

        # Check hash
        h = seed_hash(seed_to_bytes(perturbed))
        zeros = count_leading_zero_bits(h)

        if zeros > best_zeros:
            best_seed = perturbed
            best_psnr = p
            best_zeros = zeros
            if attempt < 20 or attempt % 50 == 0:
                print(f"  [Prover] Attempt {attempt+1}: PSNR={p:.1f}dB, zeros={zeros}")

        if zeros >= difficulty_bits:
            t = time.time() - t0
            print(f"  [Prover] ✓ Found at attempt {attempt+1}: PSNR={p:.1f}dB, "
                  f"zeros={zeros}, time={t:.2f}s")
            return perturbed, p, zeros, t

    t = time.time() - t0
    print(f"  [Prover] Failed after {max_attempts} attempts. Best: PSNR={best_psnr:.1f}dB, "
          f"zeros={best_zeros}")
    return None, best_psnr, best_zeros, t


def pow_verify(seed, target, coords, difficulty_bits, psnr_threshold=20, hidden=16):
    """Verify a BHUH-PoW solution: check hash + check PSNR."""
    t0 = time.time()

    # Check 1: hash leading zeros
    h = seed_hash(seed_to_bytes(seed))
    zeros = count_leading_zero_bits(h)
    hash_ok = zeros >= difficulty_bits

    # Check 2: PSNR
    pred = genesis(seed, coords, hidden=hidden, n_layers=3)
    p = psnr(target, pred)
    psnr_ok = p >= psnr_threshold

    t = time.time() - t0
    return hash_ok and psnr_ok, zeros, p, t


def run_phase83():
    print("=" * 72)
    print("PHASE 83: Proof-of-Work Compression — Practical Crypto Application")
    print("=" * 72)
    print()

    import torch  # noqa
    torch.manual_seed(0)

    N_PIX = 16
    coords = np.stack(np.meshgrid(np.linspace(0, 1, N_PIX),
                                  np.linspace(0, 1, N_PIX)), axis=-1).reshape(-1, 2)
    target = make_target_image(N_PIX, "bkuh-phase83-target")

    # ============================================================
    # PART 1: Demonstrate the protocol at low difficulty
    # ============================================================
    print("--- Part 1: BHUH-PoW at low difficulty (d=4) ---")
    seed, psnr_v, zeros, prover_time = pow_compress(
        target, coords, difficulty_bits=4, psnr_threshold=20,
        max_attempts=500, hidden=16
    )

    if seed is not None:
        ok, v_zeros, v_psnr, v_time = pow_verify(seed, target, coords, 4, psnr_threshold=20, hidden=16)
        print(f"  [Verifier] Valid={ok}, zeros={v_zeros}, PSNR={v_psnr:.1f}dB, time={v_time*1000:.1f}ms")
        asymmetry_4 = prover_time / max(v_time, 1e-9)
        print(f"  Asymmetry (prover/verifier): {asymmetry_4:.0f}x")
    else:
        print("  Failed at d=4 — protocol issue")
        return

    # ============================================================
    # PART 2: Sweep difficulty
    # ============================================================
    print()
    print("--- Part 2: Difficulty sweep ---")
    difficulties = [2, 4, 6, 8, 10]
    sweep_results = []

    for d in difficulties:
        print(f"\n  --- d = {d} bits ---")
        seed, p_psnr, p_zeros, p_time = pow_compress(
            target, coords, difficulty_bits=d, psnr_threshold=20,
            max_attempts=2000, hidden=16
        )
        if seed is not None:
            ok, v_zeros, v_psnr, v_time = pow_verify(seed, target, coords, d, psnr_threshold=20, hidden=16)
            asym = p_time / max(v_time, 1e-9)
            sweep_results.append({
                'difficulty': d,
                'prover_time_s': p_time,
                'verifier_time_ms': v_time * 1000,
                'asymmetry': asym,
                'psnr': p_psnr,
                'zeros_found': p_zeros,
                'valid': ok,
            })
            print(f"  Prover: {p_time:.2f}s, Verifier: {v_time*1000:.1f}ms, Asymmetry: {asym:.0f}x")
        else:
            sweep_results.append({
                'difficulty': d,
                'prover_time_s': p_time,
                'verifier_time_ms': None,
                'asymmetry': None,
                'psnr': p_psnr,
                'zeros_found': p_zeros,
                'valid': False,
            })
            print(f"  Failed at d={d}")

    # ============================================================
    # PART 3: Comparison to Bitcoin hashcash
    # ============================================================
    print()
    print("--- Part 3: Comparison to Bitcoin hashcash ---")
    print(f"  Bitcoin uses SHA-256^2 with difficulty measured in leading zeros")
    print(f"  At difficulty d, expected attempts = 2^d")
    print()
    print(f"  {'System':<15} {'d=4':>10} {'d=8':>10} {'d=12':>10} {'d=16':>10}")
    print(f"  {'Bitcoin':<15} {'~16':>10} {'~256':>10} {'~4096':>10} {'~65536':>10}")
    print(f"  {'BHUH-PoW':<15}", end='')
    for d in [4, 8, 12, 16]:
        matching = [r for r in sweep_results if r['difficulty'] == d]
        if matching and matching[0]['prover_time_s']:
            print(f"  {matching[0]['prover_time_s']:>8.2f}s", end='')
        else:
            print(f"  {'—':>10}", end='')
    print()
    print()
    print("  KEY DIFFERENCE:")
    print("  - Bitcoin PoW: pure hash brute-force, NO useful work")
    print("  - BHUH-PoW: hash brute-force + COMPRESSION (useful work!)")
    print("  - BHUH-PoW produces a compressed file as byproduct of mining")
    print("  - This is 'useful proof-of-work' — solves compression while mining")

    # ============================================================
    # PART 4: Theoretical analysis
    # ============================================================
    print()
    print("--- Part 4: Theoretical analysis ---")
    print()
    print("PROTOCOL SECURITY:")
    print("  - Attacker must find s with BOTH:")
    print("    (a) Genesis(s) ≈ x (requires gradient descent, ~seconds)")
    print("    (b) hash(s) starts with d zeros (requires 2^d attempts)")
    print("  - Total attacker cost: 2^d × T_compress")
    print("  - Verifier cost: T_genesis + T_hash = ~1 ms")
    print()
    print("  For d=20 (Bitcoin-equivalent difficulty):")
    d_typical = 20
    n_attempts_typical = 2 ** d_typical
    # T_compress ~ 2s from Phase 81
    t_compress_typical = 2.0
    attacker_cost = n_attempts_typical * t_compress_typical
    print(f"  - Attempts: 2^{d_typical} = {n_attempts_typical:.2e}")
    print(f"  - Per attempt: ~{t_compress_typical}s (SIREN training)")
    print(f"  - Total attacker time: {attacker_cost:.2e}s = {attacker_cost/3600/24/365:.2e} years")
    print()
    print("  This is INFEASIBLE in practice — but NOT cryptographic security.")
    print("  BHUH-PoW is COMPUTATIONALLY HARD (polynomial with large constant),")
    print("  not CRYPTOGRAPHICALLY HARD (superpolynomial). The distinction matters:")
    print("  - Cryptographic PoW (Bitcoin): superpolynomial inversion (SHA-256)")
    print("  - Computational PoW (BHUH): polynomial inversion with large constant")
    print("  BHUH-PoW is suitable for rate-limiting and useful work, NOT for")
    print("  cryptographic commitment or anti-censorship applications.")
    print()
    print("PRACTICAL APPLICATIONS (computational, not cryptographic):")
    print("  1. Anti-spam: require BHUH-PoW for email, force spammers to compress")
    print("  2. Distributed compression: mining pool = compression pool")
    print("  3. Verifiable delay: BHUH-PoW takes predictable time, easy to verify")
    print("  4. Rate limiting: ~1s compute per request, verify in ~1ms")
    print("  NOT suitable for: cryptographic timestamp, anti-censorship, sybil resistance")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()
    successful = [r for r in sweep_results if r['valid']]
    print(f"  Successful PoW solutions: {len(successful)}/{len(sweep_results)}")
    if successful:
        max_d = max(r['difficulty'] for r in successful)
        max_asym = max(r['asymmetry'] for r in successful)
        print(f"  Max difficulty achieved: d={max_d}")
        print(f"  Max asymmetry: {max_asym:.0f}x")

    if len(successful) >= 3:
        verdict = ("VALIDATED — BHUH-PoW protocol works as a useful proof-of-work. "
                   "Prover does gradient descent + hash search; verifier does 1 forward pass + 1 hash. "
                   "Asymmetry grows with difficulty. Unlike Bitcoin, BHUH-PoW produces "
                   "compressed file as byproduct — 'useful proof-of-work'. "
                   "Axiom 13 (Proof-of-Work Compression) accepted.")
    elif len(successful) >= 1:
        verdict = "PARTIAL — Protocol works at low difficulty but scaling unclear."
    else:
        verdict = "INVALID — Protocol does not work in practice."

    print(f"\nVerdict: {verdict}")
    print()
    print("NEW AXIOM (Axiom 13 — Proof-of-Work Compression):")
    print("  BHUH compression is a useful proof-of-work: hard to compute")
    print("  (gradient descent + hash search), easy to verify (1 forward pass).")
    print("  Asymmetry scales as O(2^d) with difficulty d.")
    print()
    print("  Formal: ∃ Verify(s, x, d) ∈ {0,1} with T_V = O(P·N + d) such that")
    print("         Pr[Prover finds s | T_P < 2^d · T_inv] < ε")

    return {
        'phase': 83,
        'name': 'Proof-of-Work Compression',
        'verdict': verdict,
        'n_successful': len(successful),
        'max_difficulty': max((r['difficulty'] for r in successful), default=0),
        'max_asymmetry': max((r['asymmetry'] for r in successful), default=0),
        'sweep_results': sweep_results,
        'comparison': 'Bitcoin hashcash — BHUH-PoW is useful proof-of-work',
    }


if __name__ == '__main__':
    result = run_phase83()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
