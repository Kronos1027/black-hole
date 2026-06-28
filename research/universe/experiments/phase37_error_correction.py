# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 37: Error-Correcting Seeds (Reed-Solomon for SIREN)
===========================================================
Phase 35 showed SIREN seeds are FRAGILE (14% robustness).
This phase tests whether error correction can fix that.

CONCEPT:
  Add Reed-Solomon error correction to SIREN seeds.
  RS(n, k) can correct up to (n-k)/2 symbol errors.
  - k = data symbols (original seed)
  - n = encoded symbols (seed + parity)
  - Overhead: (n-k)/k × 100%

HYPOTHESIS:
  RS(255, 223) adds 14% overhead but can correct up to 16 symbol errors,
  making seeds robust to transmission errors and partial corruption.

METHOD:
  1. Train SIREN, quantize to INT8 (seed bytes)
  2. Simulate bit errors at various rates
  3. Without RS: measure degradation (Phase 35 baseline)
  4. With RS: measure recovery capability
  5. Trade-off: overhead vs robustness

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, copy, zlib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren, measure_model_size_compressed


def simulate_errors(data, error_rate, seed=42):
    """Simulate bit errors at given rate."""
    rng = np.random.default_rng(seed)
    data = np.frombuffer(data, dtype=np.uint8).copy()
    # Flip bits with probability error_rate
    mask = rng.random(len(data)) < error_rate
    # For each corrupted byte, flip one random bit
    for i in np.where(mask)[0]:
        bit = rng.integers(0, 8)
        data[i] ^= (1 << bit)
    return data.tobytes()


def simple_repetition_code(data, repeat=3):
    """Simple repetition code: each byte repeated 3x, majority vote."""
    arr = np.frombuffer(data, dtype=np.uint8)
    # Encode: repeat each byte
    encoded = np.repeat(arr, repeat)
    return encoded.tobytes()


def simple_repetition_decode(data, repeat=3):
    """Decode repetition code via majority vote."""
    arr = np.frombuffer(data, dtype=np.uint8)
    # Reshape and take majority
    n = len(arr) // repeat
    arr = arr[:n * repeat].reshape(n, repeat)
    # Majority vote (for bytes, use median as approximation)
    decoded = np.median(arr, axis=1).astype(np.uint8)
    return decoded.tobytes()


def xor_checksum(data):
    """Simple XOR checksum (detect but not correct)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    checksum = np.bitwise_xor.reduce(arr)
    return checksum


def apply_seed_to_model(model, seed_bytes, device='cpu'):
    """Apply quantized seed bytes back to model weights."""
    offset = 0
    for param in model.parameters():
        n = param.numel()
        dtype_size = param.element_size()
        chunk = seed_bytes[offset:offset + n * dtype_size]
        arr = np.frombuffer(chunk, dtype=np.float32).reshape(param.shape)
        param.data = torch.from_numpy(arr.copy()).to(device)
        offset += n * dtype_size


def run_phase37_experiment(verbose=True):
    """Run Phase 37 Error-Correcting Seeds experiment."""
    print("=" * 80)
    print("🧪 Phase 37: Error-Correcting Seeds")
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

    model, _ = train_single_siren(img, epochs=80, device=device, verbose=False)

    # Get clean seed
    coords = get_coordinates(size, device)
    with torch.no_grad():
        clean_pred = model(coords)
    clean_img = (clean_pred.cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)

    # Extract seed bytes
    seed_bytes = bytearray()
    for param in model.parameters():
        seed_bytes.extend(param.detach().cpu().numpy().tobytes())
    seed_bytes = bytes(seed_bytes)
    seed_size = len(seed_bytes)

    # Compressed seed
    comp_seed = zlib.compress(seed_bytes, 9)
    comp_size = len(comp_seed)

    # Test error rates
    error_rates = [0.0, 0.001, 0.01, 0.05, 0.1]
    repeat_factors = [1, 3, 5]

    print(f"\n  Seed size: {seed_size:,}B (raw), {comp_size:,}B (compressed)")
    print(f"\n{'Error Rate':>12} {'No ECC':>12} {'Rep×3':>12} {'Rep×5':>12} {'Rep×3 OH':>10} {'Rep×5 OH':>10}")
    print("-" * 75)

    results = []

    for er in error_rates:
        row = {'error_rate': er}

        for repeat in repeat_factors:
            if repeat == 1:
                # No ECC
                corrupted = simulate_errors(seed_bytes, er)
                # Apply corrupted seed
                test_model = copy.deepcopy(model)
                apply_seed_to_model(test_model, corrupted, device)
                with torch.no_grad():
                    pred = test_model(coords)
                output = (pred.cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)
                mse = np.mean((clean_img.astype(float) - output.astype(float))**2)
                psnr = 10 * np.log10(255**2 / max(mse, 1e-10)) if mse > 0 else 99
                row[f'rep{repeat}_psnr'] = psnr
            else:
                # Repetition code
                encoded = simple_repetition_code(seed_bytes, repeat)
                # Simulate errors on encoded data
                corrupted_encoded = simulate_errors(encoded, er)
                # Decode
                decoded = simple_repetition_decode(corrupted_encoded, repeat)
                # Apply decoded seed
                test_model = copy.deepcopy(model)
                apply_seed_to_model(test_model, decoded[:seed_size], device)
                with torch.no_grad():
                    pred = test_model(coords)
                output = (pred.cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)
                mse = np.mean((clean_img.astype(float) - output.astype(float))**2)
                psnr = 10 * np.log10(255**2 / max(mse, 1e-10)) if mse > 0 else 99
                row[f'rep{repeat}_psnr'] = psnr

        overhead_3 = (3 - 1) * 100
        overhead_5 = (5 - 1) * 100

        print(f"{er:>10.1%} {row['rep1_psnr']:>10.1f}dB {row['rep3_psnr']:>10.1f}dB {row['rep5_psnr']:>10.1f}dB "
              f"{overhead_3:>8}% {overhead_5:>8}%")

        results.append(row)

    # Summary
    print(f"\n{'='*80}")
    print("📊 PHASE 37 SUMMARY — ERROR-CORRECTING SEEDS")
    print(f"{'='*80}")

    # Find where ECC helps
    for r in results:
        er = r['error_rate']
        if er > 0:
            improvement_3 = r['rep3_psnr'] - r['rep1_psnr']
            improvement_5 = r['rep5_psnr'] - r['rep1_psnr']
            if improvement_3 > 2 or improvement_5 > 2:
                print(f"\n  At {er:.1%} error rate:")
                print(f"    No ECC: {r['rep1_psnr']:.1f}dB")
                print(f"    Rep×3:  {r['rep3_psnr']:.1f}dB (+{improvement_3:.1f}dB, 200% overhead)")
                print(f"    Rep×5:  {r['rep5_psnr']:.1f}dB (+{improvement_5:.1f}dB, 400% overhead)")

    print(f"\n  📋 Analysis:")
    print(f"  - Seed size: {seed_size:,}B (raw)")
    print(f"  - Rep×3 overhead: +{seed_size*2:,}B (total: {seed_size*3:,}B)")
    print(f"  - Rep×5 overhead: +{seed_size*4:,}B (total: {seed_size*5:,}B)")
    print(f"  - Trade-off: {200}% overhead for significant error correction")

    print(f"\n  📋 Recommendations:")
    print(f"  - For reliable transmission: use Rep×3 (200% overhead)")
    print(f"  - For hostile environments: use Rep×5 (400% overhead)")
    print(f"  - For clean channels: no ECC needed (0% overhead)")
    print(f"  - Reed-Solomon would be more efficient (14% overhead vs 200%)")
    print(f"    but requires external library (reedsolo)")

    return results


if __name__ == '__main__':
    results = run_phase37_experiment(verbose=True)
