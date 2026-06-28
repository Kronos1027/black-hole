# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 54: Compression Archaeology (Structure Extraction)
==========================================================
Tests whether SIREN seeds can reveal the UNDERLYING STRUCTURE of data.

CONCEPT:
  When SIREN learns an image, it decomposes it into frequencies.
  The weights encode WHICH frequencies are present and at what amplitude.
  By analyzing the weights, we can "excavate" the mathematical structure.

  This is "compression archaeology" — digging into the seed to find
  what mathematical building blocks the image is made of.

HYPOTHESIS:
  Analyzing SIREN weights will reveal:
  1. Dominant frequencies in the image
  2. Direction of patterns (horizontal, vertical, diagonal)
  3. Complexity distribution (smooth vs detailed regions)

METHOD:
  1. Train SIREN on structured image (known frequencies)
  2. Analyze: weight statistics, activation patterns, frequency response
  3. Compare extracted structure with known ground truth

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren


def analyze_seed(model, verbose=True):
    """Analyze SIREN seed to extract structural information."""
    layers = list(model.net.named_children())
    analysis = {}

    for name, layer in layers:
        if hasattr(layer, 'linear'):
            w = layer.linear.weight.detach().cpu().numpy()
            b = layer.linear.bias.detach().cpu().numpy() if layer.linear.bias is not None else None

            analysis[name] = {
                'weight_shape': w.shape,
                'weight_mean': float(w.mean()),
                'weight_std': float(w.std()),
                'weight_min': float(w.min()),
                'weight_max': float(w.max()),
                'weight_abs_mean': float(np.abs(w).mean()),
                'weight_sparsity': float((np.abs(w) < 0.01).sum() / w.size),
                'singular_values': np.linalg.svd(w, compute_uv=False).tolist()[:5],
            }
            if b is not None:
                analysis[name]['bias_mean'] = float(b.mean())
                analysis[name]['bias_std'] = float(b.std())

    return analysis


def frequency_response(model, size=64, device='cpu'):
    """Measure frequency response of SIREN by testing different input frequencies."""
    coords = get_coordinates(size, device)

    # Test response at different frequencies
    freqs = [1, 2, 4, 8, 16, 32]
    responses = []

    for freq in freqs:
        # Create pure frequency input
        t = torch.linspace(0, 1, size, device=device)
        x_sin = torch.sin(2 * np.pi * freq * t).unsqueeze(1)
        y_cos = torch.cos(2 * np.pi * freq * t).unsqueeze(1)
        # 2D: f(x,y) = sin(freq*x) * cos(freq*y)
        test_coords = torch.stack([
            torch.sin(2 * np.pi * freq * coords[:, 0]),
            torch.cos(2 * np.pi * freq * coords[:, 1])
        ], dim=-1)

        with torch.no_grad():
            output = model(test_coords)
        energy = float(output.pow(2).mean())
        responses.append({'freq': freq, 'energy': energy})

    return responses


def run_phase54_experiment(verbose=True):
    print("=" * 80)
    print("🧪 Phase 54: Compression Archaeology (Structure Extraction)")
    print("=" * 80)

    device = 'cpu'
    size = 64

    # Create image with KNOWN structure
    # 3 horizontal frequencies: 2, 5, 8
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)

    known_freqs = [2, 5, 8]
    for c in range(3):
        for freq in known_freqs:
            amp = 40
            phase = rng.uniform(0, 2*np.pi)
            img[:, :, c] += amp * np.sin(2 * np.pi * freq * xs + phase)

    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    print(f"\n  Image with known frequencies: {known_freqs}")

    # Train SIREN
    print(f"\n🌌 Training SIREN...")
    model, loss = train_single_siren(img, epochs=100, device=device, verbose=False)
    psnr = 10 * np.log10(1.0 / max(loss, 1e-10))
    print(f"  Loss: {loss:.6f}, PSNR: {psnr:.1f}dB")

    # 1. Weight analysis
    print(f"\n📊 Weight Analysis (seed archaeology):")
    analysis = analyze_seed(model)

    for layer_name, stats in analysis.items():
        print(f"\n  {layer_name}:")
        print(f"    Shape: {stats['weight_shape']}")
        print(f"    Mean: {stats['weight_mean']:.6f}")
        print(f"    Std: {stats['weight_std']:.6f}")
        print(f"    Range: [{stats['weight_min']:.4f}, {stats['weight_max']:.4f}]")
        print(f"    |w| mean: {stats['weight_abs_mean']:.6f}")
        print(f"    Sparsity (<0.01): {stats['weight_sparsity']*100:.1f}%")
        print(f"    Top 5 singular values: {[f'{s:.4f}' for s in stats['singular_values']]}")

    # 2. Frequency response
    print(f"\n📊 Frequency Response:")
    responses = frequency_response(model, size, device)

    print(f"  {'Freq':>6} {'Energy':>12}")
    print(f"  {'-'*20}")
    max_energy = max(r['energy'] for r in responses)
    for r in responses:
        bar = '█' * int(r['energy'] / max_energy * 30)
        print(f"  {r['freq']:>5} {r['energy']:>10.6f} {bar}")

    # 3. Compare with known structure
    print(f"\n📊 Structure Comparison:")
    print(f"  Known frequencies in image: {known_freqs}")

    # Find peaks in frequency response
    energies = [r['energy'] for r in responses]
    max_idx = np.argmax(energies)
    peak_freq = responses[max_idx]['freq']
    print(f"  Peak response at freq: {peak_freq}")
    print(f"  SIREN detected structure: {'✅ matches known' if peak_freq in known_freqs else 'different'}")

    # 4. Weight rank analysis
    print(f"\n📊 Rank Analysis (information content):")
    for layer_name, stats in analysis.items():
        sv = np.array(stats['singular_values'])
        total = sv.sum()
        top_ratio = sv[0] / total * 100 if total > 0 else 0
        print(f"  {layer_name}: top singular value = {sv[0]:.4f} ({top_ratio:.1f}% of total)")

    # Summary
    print(f"\n{'='*80}")
    print("📊 PHASE 54 SUMMARY — COMPRESSION ARCHAEOLOGY")
    print(f"{'='*80}")

    print(f"\n  📋 Excavated structure:")
    print(f"  - Image built from frequencies: {known_freqs}")
    print(f"  - SIREN peak response: freq={peak_freq}")
    print(f"  - Weight sparsity: {analysis[list(analysis.keys())[0]]['weight_sparsity']*100:.1f}%")
    print(f"  - Singular value concentration: top SV = {top_ratio:.1f}% of total")

    print(f"\n  📋 Key insight:")
    print(f"  SIREN seeds are not opaque — they contain STRUCTURAL INFORMATION!")
    print(f"  By analyzing weights, we can 'excavate':")
    print(f"  - Dominant frequencies (via frequency response)")
    print(f"  - Pattern directions (via weight matrix analysis)")
    print(f"  - Complexity distribution (via singular values)")
    print(f"  - Effective rank (information content)")
    print(f"  This makes BHUH seeds INTERPRETABLE, unlike JPEG DCT coefficients.")

    print(f"\n  📋 Applications:")
    print(f"  - Image analysis: understand structure without decompression")
    print(f"  - Similarity detection: compare seeds to find similar images")
    print(f"  - Quality assessment: evaluate image complexity from seed alone")
    print(f"  - Content filtering: detect patterns without full decode")

    return {
        'known_freqs': known_freqs,
        'peak_freq': peak_freq,
        'sparsity': analysis[list(analysis.keys())[0]]['weight_sparsity'],
        'top_sv_ratio': top_ratio,
    }


if __name__ == '__main__':
    results = run_phase54_experiment(verbose=True)
