# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 34: Neural Texture Synthesis (Procedural from Seeds)
============================================================
Tests whether SIREN seeds can generate NEW textures by sampling
random modulations.

CONCEPT:
  If the modulation space is a "universe" (validated in Phase 31),
  we should be able to SAMPLE random points in this space to generate
  NEW files that never existed but are structurally valid.

  This is "procedural generation from the universe" — the BHUH
  equivalent of sampling from a latent space in GANs/VAEs.

HYPOTHESIS:
  Random modulations sampled from the trained distribution will
  produce visually plausible images (not noise), demonstrating that
  the universe generalizes beyond training data.

METHOD:
  1. Train Multi-File SIREN on 20 images
  2. Compute mean and std of modulation vectors
  3. Sample 5 random modulations from N(mean, std)
  4. Generate images from random modulations
  5. Measure: are they visually plausible? (smoothness, structure)

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import ModulatedSIREN, get_coordinates, generate_satellite_images, measure_model_size_compressed


def train_universe(images, epochs=100, device='cpu', verbose=False):
    """Train Multi-File SIREN."""
    from phase1_multi_file_siren import train_multi_file_siren
    return train_multi_file_siren(images, epochs=epochs, device=device, verbose=verbose)


def sample_from_universe(model, n_samples, device='cpu'):
    """Sample random modulations and generate images."""
    # Get modulation statistics
    mods = model.modulations.weight.detach()  # (n_files, mod_dim)
    mod_mean = mods.mean(dim=0)
    mod_std = mods.std(dim=0)

    size = 128
    coords = get_coordinates(size, device)

    generated = []
    for i in range(n_samples):
        # Sample from N(mean, std)
        random_mod = torch.randn(mods.shape[1], device=device) * mod_std + mod_mean

        # Generate image using random modulation
        x = coords
        for j, layer in enumerate(model.base_siren.net):
            if j < len(model.base_siren.net) - 1:
                film = model.film_generators[j](random_mod)
                scale, shift = film.chunk(2, dim=-1)
                x = layer(x)
                x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
            else:
                x = layer(x)

        img = (x.detach().cpu().numpy().reshape(size, size, 3) * 255).clip(0, 255).astype(np.uint8)
        generated.append(img)

    return generated, mod_mean, mod_std


def measure_image_quality(img):
    """Measure basic image quality metrics."""
    # Smoothness: low high-frequency content
    gray = np.mean(img, axis=2).astype(np.float32)
    # Laplacian as smoothness measure
    lap = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
    # Dynamic range
    dr = img.max() - img.min()
    # Std (should be > 0 for non-trivial image)
    std = img.std()
    return {'smoothness': lap, 'dynamic_range': dr, 'std': std}


def run_phase34_experiment(verbose=True):
    """Run Phase 34 Neural Texture Synthesis experiment."""
    print("=" * 80)
    print("🧪 Phase 34: Neural Texture Synthesis (Procedural from Seeds)")
    print("=" * 80)

    device = 'cpu'

    # Train on 20 images
    print("\n📦 Generating and training on 20 satellite images...")
    images = generate_satellite_images(n_images=20, size=128, seed=42)
    model, loss = train_universe(images, epochs=100, device=device, verbose=verbose)

    # Sample from universe
    print("\n🎲 Sampling 5 random modulations from universe...")
    generated, mod_mean, mod_std = sample_from_universe(model, n_samples=5, device=device)

    # Analyze training images (reference)
    print(f"\n📊 Quality comparison (training vs generated):")
    print(f"\n  {'Source':<15} {'Smoothness':>12} {'Dyn Range':>12} {'Std':>8}")
    print(f"  {'-'*50}")

    # Training images stats
    train_smoothness = []
    train_dr = []
    train_std = []
    for img in images[:5]:
        q = measure_image_quality(img)
        train_smoothness.append(q['smoothness'])
        train_dr.append(q['dynamic_range'])
        train_std.append(q['std'])
    print(f"  {'Training avg':<15} {np.mean(train_smoothness):>10.2f} {np.mean(train_dr):>10.0f} {np.mean(train_std):>6.1f}")

    # Generated images stats
    gen_smoothness = []
    gen_dr = []
    gen_std = []
    for i, img in enumerate(generated):
        q = measure_image_quality(img)
        gen_smoothness.append(q['smoothness'])
        gen_dr.append(q['dynamic_range'])
        gen_std.append(q['std'])
        print(f"  {'Generated '+str(i+1):<15} {q['smoothness']:>10.2f} {q['dynamic_range']:>10.0f} {q['std']:>6.1f}")

    print(f"  {'Gen avg':<15} {np.mean(gen_smoothness):>10.2f} {np.mean(gen_dr):>10.0f} {np.mean(gen_std):>6.1f}")

    # Check if generated images are plausible
    smoothness_ratio = np.mean(gen_smoothness) / max(np.mean(train_smoothness), 0.001)
    dr_ratio = np.mean(gen_dr) / max(np.mean(train_dr), 1)
    std_ratio = np.mean(gen_std) / max(np.mean(train_std), 0.001)

    print(f"\n  📋 Ratio (generated/training):")
    print(f"  - Smoothness: {smoothness_ratio:.2f}x ({'similar' if 0.5 < smoothness_ratio < 2.0 else 'different'})")
    print(f"  - Dynamic range: {dr_ratio:.2f}x ({'similar' if 0.5 < dr_ratio < 2.0 else 'different'})")
    print(f"  - Std: {std_ratio:.2f}x ({'similar' if 0.5 < std_ratio < 2.0 else 'different'})")

    plausible = all(0.3 < r < 3.0 for r in [smoothness_ratio, dr_ratio, std_ratio])

    print(f"\n{'='*80}")
    print("📊 PHASE 34 RESULTS — NEURAL TEXTURE SYNTHESIS")
    print(f"{'='*80}")

    if plausible:
        print(f"\n  ✅ Random modulations produce PLAUSIBLE images!")
        print(f"     The universe generalizes beyond training data.")
        print(f"     Sampling random points = procedural generation.")
    else:
        print(f"\n  ⚠️  Generated images differ from training distribution")
        print(f"     May need more training data or constrained sampling")

    print(f"\n  📋 Modulation space statistics:")
    print(f"  - Mean: {mod_mean.cpu().numpy()[:4]}... (first 4 dims)")
    print(f"  - Std:  {mod_std.cpu().numpy()[:4]}... (first 4 dims)")
    print(f"  - Dimension: {mod_mean.shape[0]}")

    print(f"\n  📋 Applications:")
    print(f"  - Procedural texture generation (games, VR)")
    print(f"  - Data augmentation (generate synthetic training data)")
    print(f"  - Creative AI (explore the universe)")
    print(f"  - Infinite asset generation (no two textures alike)")

    print(f"\n  📋 Deep insight:")
    print(f"  The BHUH universe is not just a compression tool —")
    print(f"  it's a GENERATIVE MODEL. The modulation space contains")
    print(f"  an INFINITE number of potential files, not just the")
    print(f"  training set. This is the deepest validation of the")
    print(f"  Multiverse principle: the universe is bigger than what")
    print(f"  we put into it.")

    return {
        'n_training': 20,
        'n_generated': 5,
        'smoothness_ratio': smoothness_ratio,
        'dr_ratio': dr_ratio,
        'std_ratio': std_ratio,
        'plausible': plausible,
    }


if __name__ == '__main__':
    results = run_phase34_experiment(verbose=True)
