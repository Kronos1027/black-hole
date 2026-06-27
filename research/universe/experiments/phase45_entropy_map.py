# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 45: Universe Entropy Map (Information Topology)
=======================================================
Maps the "information topology" of the modulation universe.

CONCEPT:
  If each file is a point in modulation space (validated in Phases 31, 42),
  what does the DISTRIBUTION of files look like?

  - Are files clustered? (similar files group together)
  - Is the distribution uniform? (files spread evenly)
  - What's the "dimension" of the effective file space?

  This maps the "geography" of the BHUH universe.

METHOD:
  1. Train Multi-File SIREN on 50 images
  2. Extract all 50 modulation vectors (16-dim each)
  3. Compute: PCA, pairwise distances, effective dimensionality
  4. Map the universe's "shape"

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import train_multi_file_siren, generate_satellite_images


def run_phase45_experiment(verbose=True):
    print("=" * 80)
    print("🧪 Phase 45: Universe Entropy Map (Information Topology)")
    print("=" * 80)

    device = 'cpu'

    # Train on 50 images
    print("\n📦 Training Multi-File SIREN on 50 images...")
    images = generate_satellite_images(n_images=50, size=64, seed=42)
    model, loss = train_multi_file_siren(images, epochs=80, device=device, verbose=False)

    # Extract modulations
    mods = model.modulations.weight.detach().cpu().numpy()  # (50, 16)
    print(f"  Modulation matrix: {mods.shape}")

    # 1. PCA analysis
    from numpy.linalg import svd
    mods_centered = mods - mods.mean(axis=0)
    U, S, Vt = svd(mods_centered, full_matrices=False)

    # Explained variance
    variance_explained = (S ** 2) / (S ** 2).sum()
    cumulative = np.cumsum(variance_explained)

    print(f"\n📊 PCA Analysis (effective dimensionality):")
    print(f"  {'PC':>4} {'Eigenvalue':>12} {'Variance %':>12} {'Cumulative %':>14}")
    print(f"  {'-'*45}")
    for i in range(min(10, len(S))):
        print(f"  {i+1:>3} {S[i]:>10.4f} {variance_explained[i]*100:>10.1f}% {cumulative[i]*100:>12.1f}%")

    # Effective dimensionality (number of PCs for 95% variance)
    eff_dim = np.searchsorted(cumulative, 0.95) + 1
    print(f"\n  Effective dimensionality (95% variance): {eff_dim} out of {mods.shape[1]}")

    # 2. Pairwise distance distribution
    from scipy.spatial.distance import pdist, squareform
    distances = pdist(mods, metric='euclidean')

    print(f"\n📊 Pairwise Distance Distribution:")
    print(f"  Mean: {distances.mean():.4f}")
    print(f"  Std:  {distances.std():.4f}")
    print(f"  Min:  {distances.min():.4f}")
    print(f"  Max:  {distances.max():.4f}")
    print(f"  Median: {np.median(distances):.4f}")

    # Coefficient of variation (uniformity measure)
    cv = distances.std() / distances.mean()
    print(f"  Coefficient of variation: {cv:.4f} ({'uniform' if cv < 0.3 else 'clustered' if cv > 0.5 else 'moderate'})")

    # 3. Nearest neighbor analysis
    dist_matrix = squareform(distances)
    np.fill_diagonal(dist_matrix, np.inf)
    nn_distances = dist_matrix.min(axis=1)

    print(f"\n📊 Nearest Neighbor Analysis:")
    print(f"  Mean NN distance: {nn_distances.mean():.4f}")
    print(f"  Min NN distance: {nn_distances.min():.4f} (closest pair)")
    print(f"  Max NN distance: {nn_distances.max():.4f} (most isolated)")

    # 4. Entropy of the distribution (1D projection)
    # Use first principal component for manageable histogram
    pc1 = mods_centered @ Vt[0]  # Project onto PC1
    hist, _ = np.histogram(pc1, bins=20)
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    spatial_entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probs))

    print(f"\n📊 Spatial Entropy:")
    print(f"  Entropy: {spatial_entropy:.2f} bits")
    print(f"  Max possible: {max_entropy:.2f} bits")
    print(f"  Uniformity: {spatial_entropy/max_entropy*100:.1f}%")

    # Summary
    print(f"\n{'='*80}")
    print("📊 PHASE 45 SUMMARY — UNIVERSE TOPOLOGY")
    print(f"{'='*80}")

    print(f"\n  📋 Universe Geography:")
    print(f"  - Embedding dimension: {mods.shape[1]}")
    print(f"  - Effective dimensionality: {eff_dim} (95% variance)")
    print(f"  - Distribution: {'uniform' if cv < 0.3 else 'clustered' if cv > 0.5 else 'moderate'}")
    print(f"  - Spatial entropy: {spatial_entropy/max_entropy*100:.1f}% of maximum")
    print(f"  - Files occupy {eff_dim}D subspace of {mods.shape[1]}D space")

    print(f"\n  📋 Interpretation:")
    if eff_dim < mods.shape[1] // 2:
        print(f"  - Universe is LOW-DIMENSIONAL (files live in small subspace)")
        print(f"  - This means files share COMMON structure (validated shared roots!)")
    else:
        print(f"  - Universe is HIGH-DIMENSIONAL (files spread across dimensions)")
        print(f"  - Less structure sharing between files")

    print(f"\n  📋 Topology type:")
    if cv < 0.3:
        print(f"  - UNIFORM distribution (files spread evenly in universe)")
        print(f"  - No clusters — all files equally different")
    elif cv > 0.5:
        print(f"  - CLUSTERED distribution (files group by similarity)")
        print(f"  - Clusters = file families with shared structure")
    else:
        print(f"  - MODERATE distribution (some structure, some spread)")

    return {
        'eff_dim': eff_dim,
        'total_dim': mods.shape[1],
        'cv': cv,
        'spatial_entropy_pct': spatial_entropy / max_entropy * 100,
        'nn_mean': nn_distances.mean(),
    }


if __name__ == '__main__':
    results = run_phase45_experiment(verbose=True)
