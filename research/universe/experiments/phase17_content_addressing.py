# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 17: Neural Content Addressing (Dedup via SIREN)
======================================================
Tests whether SIREN embeddings can serve as content-addressable hashes
for deduplication.

CONCEPT:
  If two files are similar, their SIREN modulation vectors should be
  similar. We can use this for:
  1. Deduplication: detect near-duplicate files by modulation distance
  2. Content addressing: file ID = hash of modulation vector
  3. Similarity search: find similar files in O(1) via modulation lookup

HYPOTHESIS:
  Modulation vectors from Multi-File SIREN will cluster similar images
  together, enabling neural deduplication.

METHOD:
  1. Train Multi-File SIREN on 20 images (10 pairs of similar)
  2. Extract modulation vectors
  3. Compute pairwise distances
  4. Check if similar pairs have smaller distances

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import ModulatedSIREN, get_coordinates, measure_model_size_compressed
import torch.nn.functional as F


def generate_paired_images(n_pairs=10, size=64, seed=42):
    """Generate pairs of similar images (same base, different phase)."""
    images = []
    labels = []  # pair_id for each image

    for pair in range(n_pairs):
        rng = np.random.default_rng(seed + pair)
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
        kx = rng.integers(2, 5)
        ky = rng.integers(2, 5)
        amp = rng.uniform(40, 80)

        # Image A: phase 0
        img_a = np.zeros((size, size, 3), dtype=np.float32)
        for c in range(3):
            img_a[:, :, c] = amp * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
        img_a = ((img_a - img_a.min()) / (img_a.max() - img_a.min()) * 255).astype(np.uint8)
        images.append(img_a)
        labels.append(pair)

        # Image B: same frequencies, different phase (similar but not identical)
        phase = rng.uniform(0.5, 1.5)
        img_b = np.zeros((size, size, 3), dtype=np.float32)
        for c in range(3):
            img_b[:, :, c] = amp * np.sin(2 * np.pi * kx * xs + phase) * np.cos(2 * np.pi * ky * ys + phase * c)
        img_b = ((img_b - img_b.min()) / (img_b.max() - img_b.min()) * 255).astype(np.uint8)
        images.append(img_b)
        labels.append(pair)

    return images, labels


def extract_modulations(model, n_files):
    """Extract modulation vectors from trained model."""
    modulations = model.modulations.weight.detach().cpu().numpy()
    return modulations  # (n_files, modulation_dim)


def compute_distance_matrix(vectors):
    """Compute pairwise Euclidean distance matrix."""
    n = len(vectors)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.linalg.norm(vectors[i] - vectors[j])
    return dists


def run_phase17_experiment(verbose=True):
    """Run Phase 17 Neural Content Addressing experiment."""
    print("=" * 80)
    print("🧪 Phase 17: Neural Content Addressing (Dedup via SIREN)")
    print("=" * 80)

    device = 'cpu'

    # Generate paired images
    print("\n📦 Generating 10 pairs of similar images...")
    images, labels = generate_paired_images(n_pairs=10, size=64, seed=42)
    n_files = len(images)
    print(f"  {n_files} images (10 pairs, same freq different phase)")

    # Train Multi-File SIREN
    print("\n🌌 Training Multi-File SIREN...")
    from phase1_multi_file_siren import train_multi_file_siren
    model, loss = train_multi_file_siren(images, epochs=100, device=device, verbose=verbose)

    # Extract modulations
    print("\n🔍 Extracting modulation vectors...")
    modulations = extract_modulations(model, n_files)
    print(f"  Shape: {modulations.shape} ({n_files} files × {modulations.shape[1]} dim)")

    # Compute distance matrix
    dists = compute_distance_matrix(modulations)

    # Analyze: are pairs closer than non-pairs?
    pair_dists = []
    non_pair_dists = []

    for i in range(n_files):
        for j in range(i + 1, n_files):
            d = dists[i, j]
            if labels[i] == labels[j]:
                pair_dists.append(d)
            else:
                non_pair_dists.append(d)

    pair_avg = np.mean(pair_dists)
    non_pair_avg = np.mean(non_pair_dists)
    pair_std = np.std(pair_dists)
    non_pair_std = np.std(non_pair_dists)

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 17 RESULTS — NEURAL CONTENT ADDRESSING")
    print(f"{'='*80}")
    print(f"\n  {'Metric':<35} {'Value':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Pair distance (avg)':<35} {pair_avg:>9.4f}")
    print(f"  {'Pair distance (std)':<35} {pair_std:>9.4f}")
    print(f"  {'Non-pair distance (avg)':<35} {non_pair_avg:>9.4f}")
    print(f"  {'Non-pair distance (std)':<35} {non_pair_std:>9.4f}")
    print(f"  {'Separation ratio (non/pair)':<35} {non_pair_avg/pair_avg:>9.2f}x")

    # How many pairs are correctly identified?
    correct = 0
    for i in range(n_files):
        # Find closest other file
        dists_i = dists[i].copy()
        dists_i[i] = float('inf')
        closest = np.argmin(dists_i)
        if labels[closest] == labels[i]:
            correct += 1

    accuracy = correct / n_files * 100
    print(f"  {'Nearest-neighbor pair accuracy':<35} {accuracy:>8.1f}%")

    if accuracy >= 80:
        print(f"\n✅ Neural content addressing WORKS!")
        print(f"   Modulation vectors cluster similar files together")
        print(f"   {accuracy:.0f}% of files find their pair as nearest neighbor")
    elif accuracy >= 60:
        print(f"\n⚠️  Partial clustering — {accuracy:.0f}% accuracy")
    else:
        print(f"\n❌ Modulations don't cluster well ({accuracy:.0f}%)")

    print(f"\n  📋 Applications:")
    print(f"  - Deduplication: detect near-duplicates by modulation distance")
    print(f"  - Content addressing: file ID = hash of modulation vector")
    print(f"  - Similarity search: O(1) lookup in modulation space")

    return {
        'pair_avg': pair_avg,
        'non_pair_avg': non_pair_avg,
        'separation': non_pair_avg / pair_avg,
        'accuracy': accuracy,
    }


if __name__ == '__main__':
    results = run_phase17_experiment(verbose=True)
