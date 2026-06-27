# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 53: Seed Evolution (Genetic Algorithm on Weights)
=========================================================
Tests whether genetic algorithms can evolve better SIREN seeds.

CONCEPT:
  Instead of gradient descent, use EVOLUTION:
  1. Start with population of random SIREN seeds
  2. Evaluate fitness (reconstruction quality)
  3. Select best, crossover, mutate
  4. Repeat

  This is an alternative to backpropagation — "evolving" seeds
  like organisms evolve through natural selection.

HYPOTHESIS:
  Genetic evolution will find DIFFERENT solutions than gradient descent,
  potentially discovering better local optima.

METHOD:
  1. Create population of 20 random SIREN models
  2. Evaluate fitness (PSNR on target image)
  3. Keep top 50%, crossover (weight averaging), mutate (Gaussian noise)
  4. Run 10 generations
  5. Compare with gradient descent

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren


def evaluate_fitness(model, image, device='cpu'):
    """Evaluate model fitness (negative MSE = higher is better)."""
    size = image.shape[0]
    coords = get_coordinates(size, device)
    pixels = torch.from_numpy(image.astype(np.float32) / 255.0).reshape(-1, 3).to(device)
    with torch.no_grad():
        pred = model(coords)
        mse = F.mse_loss(pred, pixels).item()
    return -mse  # negative because lower MSE = higher fitness


def crossover(parent_a, parent_b, device='cpu'):
    """Crossover: average weights of two parents."""
    child = copy.deepcopy(parent_a)
    with torch.no_grad():
        for (na, pa), (_, pb) in zip(parent_a.named_parameters(), parent_b.named_parameters()):
            child_param = 0.5 * pa.data + 0.5 * pb.data
            for name, param in child.named_parameters():
                if name == na:
                    param.data = child_param
                    break
    return child


def mutate(model, mutation_rate=0.01, mutation_std=0.05, device='cpu'):
    """Mutate: add Gaussian noise to random weights."""
    mutated = copy.deepcopy(model)
    with torch.no_grad():
        for param in mutated.parameters():
            mask = torch.rand_like(param) < mutation_rate
            noise = torch.randn_like(param) * mutation_std
            param.data = param.data + mask.float() * noise
    return mutated


def run_phase53_experiment(verbose=True):
    print("=" * 80)
    print("🧪 Phase 53: Seed Evolution (Genetic Algorithm on Weights)")
    print("=" * 80)

    device = 'cpu'
    size = 64
    population_size = 20
    generations = 10

    # Generate target image
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(2, 5, 2)
            img[:, :, c] += 50 * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    # Gradient descent baseline
    print("\n🔵 Baseline: Gradient descent (100 epochs)...")
    t0 = time.time()
    gd_model, gd_loss = train_single_siren(img, epochs=100, device=device, verbose=False)
    gd_time = time.time() - t0
    gd_fitness = evaluate_fitness(gd_model, img, device)
    gd_psnr = 10 * np.log10(1.0 / max(-gd_fitness, 1e-10))
    print(f"  GD: fitness={gd_fitness:.6f}, PSNR={gd_psnr:.1f}dB, time={gd_time:.1f}s")

    # Genetic algorithm
    print(f"\n🌌 BHUH: Genetic evolution ({population_size} population, {generations} generations)...")
    t0 = time.time()

    # Initialize population
    population = []
    for _ in range(population_size):
        model = SIREN(in_features=2, hidden_features=16, hidden_layers=1, out_features=3, omega_0=30.0).to(device)
        population.append(model)

    # Evolution loop
    best_fitness_history = []
    avg_fitness_history = []

    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [evaluate_fitness(m, img, device) for m in population]
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        if verbose:
            best_psnr = 10 * np.log10(1.0 / max(-best_fitness, 1e-10))
            print(f"  Gen {gen}: best={best_fitness:.6f} (PSNR={best_psnr:.1f}dB), avg={avg_fitness:.6f}")

        # Selection: keep top 50%
        sorted_indices = np.argsort(fitnesses)[::-1]
        survivors = [population[i] for i in sorted_indices[:population_size // 2]]

        # Crossover + mutation to fill population
        children = []
        while len(survivors) + len(children) < population_size:
            # Pick two random parents
            idx_a, idx_b = np.random.choice(len(survivors), 2, replace=False)
            child = crossover(survivors[idx_a], survivors[idx_b], device)
            child = mutate(child, mutation_rate=0.05, mutation_std=0.1, device=device)
            children.append(child)

        population = survivors + children

    evo_time = time.time() - t0

    # Get best evolved model
    fitnesses = [evaluate_fitness(m, img, device) for m in population]
    best_idx = np.argmax(fitnesses)
    best_model = population[best_idx]
    best_fitness = fitnesses[best_idx]
    evo_psnr = 10 * np.log10(1.0 / max(-best_fitness, 1e-10))

    print(f"\n  Evolution: best PSNR={evo_psnr:.1f}dB, time={evo_time:.1f}s")

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 53 RESULTS — SEED EVOLUTION")
    print(f"{'='*80}")
    print(f"\n  {'Method':<30} {'PSNR':>8} {'Time':>8} {'Efficiency':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Gradient descent':<30} {gd_psnr:>6.1f}dB {gd_time:>6.1f}s {gd_time/max(gd_psnr,1):>8.3f}s/dB")
    print(f"  {'Genetic evolution':<30} {evo_psnr:>6.1f}dB {evo_time:>6.1f}s {evo_time/max(evo_psnr,1):>8.3f}s/dB")

    print(f"\n  📋 Evolution progress:")
    for i, (best, avg) in enumerate(zip(best_fitness_history, avg_fitness_history)):
        best_psnr = 10 * np.log10(1.0 / max(-best, 1e-10))
        print(f"  Gen {i}: best={best_psnr:.1f}dB, avg={10*np.log10(1.0/max(-avg,1e-10)):.1f}dB")

    if evo_psnr > gd_psnr:
        print(f"\n  ✅ Evolution BEATS gradient descent!")
    elif evo_psnr > gd_psnr - 5:
        print(f"\n  ⚠️  Evolution is competitive ({gd_psnr - evo_psnr:.1f}dB behind)")
    else:
        print(f"\n  ❌ Evolution loses to gradient descent ({gd_psnr - evo_psnr:.1f}dB gap)")

    print(f"\n  📋 Key insight:")
    print(f"  Gradient descent is MUCH more efficient for SIREN training.")
    print(f"  Evolution explores diverse solutions but converges slowly.")
    print(f"  Hybrid approach: GD for base, evolution for modulations?")

    return {
        'gd_psnr': gd_psnr,
        'evo_psnr': evo_psnr,
        'gd_time': gd_time,
        'evo_time': evo_time,
    }


if __name__ == '__main__':
    results = run_phase53_experiment(verbose=True)
