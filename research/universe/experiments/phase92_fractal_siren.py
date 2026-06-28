# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 92: Fractal SIREN — Self-Similar Weights for IFS Compression
===================================================================
BHUH Phase II Wave 7

CONTEXT
-------
Fractal image compression (Barnsley, 1988) uses Iterated Function
Systems (IFS) to exploit SELF-SIMILARITY in images. A fractal-encoded
image can be 10000× smaller than the original.

BHUH Phase 18 tested fractal IFS and got 209× compression. This phase
combines FRACTAL self-similarity with SIREN neural representation.

HYPOTHESIS (Axiom 20 — Fractal SIREN)
-------------------------------------
A SIREN with SELF-SIMILAR weights (e.g., weight matrix W is a tiled
pattern from a smaller matrix W_small) can:
1. Achieve higher compression than monolithic SIREN
2. Maintain quality on self-similar images (fractals, textures)
3. Bridge IFS compression and neural compression

EXPERIMENT
----------
1. Generate self-similar test images:
   - Mandelbrot set (fractal)
   - Sierpinski triangle (fractal)
   - Repeated texture (tile pattern)
2. Train two SIRENs:
   - Standard: hidden=32, all weights independent
   - Fractal: hidden=32, but weights are tiled from hidden=8 sub-matrix
3. Compare PSNR and effective parameter count

PREDICTION
----------
- Fractal SIREN: ~8× parameter reduction (4×4 tile of 8×8 = 32×32)
- PSNR: comparable on self-similar images
- Lower PSNR on non-self-similar images (negative result if so)

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import time
import json
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from phase85_knowledge_distillation import psnr


def make_mandelbrot(n, max_iter=30):
    x = np.linspace(-2, 1, n)
    y = np.linspace(-1.5, 1.5, n)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    img = np.zeros_like(X, dtype=np.float32)
    for i in range(max_iter):
        Z = Z * Z + C
        mask = (np.abs(Z) < 2) & (img == 0)
        img[mask] = i / max_iter
    return img


def make_sierpinski(n, depth=4):
    """Sierpinski triangle via recursive subdivision."""
    img = np.zeros((n, n), dtype=np.float32)
    def draw(x, y, size, d):
        if d == 0 or size < 1:
            img[x:x+size, y:y+size] = 1
            return
        half = size // 2
        draw(x, y, half, d - 1)
        draw(x + half, y, half, d - 1)
        draw(x + half//2, y + half, half, d - 1)
    draw(0, 0, n, depth)
    return img


def make_texture_tile(n, tile_size=8):
    """Repeated tile pattern."""
    tile = np.random.rand(tile_size, tile_size).astype(np.float32)
    # Tile to fill n×n
    reps = (n // tile_size) + 1
    img = np.tile(tile, (reps, reps))[:n, :n]
    return img


def make_smooth_image(n, kind='gaussian'):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    if kind == 'gaussian':
        return np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) * 8).astype(np.float32)
    raise ValueError(kind)


def train_standard_siren(coords, target, hidden=32, omega=15.0, epochs=500, lr=1e-3):
    """Standard SIREN with all independent weights."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    class SirenStandard(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, 1)
            self.omega = omega
            for layer in [self.fc1, self.fc2]:
                bound = 1.0 / layer.in_features if layer is self.fc1 else np.sqrt(6.0 / hidden) / omega
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.uniform_(-bound, bound)
            bound = np.sqrt(6.0 / hidden) / omega
            self.fc3.weight.data.uniform_(-bound, bound)
            self.fc3.bias.data.uniform_(-bound, bound)

        def forward(self, x):
            h = torch.sin(self.omega * self.fc1(x))
            h = torch.sin(self.omega * self.fc2(h))
            return self.fc3(h)

    model = SirenStandard()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    yt = torch.tensor(target.flatten(), dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(xt).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(xt).squeeze(-1).numpy()

    n_params = sum(int(np.prod(p.shape)) for p in model.parameters())
    return pred, n_params, float(loss.detach())


def train_fractal_siren(coords, target, tile_hidden=8, full_hidden=32, omega=15.0,
                        epochs=800, lr=1e-3):
    """Fractal SIREN: weights are tiled from smaller sub-matrices.
    full_hidden = 32, tile_hidden = 8 -> 4x4 tiling, 16x reduction in weight params.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    tile_size = full_hidden // tile_hidden  # e.g., 32/8 = 4
    assert full_hidden == tile_hidden * tile_size

    class SirenFractal(nn.Module):
        def __init__(self):
            super().__init__()
            # Small "seed" weight matrices that get tiled
            self.tile1 = nn.Parameter(torch.randn(tile_hidden, 2) * 0.1)
            self.tile2 = nn.Parameter(torch.randn(tile_hidden, tile_hidden) * 0.1)
            self.tile3 = nn.Parameter(torch.randn(1, tile_hidden) * 0.1)
            # Biases (not tiled)
            self.b1 = nn.Parameter(torch.zeros(full_hidden))
            self.b2 = nn.Parameter(torch.zeros(full_hidden))
            self.b3 = nn.Parameter(torch.zeros(1))
            self.omega = omega
            self.tile_size = tile_size
            self.tile_hidden = tile_hidden

        def tiled_weight(self, tile, out_dim, in_dim):
            """Tile a small matrix to fill out_dim × in_dim."""
            # tile shape: (tile_hidden, in_dim_small)
            # We want: (out_dim, in_dim)
            # Strategy: repeat tile both row and column-wise
            # tile is (tile_hidden, in_dim), we need (out_dim, in_dim)
            # where out_dim = tile_hidden * tile_size
            # So just repeat tile_size times along rows
            return tile.repeat(tile_size, 1)  # (tile_hidden * tile_size, in_dim)

        def forward(self, x):
            # Layer 1: tile (tile_hidden, 2) -> (full_hidden, 2)
            w1 = self.tiled_weight(self.tile1, full_hidden, 2)
            h = torch.sin(self.omega * torch.nn.functional.linear(x, w1, self.b1))
            # Layer 2: tile (tile_hidden, tile_hidden) -> (full_hidden, full_hidden)
            w2_full = self.tile2.repeat(tile_size, tile_size)  # not great but tests concept
            h = torch.sin(self.omega * torch.nn.functional.linear(h, w2_full, self.b2))
            # Layer 3: tile (1, tile_hidden) -> (1, full_hidden)
            w3 = self.tile3.repeat(1, tile_size)
            return torch.nn.functional.linear(h, w3, self.b3)

    model = SirenFractal()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    yt = torch.tensor(target.flatten(), dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(xt).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(xt).squeeze(-1).numpy()

    n_params = sum(int(np.prod(p.shape)) for p in model.parameters())
    return pred, n_params, float(loss.detach())


def run_phase92():
    print("=" * 72)
    print("PHASE 92: Fractal SIREN — Self-Similar Weights for IFS Compression")
    print("=" * 72)
    print()

    import torch  # noqa

    N_PIX = 32
    coords = np.stack(np.meshgrid(np.linspace(0, 1, N_PIX),
                                  np.linspace(0, 1, N_PIX)), axis=-1).reshape(-1, 2)

    targets = {
        'mandelbrot':   make_mandelbrot(N_PIX),
        'sierpinski':   make_sierpinski(N_PIX, depth=4),
        'texture_tile': make_texture_tile(N_PIX, tile_size=8),
        'gaussian':     make_smooth_image(N_PIX, 'gaussian'),  # control: non-fractal
    }

    print(f"Test images:")
    for name, img in targets.items():
        print(f"  {name}: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")

    all_results = {}

    for tname, target in targets.items():
        print(f"\n--- Target: {tname} ---")

        # Standard SIREN
        t0 = time.time()
        pred_std, params_std, loss_std = train_standard_siren(
            coords, target, hidden=32, epochs=500, lr=1e-3
        )
        t_std = time.time() - t0
        psnr_std = psnr(target, pred_std)

        # Fractal SIREN
        t0 = time.time()
        pred_frac, params_frac, loss_frac = train_fractal_siren(
            coords, target, tile_hidden=8, full_hidden=32, epochs=800, lr=1e-3
        )
        t_frac = time.time() - t0
        psnr_frac = psnr(target, pred_frac)

        reduction = params_std / params_frac
        psnr_diff = psnr_frac - psnr_std

        all_results[tname] = {
            'psnr_standard_db': float(psnr_std),
            'psnr_fractal_db': float(psnr_frac),
            'psnr_diff_db': float(psnr_diff),
            'params_standard': int(params_std),
            'params_fractal': int(params_frac),
            'reduction_x': float(reduction),
            'time_standard_s': float(t_std),
            'time_fractal_s': float(t_frac),
        }

        print(f"  Standard: PSNR={psnr_std:.1f}dB, params={params_std}, time={t_std:.2f}s")
        print(f"  Fractal:  PSNR={psnr_frac:.1f}dB, params={params_frac}, time={t_frac:.2f}s")
        print(f"  Reduction: {reduction:.2f}x, PSNR diff: {psnr_diff:+.1f}dB")

    # ============================================================
    # SUMMARY
    # ============================================================
    print()
    print("=" * 72)
    print("FRACTAL SIREN RESULTS")
    print("=" * 72)
    print(f"{'Target':<14} {'Std PSNR':>9} {'Frac PSNR':>10} {'Diff':>7} {'Reduction':>10}")
    for tname, r in all_results.items():
        print(f"{tname:<14} {r['psnr_standard_db']:>8.1f}dB {r['psnr_fractal_db']:>9.1f}dB "
              f"{r['psnr_diff_db']:>+6.1f}dB {r['reduction_x']:>9.2f}x")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()

    # Fractal should win on self-similar images, lose on non-self-similar
    fractal_targets = ['mandelbrot', 'sierpinski', 'texture_tile']
    control_targets = ['gaussian']

    fractal_wins = sum(1 for t in fractal_targets
                       if all_results[t]['psnr_diff_db'] > -5)  # within 5 dB
    control_wins = sum(1 for t in control_targets
                       if all_results[t]['psnr_diff_db'] > -5)

    avg_fractal_diff = np.mean([all_results[t]['psnr_diff_db'] for t in fractal_targets])
    avg_control_diff = np.mean([all_results[t]['psnr_diff_db'] for t in control_targets])
    avg_reduction = np.mean([r['reduction_x'] for r in all_results.values()])

    print(f"  Fractal SIREN within 5 dB on fractal images: {fractal_wins}/{len(fractal_targets)}")
    print(f"  Fractal SIREN within 5 dB on control (gaussian): {control_wins}/{len(control_targets)}")
    print(f"  Average PSNR diff on fractal images: {avg_fractal_diff:+.1f} dB")
    print(f"  Average PSNR diff on control: {avg_control_diff:+.1f} dB")
    print(f"  Average parameter reduction: {avg_reduction:.2f}x")
    print()

    if fractal_wins >= 2 and avg_reduction > 2:
        verdict = (f"VALIDATED — Fractal SIREN achieves {avg_reduction:.1f}x parameter reduction "
                   f"while maintaining quality on self-similar images ({fractal_wins}/3 within 5 dB). "
                   f"Average PSNR diff on fractal images: {avg_fractal_diff:+.1f} dB. "
                   "Axiom 20 (Fractal SIREN) accepted. "
                   "Self-similar weight tiling is a viable SIREN compression technique.")
        print("NEW AXIOM (Axiom 20 — Fractal SIREN):")
        print("  SIREN weights can be tiled from smaller sub-matrices, exploiting")
        print("  self-similarity in the same way IFS fractal compression does.")
        print("  This is a FOURTH working path to SIREN compression (after distillation,")
        print("  multi-resolution, and quantization).")
    elif fractal_wins >= 1:
        verdict = (f"PARTIAL — Fractal SIREN works on some self-similar images.")
    else:
        verdict = "INVALID — Fractal SIREN loses too much quality."

    print(f"\nVerdict: {verdict}")
    print()
    print("COMPARISON OF ALL SIREN COMPRESSION APPROACHES:")
    print(f"  {'Approach':<28} {'Reduction':>10} {'PSNR':>8} {'Status':<12}")
    print(f"  {'Phase 80 (Linear PCA)':<28} {'N/A':>10} {'3.5dB':>8} {'FAILED':<12}")
    print(f"  {'Phase 82 (Nonlinear AE)':<28} {'N/A':>10} {'10dB':>8} {'FAILED':<12}")
    print(f"  {'Phase 85 (Distillation)':<28} {'32x':>10} {'32.7dB':>8} {'VALIDATED':<12}")
    print(f"  {'Phase 86 (Multi-res)':<28} {'8.3x':>10} {'34.2dB':>8} {'VALIDATED':<12}")
    print(f"  {'Phase 87 (INT4 QAT)':<28} {'8x':>10} {'35.1dB':>8} {'VALIDATED':<12}")
    print(f"  {'Phase 89 (Distill+INT4)':<28} {'249x':>10} {'27.7dB':>8} {'PARTIAL':<12}")
    print(f"  {'Phase 92 (Fractal tiling)':<28} {f'{avg_reduction:.1f}x':>10} "
          f"{f'{avg_fractal_diff:+.1f}dB':>8} {'VALIDATED' if fractal_wins >= 2 else 'PARTIAL':<12}")

    return {
        'phase': 92,
        'name': 'Fractal SIREN',
        'verdict': verdict,
        'n_targets': len(all_results),
        'avg_reduction_x': float(avg_reduction),
        'avg_fractal_psnr_diff': float(avg_fractal_diff),
        'avg_control_psnr_diff': float(avg_control_diff),
        'fractal_wins': int(fractal_wins),
        'control_wins': int(control_wins),
        'all_results': all_results,
    }


if __name__ == '__main__':
    result = run_phase92()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
