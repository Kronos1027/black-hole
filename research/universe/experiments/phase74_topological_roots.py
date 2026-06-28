# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 74: Topological Roots — Persistent Homology as Universal Features
========================================================================
BHUH Phase II — Beyond Singularities

MOTIVATION
----------
BHUH claims files share "roots" across domains (Phase 6). But what ARE
these roots mathematically? This phase proposes that TOPOLOGICAL
INVARIANTS — specifically, persistent homology features (Betti numbers,
persistence diagrams) — are universal roots that transcend representation.

Topological Data Analysis (TDA) extracts shape information that is:
  - Coordinate-free (invariant to rotations/translations)
  - Scale-aware (persistence across scales)
  - Domain-agnostic (works on images, audio, point clouds, graphs)

HYPOTHESIS
----------
The SIREN network learns features that correlate with topological
invariants of the input data. If two files have similar persistence
diagrams (same Betti numbers), they share roots in BHUH space.

EXPERIMENT
----------
1. Generate families of synthetic images with KNOWN topologies:
   - 0 connected components (β₀)
   - 1 hole (β₁)
   - 2 holes
   - 3 holes
   - various positions
2. Compute persistence diagrams (Betti numbers) via cubical homology
3. Train SIREN on each, compute parameter distance
4. Correlate topological distance with SIREN parameter distance

LIMITATION: This is a Python-only experiment (no gudhi/ripser).
We use a simple Betti number counter via flood fill for binary images.

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import json
import time
from collections import deque


# ============================================================
# Topology: Betti numbers via flood fill (binary images)
# ============================================================

def to_binary(arr, threshold=0.5):
    return (arr > threshold).astype(np.uint8)


def betti_numbers(binary):
    """Compute Betti-0 (connected components) and Betti-1 (holes)
    for a 2D binary image via flood fill."""
    H, W = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    b0 = 0
    b1 = 0

    # β₀: count connected components of foreground (1s)
    for i in range(H):
        for j in range(W):
            if binary[i, j] == 1 and not visited[i, j]:
                b0 += 1
                # BFS
                q = deque([(i, j)])
                visited[i, j] = True
                while q:
                    ci, cj = q.popleft()
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < H and 0 <= nj < W and binary[ni, nj] == 1 and not visited[ni, nj]:
                            visited[ni, nj] = True
                            q.append((ni, nj))

    # β₁: count holes = connected components of background (0s) that are NOT the outer background
    # Approach: flood-fill from border, then count remaining 0-regions
    visited_bg = np.zeros_like(binary, dtype=bool)
    # Mark border 0-cells as outer background
    q = deque()
    for i in range(H):
        for j in [0, W - 1]:
            if binary[i, j] == 0 and not visited_bg[i, j]:
                visited_bg[i, j] = True
                q.append((i, j))
    for j in range(W):
        for i in [0, H - 1]:
            if binary[i, j] == 0 and not visited_bg[i, j]:
                visited_bg[i, j] = True
                q.append((i, j))
    while q:
        ci, cj = q.popleft()
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < H and 0 <= nj < W and binary[ni, nj] == 0 and not visited_bg[ni, nj]:
                visited_bg[ni, nj] = True
                q.append((ni, nj))

    # Now count remaining 0-regions (these are holes)
    for i in range(H):
        for j in range(W):
            if binary[i, j] == 0 and not visited_bg[i, j]:
                b1 += 1
                q = deque([(i, j)])
                visited_bg[i, j] = True
                while q:
                    ci, cj = q.popleft()
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < H and 0 <= nj < W and binary[ni, nj] == 0 and not visited_bg[ni, nj]:
                            visited_bg[ni, nj] = True
                            q.append((ni, nj))

    return b0, b1


# ============================================================
# Image generators with known topology
# ============================================================

def make_blobs(n, centers, radius=0.15):
    """β₀ = len(centers), β₁ = 0"""
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    img = np.zeros((n, n), dtype=np.float32)
    for (cx, cy) in centers:
        img += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * (radius / 3) ** 2))
    return np.clip(img, 0, 1)


def make_rings(n, centers, radius=0.12, thickness=0.06):
    """β₁ = len(centers) (each ring has 1 hole).
    Uses a hard ring (annulus) to ensure clean topology."""
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    img = np.zeros((n, n), dtype=np.float32)
    for (cx, cy) in centers:
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # Hard annulus: 1 if radius-thickness/2 < r < radius+thickness/2
        ring = ((r > radius - thickness / 2) & (r < radius + thickness / 2)).astype(np.float32)
        img = np.maximum(img, ring)
    return img


def make_grid_holes(n, n_holes=4):
    """Image with n_holes punched out"""
    img = np.ones((n, n), dtype=np.float32)
    hole_r = 0.08
    positions = np.linspace(0.2, 0.8, int(np.ceil(np.sqrt(n_holes))))[:n_holes]
    for i, p in enumerate(positions):
        cx = 0.25 + 0.5 * (i % 2)
        cy = 0.25 + 0.5 * (i // 2)
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        img *= (r > hole_r).astype(np.float32)
    return img


# ============================================================
# Small SIREN
# ============================================================

def fit_siren(coords, target, hidden=64, n_layers=3, omega=15.0, epochs=500, lr=1e-3):
    import torch
    torch.manual_seed(0)

    class Siren(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList()
            d = 2
            for k in range(n_layers - 1):
                lin = torch.nn.Linear(d, hidden)
                bound = 1.0 / d if k == 0 else np.sqrt(6.0 / hidden) / omega
                lin.weight.data.uniform_(-bound, bound)
                lin.bias.data.uniform_(-bound, bound)
                self.layers.append(lin)
                d = hidden
            self.head = torch.nn.Linear(hidden, 1)
            bound = np.sqrt(6.0 / hidden) / omega
            self.head.weight.data.uniform_(-bound, bound)
            self.head.bias.data.uniform_(-bound, bound)
            self.omega = omega

        def forward(self, x):
            h = x
            for layer in self.layers:
                h = torch.sin(self.omega * layer(h))
            return self.head(h)

    model = Siren()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)
    yt = torch.tensor(target.flatten(), dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(xt).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()

    params = []
    for p in model.parameters():
        params.append(p.detach().numpy().flatten())
    return np.concatenate(params), float(loss.detach())


def param_distance(p1, p2):
    return float(np.linalg.norm(p1 - p2))


def run_phase74():
    print("=" * 72)
    print("PHASE 74: Topological Roots — Persistent Homology as Universal Features")
    print("=" * 72)
    print()

    import torch  # noqa

    N_PIX = 32
    coords = np.stack(np.meshgrid(np.linspace(0, 1, N_PIX),
                                  np.linspace(0, 1, N_PIX)), axis=-1).reshape(-1, 2)

    # Generate topological families — 3 variants each (same topology, different position)
    families = {
        '1_blob':  [make_blobs(N_PIX, [(0.5, 0.5)]),
                    make_blobs(N_PIX, [(0.3, 0.3)]),
                    make_blobs(N_PIX, [(0.7, 0.7)])],
        '2_blobs': [make_blobs(N_PIX, [(0.3, 0.5), (0.7, 0.5)]),
                    make_blobs(N_PIX, [(0.5, 0.3), (0.5, 0.7)]),
                    make_blobs(N_PIX, [(0.2, 0.2), (0.8, 0.8)])],
        '3_blobs': [make_blobs(N_PIX, [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]),
                    make_blobs(N_PIX, [(0.3, 0.7), (0.7, 0.7), (0.5, 0.3)]),
                    make_blobs(N_PIX, [(0.2, 0.5), (0.5, 0.2), (0.8, 0.5)])],
        '1_ring':  [make_rings(N_PIX, [(0.5, 0.5)]),
                    make_rings(N_PIX, [(0.3, 0.3)]),
                    make_rings(N_PIX, [(0.7, 0.7)])],
        '2_rings': [make_rings(N_PIX, [(0.3, 0.5), (0.7, 0.5)]),
                    make_rings(N_PIX, [(0.5, 0.3), (0.5, 0.7)]),
                    make_rings(N_PIX, [(0.2, 0.2), (0.8, 0.8)])],
        '3_rings': [make_rings(N_PIX, [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]),
                    make_rings(N_PIX, [(0.3, 0.7), (0.7, 0.7), (0.5, 0.3)]),
                    make_rings(N_PIX, [(0.2, 0.5), (0.5, 0.2), (0.8, 0.5)])],
    }

    # Compute Betti numbers and fit SIREN
    print("--- Family: Betti numbers + SIREN fit ---")
    print(f"{'Family':<10} {'Variant':>8} {'β₀':>4} {'β₁':>4} {'SIREN loss':>12} {'# params':>10}")
    family_data = []
    for name, imgs in families.items():
        for v_idx, img in enumerate(imgs):
            b0, b1 = betti_numbers(to_binary(img))
            params, loss = fit_siren(coords, img, hidden=32, n_layers=3, omega=15.0,
                                      epochs=400, lr=1e-3)
            family_data.append({
                'name': name,
                'variant': v_idx,
                'image': img,
                'b0': b0,
                'b1': b1,
                'params': params,
                'loss': float(loss.detach()) if hasattr(loss, 'detach') else float(loss),
            })
            print(f"{name:<10} {v_idx:>8} {b0:>4} {b1:>4} {float(loss):>12.6f} {len(params):>10}")

    # ============================================================
    # Compute pairwise distances
    # ============================================================
    print()
    print("--- Topological distance vs SIREN parameter distance ---")

    # Topological distance: Hamming on (β₀, β₁)
    # SIREN distance: L2 norm of param vectors (normalized)
    topo_dists = []
    siren_dists = []
    same_b0 = []
    same_b1 = []
    same_topo = []

    n_fam = len(family_data)
    for i in range(n_fam):
        for j in range(i + 1, n_fam):
            a = family_data[i]
            b = family_data[j]
            t_d = abs(a['b0'] - b['b0']) + abs(a['b1'] - b['b1'])
            # Normalize params before L2
            pa = a['params'] / (np.linalg.norm(a['params']) + 1e-12)
            pb = b['params'] / (np.linalg.norm(b['params']) + 1e-12)
            s_d = float(np.linalg.norm(pa - pb))
            topo_dists.append(t_d)
            siren_dists.append(s_d)
            same_b0.append(a['b0'] == b['b0'])
            same_b1.append(a['b1'] == b['b1'])
            same_topo.append(t_d == 0)

    # Compute correlation
    topo_arr = np.array(topo_dists, dtype=float)
    siren_arr = np.array(siren_dists, dtype=float)

    # Group: same topology vs different topology
    same_mask = np.array(same_topo)
    diff_mask = ~same_mask

    same_dists = siren_arr[same_mask] if same_mask.any() else np.array([])
    diff_dists = siren_arr[diff_mask] if diff_mask.any() else np.array([])

    print(f"  Pairs with SAME topology (β₀, β₁):  n={len(same_dists)}, "
          f"mean SIREN dist={same_dists.mean() if len(same_dists) > 0 else 0:.4f}")
    print(f"  Pairs with DIFFERENT topology:       n={len(diff_dists)}, "
          f"mean SIREN dist={diff_dists.mean() if len(diff_dists) > 0 else 0:.4f}")

    # Statistical test: are SIREN distances smaller for same-topology pairs?
    from scipy import stats
    if len(same_dists) > 1 and len(diff_dists) > 1:
        # Simple t-test
        t_stat, p_val = stats.ttest_ind(same_dists, diff_dists, equal_var=False)
        print(f"  Welch t-test: t={t_stat:.3f}, p={p_val:.4f}")
        sig = p_val < 0.05
    else:
        p_val = 1.0
        sig = False

    # ============================================================
    # Correlation between topological and SIREN distance
    # ============================================================
    if len(topo_arr) > 2 and len(set(topo_arr.tolist())) > 1:
        corr, corr_p = stats.spearmanr(topo_arr, siren_arr)
        print(f"  Spearman correlation (topo vs SIREN): ρ={corr:.3f}, p={corr_p:.4f}")
    else:
        corr, corr_p = 0.0, 1.0

    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()
    print(f"  Same-topology pairs have {'SMALLER' if same_dists.mean() < diff_dists.mean() else 'LARGER'} SIREN distance")
    print(f"  Statistical significance (p<0.05): {'YES' if sig else 'NO'}")
    print(f"  Spearman correlation: {corr:.3f} ({'strong' if abs(corr)>0.5 else 'weak'})")
    print()

    if sig and corr > 0.3:
        verdict = ("VALIDATED — Topology is a universal root. SIREN parameters "
                   "encode topological invariants: same-topology files have smaller "
                   "parameter distance. This adds Axiom 7 (Topological Roots).")
    elif corr_p < 0.01 and corr > 0.2:
        verdict = ("PARTIAL — Topology WEAKLY correlates with SIREN parameter distance "
                   f"(Spearman ρ={corr:.3f}, p={corr_p:.2e}). Topology is one factor "
                   "among many (geometry, frequency, intensity also matter). Axiom 7 "
                   "holds as a statistical tendency, not a strict law.")
    elif sig:
        verdict = "PARTIAL — Topology influences SIREN params but correlation is weak."
    else:
        verdict = ("INCONCLUSIVE — Topology does not significantly determine SIREN "
                   "parameter distance. Roots may live in deeper geometric structure.")

    print(f"Verdict: {verdict}")
    print()
    print("PROPOSED NEW AXIOM:")
    print("  Axiom 7 (Topological Roots): The 'roots' shared between files in a BHUH")
    print("  universe are partially determined by topological invariants (Betti numbers).")
    print("  Formal: ∀x, y: Betti(x) = Betti(y)  ⟹  d_bkuh(x, y) < d_bkuh(x, z) for z with Betti(z) ≠ Betti(x)")

    return {
        'phase': 74,
        'name': 'Topological Roots',
        'verdict': verdict,
        'n_families': len(family_data),
        'n_pairs': len(topo_arr),
        'same_topo_count': int(same_mask.sum()),
        'diff_topo_count': int(diff_mask.sum()),
        'mean_siren_dist_same': float(same_dists.mean()) if len(same_dists) > 0 else 0,
        'mean_siren_dist_diff': float(diff_dists.mean()) if len(diff_dists) > 0 else 0,
        'welch_p_value': float(p_val),
        'spearman_corr': float(corr),
        'spearman_p': float(corr_p),
        'significant': bool(sig),
    }


if __name__ == '__main__':
    result = run_phase74()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
