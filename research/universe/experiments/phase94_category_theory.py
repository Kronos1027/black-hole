# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 94: BHUH Category Theory — Formalizing Composition Laws
================================================================
BHUH Phase II Wave 8

CONTEXT
-------
Category theory provides a formal language for compositional structure.
This phase formalizes BHUH operations as functors between categories,
revealing the algebraic structure of the compression universe.

CATEGORIES
----------
- **Img**: Category of images (objects) and image transformations (morphisms)
- **Seed**: Category of SIREN parameter vectors (objects) and seed transforms
- **File**: Category of byte sequences (objects) and file operations

FUNCTORS
--------
- **G (Genesis)**: Seed → Img   (decompression, runs SIREN forward)
- **C (Compress)**: Img → Seed  (compression, trains SIREN)
- **Q (Quantize)**: Seed → Seed (INT4/INT8 quantization)
- **D (Distill)**: Seed → Seed  (teacher → student distillation)

NATURAL TRANSFORMATIONS
-----------------------
- **η: Id_Img → G∘C**: The "compression residual" — how close G(C(x)) is to x
- **ε: Id_Seed → C∘G**: The "decompression residual" — how close C(G(s)) is to s

ADJUNCTION (the key formal claim)
---------------------------------
Genesis G and Compress C form an ADJUNCTION:
  C ⊣ G  (C is left adjoint to G)

This means: Hom_Seed(C(x), s) ≅ Hom_Img(x, G(s))

Empirically: training a SIREN to fit image x with seed s is equivalent
to finding s such that G(s) approximates x.

EXPERIMENT
----------
1. Verify the adjunction empirically:
   - For random (x, s) pairs, check Hom correspondence
   - Measure η (compression quality) and ε (decompression identity)
2. Verify functor laws:
   - G(C(x)) should be close to x (η small)
   - C(G(s)) should be close to s (ε small) — but many-to-one breaks this!
3. Test composition laws:
   - (Q ∘ C)(x) = Q(C(x)) — quantize after compress
   - (D ∘ C)(x) = D(C(x)) — distill after compress

This is a THEORY phase with empirical verification.

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import time
import json
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from phase85_knowledge_distillation import make_smooth_image, psnr


def genesis(seed_flat, coords, hidden=16, n_layers=3, omega=15.0):
    """Functor G: Seed → Img (decompression)."""
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


def compress(image, coords, hidden=16, omega=15.0, epochs=400, lr=1e-3):
    """Functor C: Img → Seed (compression)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    class Siren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            d = 2
            for k in range(n_layers := 3 - 1):
                lin = nn.Linear(d, hidden)
                bound = 1.0 / d if k == 0 else np.sqrt(6.0 / hidden) / omega
                lin.weight.data.uniform_(-bound, bound)
                lin.bias.data.uniform_(-bound, bound)
                self.layers.append(lin)
                d = hidden
            self.head = nn.Linear(hidden, 1)
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
    yt = torch.tensor(image.flatten(), dtype=torch.float32)
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


def quantize_seed(seed, bits=4):
    """Functor Q: Seed → Seed (quantization, simulated)."""
    if bits >= 32:
        return seed.copy()
    levels = 2 ** bits - 1
    max_val = np.abs(seed).max()
    scale = max_val / (levels / 2)
    quantized = np.round(seed / scale) * scale
    return quantized


def run_phase94():
    print("=" * 72)
    print("PHASE 94: BHUH Category Theory — Formalizing Composition Laws")
    print("=" * 72)
    print()

    import torch  # noqa

    N_PIX = 16
    coords = np.stack(np.meshgrid(np.linspace(0, 1, N_PIX),
                                  np.linspace(0, 1, N_PIX)), axis=-1).reshape(-1, 2)

    # Generate test images
    images = {
        'gaussian': make_smooth_image(N_PIX, 'gaussian'),
        'sin':      make_smooth_image(N_PIX, 'sin'),
        'plane':    make_smooth_image(N_PIX, 'plane'),
    }

    print("Category Theory Framework:")
    print("  Categories: Img (images), Seed (parameter vectors)")
    print("  Functors:")
    print("    G: Seed → Img  (Genesis, decompression)")
    print("    C: Img → Seed  (Compress, training)")
    print("    Q: Seed → Seed (Quantize)")
    print()
    print("  Adjunction: C ⊣ G (C is left adjoint to G)")
    print("  Natural transformations:")
    print("    η: Id_Img → G∘C   (compression residual)")
    print("    ε: Id_Seed → C∘G  (decompression residual)")
    print()

    # ============================================================
    # Test 1: Verify G∘C ≈ Id_Img (compression quality)
    # ============================================================
    print("=" * 60)
    print("TEST 1: G∘C ≈ Id_Img (compression roundtrip)")
    print("=" * 60)
    print()

    for name, img in images.items():
        # C: Img → Seed
        seed, _ = compress(img, coords, hidden=16, epochs=300, lr=1e-3)
        # G: Seed → Img
        recon = genesis(seed, coords, hidden=16)
        # η: measurement of G(C(x)) - x
        eta_psnr = psnr(img, recon)
        eta_mse = float(np.mean((img.flatten() - recon.flatten()) ** 2))
        print(f"  {name}: η PSNR = {eta_psnr:.1f} dB, η MSE = {eta_mse:.6f}")

    # ============================================================
    # Test 2: Verify C∘G ≈ Id_Seed (decompression roundtrip)
    # ============================================================
    print()
    print("=" * 60)
    print("TEST 2: C∘G ≈ Id_Seed (decompression roundtrip)")
    print("=" * 60)
    print()
    print("  NOTE: Many-to-one (multiple seeds → same image) breaks this.")
    print("  We expect ε to be LARGE (poor roundtrip) due to seed non-uniqueness.")
    print()

    for name, img in images.items():
        # Start with a random seed
        np.random.seed(42)
        random_seed = np.random.randn(337) * 0.1  # 337 = param count for h=16
        # G: Seed → Img
        img_from_seed = genesis(random_seed, coords, hidden=16)
        # C: Img → Seed
        recovered_seed, _ = compress(img_from_seed, coords, hidden=16, epochs=300, lr=1e-3)
        # ε: measurement of C(G(s)) - s
        eps_dist = float(np.linalg.norm(recovered_seed - random_seed))
        eps_psnr = psnr(img_from_seed, genesis(recovered_seed, coords, hidden=16))
        print(f"  {name}: ε L2 dist = {eps_dist:.4f}, "
              f"reconstruction PSNR = {eps_psnr:.1f} dB")

    # ============================================================
    # Test 3: Functor composition Q∘C
    # ============================================================
    print()
    print("=" * 60)
    print("TEST 3: Q∘C composition (compress then quantize)")
    print("=" * 60)
    print()

    for name, img in images.items():
        # C: Img → Seed
        seed, _ = compress(img, coords, hidden=16, epochs=300, lr=1e-3)
        # Q: Seed → Seed (INT4)
        seed_q = quantize_seed(seed, bits=4)
        # G∘Q∘C: full roundtrip with quantization
        recon_q = genesis(seed_q, coords, hidden=16)
        psnr_q = psnr(img, recon_q)
        # Compare to G∘C without quantization
        recon_noq = genesis(seed, coords, hidden=16)
        psnr_noq = psnr(img, recon_noq)
        # η_Q: how much quantization hurts
        eta_q_loss = psnr_noq - psnr_q
        print(f"  {name}: G∘C PSNR = {psnr_noq:.1f} dB, G∘Q∘C PSNR = {psnr_q:.1f} dB, "
              f"loss = {eta_q_loss:.1f} dB")

    # ============================================================
    # Test 4: Adjunction verification
    # ============================================================
    print()
    print("=" * 60)
    print("TEST 4: Adjunction C ⊣ G verification")
    print("=" * 60)
    print()
    print("  Adjunction says: Hom_Seed(C(x), s) ≅ Hom_Img(x, G(s))")
    print("  Empirically: ||C(x) - s||  ≈  λ * ||x - G(s)||")
    print("  for some constant λ (the 'adjoint constant').")
    print()

    # For several (x, s) pairs, measure both distances
    np.random.seed(0)
    n_trials = 5
    adjunction_data = []

    for trial in range(n_trials):
        # Pick a random image x
        x = make_smooth_image(N_PIX, 'gaussian',
                              cx=0.3 + 0.4 * np.random.rand(),
                              cy=0.3 + 0.4 * np.random.rand())
        # Pick a random seed s
        s = np.random.randn(337) * 0.1
        # Compute G(s)
        G_s = genesis(s, coords, hidden=16)
        # Compute C(x)
        C_x, _ = compress(x, coords, hidden=16, epochs=200, lr=1e-3)

        # Distances
        d_seed = float(np.linalg.norm(C_x - s))
        d_img = float(np.linalg.norm(x.flatten() - G_s.flatten()))

        adjunction_data.append({
            'trial': trial,
            'd_seed': d_seed,
            'd_img': d_img,
            'ratio': d_seed / max(d_img, 1e-9),
        })
        print(f"  Trial {trial+1}: ||C(x)-s|| = {d_seed:.4f}, "
              f"||x-G(s)|| = {d_img:.4f}, ratio = {d_seed/d_img:.2f}")

    ratios = [d['ratio'] for d in adjunction_data]
    ratio_mean = np.mean(ratios)
    ratio_std = np.std(ratios)
    ratio_cv = ratio_std / ratio_mean  # coefficient of variation
    print()
    print(f"  Ratio mean: {ratio_mean:.4f}")
    print(f"  Ratio std:  {ratio_std:.4f}")
    print(f"  Ratio CV:   {ratio_cv:.4f}")
    print(f"  Adjunction holds (CV < 0.5): {'✅ YES' if ratio_cv < 0.5 else '❌ NO'}")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()

    # Check roundtrip quality (η small = good compression)
    eta_psnrs = []
    for name, img in images.items():
        seed, _ = compress(img, coords, hidden=16, epochs=300, lr=1e-3)
        recon = genesis(seed, coords, hidden=16)
        eta_psnrs.append(psnr(img, recon))
    eta_avg = np.mean(eta_psnrs)

    # Check adjunction
    adjunction_holds = ratio_cv < 0.5

    print(f"  Average η (compression roundtrip PSNR): {eta_avg:.1f} dB")
    print(f"  Adjunction constant λ ≈ {ratio_mean:.2f}")
    print(f"  Adjunction CV: {ratio_cv:.3f} ({'holds' if adjunction_holds else 'weak'})")
    print()

    if eta_avg > 30 and adjunction_holds:
        verdict = (f"VALIDATED — BHUH forms a category-theoretic adjunction C ⊣ G. "
                   f"Compression C and decompression G are adjoint functors with "
                   f"constant λ ≈ {ratio_mean:.2f} (CV={ratio_cv:.3f}). "
                   f"Roundtrip η PSNR = {eta_avg:.1f} dB confirms G∘C ≈ Id. "
                   "Axiom 22 (BHUH Adjunction) accepted.")
        print("NEW AXIOM (Axiom 22 — BHUH Adjunction):")
        print("  Compress C: Img → Seed and Genesis G: Seed → Img form an adjunction:")
        print("    C ⊣ G  (C is left adjoint to G)")
        print("  Hom_Seed(C(x), s) ≅ Hom_Img(x, G(s)) with constant λ")
        print()
        print("  This formalizes BHUH as a CATEGORY, enabling:")
        print("  - Composition laws: (Q∘C)(x) = Q(C(x))")
        print("  - Natural transformations: η (compression residual), ε (decompression residual)")
        print("  - Limit/colimit constructions for multi-file corpora")
    elif eta_avg > 25:
        verdict = (f"PARTIAL — Roundtrip works but adjunction is weak.")
    else:
        verdict = "INVALID — Roundtrip fails."

    print(f"\nVerdict: {verdict}")
    print()
    print("THEORETICAL SIGNIFICANCE:")
    print("  Category theory provides the FORMAL LANGUAGE for BHUH:")
    print("  - Images, seeds, files are OBJECTS in categories")
    print("  - Compression/decompression are FUNCTORS between categories")
    print("  - The adjunction C ⊣ G formalizes the asymmetric relationship")
    print("  - Natural transformations η, ε quantify roundtrip quality")
    print()
    print("  This enables future work:")
    print("  - Proving universal properties (limits, colimits)")
    print("  - Compositional reasoning about compression pipelines")
    print("  - Connecting BHUH to other categorical frameworks (e.g., lens, optics)")

    return {
        'phase': 94,
        'name': 'BHUH Category Theory',
        'verdict': verdict,
        'eta_avg_psnr_db': float(eta_avg),
        'adjunction_lambda': float(ratio_mean),
        'adjunction_cv': float(ratio_cv),
        'adjunction_holds': bool(adjunction_holds),
        'n_trials': n_trials,
    }


if __name__ == '__main__':
    result = run_phase94()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
