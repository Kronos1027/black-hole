# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 85: Knowledge Distillation for SIREN Compression
========================================================
BHUH Phase II Wave 5

CONTEXT
-------
Phase 80 (linear Fisher projection) and Phase 82 (nonlinear autoencoder)
both failed to compress SIREN seeds via parameter-space projection.
The DEEPER finding was that SIREN's solution manifold is high-dimensional
despite the Fisher effective rank being low.

This phase tries a completely different approach: KNOWLEDGE DISTILLATION.
Instead of projecting the trained SIREN's parameters, we train a SMALLER
SIREN from scratch to mimic the OUTPUT of the larger SIREN.

HYPOTHESIS
----------
A smaller SIREN (e.g., hidden=8 instead of 32) can mimic the output of
a larger SIREN (hidden=32) with high fidelity, because:
  - The TARGET is now a smooth function (SIREN output) — easier to fit
  - Smooth functions have low Kolmogorov complexity (BHUH Axiom 2)
  - Distillation bypasses the high-dimensional solution manifold problem

EXPERIMENT
----------
1. Train a "teacher" SIREN (hidden=32, ~1185 params) on a target image
2. Train a "student" SIREN (hidden=8, ~129 params) to match teacher's output
3. Compare:
   - Student PSNR on original image (target: >25 dB)
   - Student size vs teacher size (reduction ratio)
   - Student training time
4. Sweep student sizes: hidden=4, 8, 16, 32 (control)

PREDICTION
----------
- Student hidden=8 should achieve >30 dB PSNR (9x parameter reduction)
- Student hidden=4 might struggle (<25 dB) for complex images
- This validates a NEW path to SIREN compression: distillation, not projection

If successful, this gives BHUH a working "Subspace Compression" via a
fundamentally different mechanism than Phase 80/82 attempted.

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import time
import json
import sys, os

sys.path.insert(0, os.path.dirname(__file__))


def make_smooth_image(n, kind='gaussian', cx=0.5, cy=0.5, sigma=0.15):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    if kind == 'gaussian':
        return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    if kind == 'sin':
        return (0.5 + 0.3 * np.sin(2 * np.pi * (X + Y))).astype(np.float32)
    if kind == 'plane':
        return (0.3 + 0.4 * X + 0.2 * Y).astype(np.float32)
    if kind == 'sinc':
        r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) + 1e-6
        return (0.5 + 0.4 * np.sin(6 * r) / (6 * r)).astype(np.float32)
    raise ValueError(kind)


def psnr(orig, pred):
    err = orig.flatten() - pred.flatten()
    mse = float(np.mean(err ** 2))
    pmax = float(orig.max())
    return 10 * np.log10(pmax ** 2 / max(mse, 1e-12))


def siren_param_count(hidden, n_layers=3, in_dim=2, out_dim=1):
    d = in_dim
    total = 0
    for k in range(n_layers - 1):
        total += d * hidden + hidden
        d = hidden
    total += d * out_dim + out_dim
    return total


def train_siren(coords, target, hidden=32, n_layers=3, omega=15.0, epochs=500, lr=1e-3,
                return_model=False):
    """Train a SIREN to fit a target image. Returns params (or model)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    class Siren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            d = 2
            for k in range(n_layers - 1):
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
    yt = torch.tensor(target.flatten(), dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(xt).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()

    if return_model:
        return model, float(loss.detach())

    params = []
    for p in model.parameters():
        params.append(p.detach().numpy().flatten())
    return np.concatenate(params), float(loss.detach())


def distill_siren(teacher_model, coords, student_hidden, n_layers=3, omega=15.0,
                   epochs=800, lr=1e-3):
    """Train a smaller student SIREN to mimic teacher's output."""
    import torch
    import torch.nn as nn
    torch.manual_seed(42)

    class Siren(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            d = 2
            for k in range(n_layers - 1):
                lin = nn.Linear(d, student_hidden)
                bound = 1.0 / d if k == 0 else np.sqrt(6.0 / student_hidden) / omega
                lin.weight.data.uniform_(-bound, bound)
                lin.bias.data.uniform_(-bound, bound)
                self.layers.append(lin)
                d = student_hidden
            self.head = nn.Linear(student_hidden, 1)
            bound = np.sqrt(6.0 / student_hidden) / omega
            self.head.weight.data.uniform_(-bound, bound)
            self.head.bias.data.uniform_(-bound, bound)
            self.omega = omega

        def forward(self, x):
            h = x
            for layer in self.layers:
                h = torch.sin(self.omega * layer(h))
            return self.head(h)

    student = Siren()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    xt = torch.tensor(coords, dtype=torch.float32)

    # Generate teacher outputs (target for distillation)
    teacher_model.eval()
    with torch.no_grad():
        teacher_out = teacher_model(xt).squeeze(-1)

    for ep in range(epochs):
        opt.zero_grad()
        student_out = student(xt).squeeze(-1)
        # Distillation loss: match teacher output
        loss = ((student_out - teacher_out) ** 2).mean()
        loss.backward()
        opt.step()

    # Extract params
    params = []
    for p in student.parameters():
        params.append(p.detach().numpy().flatten())
    return np.concatenate(params), float(loss.detach())


def predict_with_params(coords, theta, hidden, n_layers=3, omega=15.0):
    """Run forward pass with given params."""
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
            p.copy_(torch.tensor(theta[idx:idx+n].reshape(p.shape), dtype=torch.float32))
            idx += n
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(coords, dtype=torch.float32)).squeeze(-1).numpy()


def run_phase85():
    print("=" * 72)
    print("PHASE 85: Knowledge Distillation for SIREN Compression")
    print("=" * 72)
    print()

    import torch  # noqa

    N_PIX = 32
    coords = np.stack(np.meshgrid(np.linspace(0, 1, N_PIX),
                                  np.linspace(0, 1, N_PIX)), axis=-1).reshape(-1, 2)

    # Test images with varying complexity
    targets = {
        'gaussian': make_smooth_image(N_PIX, 'gaussian'),
        'sin':      make_smooth_image(N_PIX, 'sin'),
        'plane':    make_smooth_image(N_PIX, 'plane'),
        'sinc':     make_smooth_image(N_PIX, 'sinc'),
    }

    TEACHER_HIDDEN = 32
    TEACHER_PARAMS = siren_param_count(TEACHER_HIDDEN)
    print(f"Teacher: hidden={TEACHER_HIDDEN}, params={TEACHER_PARAMS}")
    print()

    # Student sizes to test
    student_hiddens = [4, 8, 16, 32]  # 32 is control (same as teacher)

    all_results = {}

    for target_name, target in targets.items():
        print(f"\n--- Target: {target_name} ---")
        # Train teacher (returns model and loss)
        t0 = time.time()
        teacher_model, teacher_loss = train_siren(coords, target, hidden=TEACHER_HIDDEN,
                                                   epochs=500, lr=1e-3, return_model=True)
        teacher_time = time.time() - t0
        # Get teacher prediction
        teacher_model.eval()
        with torch.no_grad():
            xt = torch.tensor(coords, dtype=torch.float32)
            teacher_pred = teacher_model(xt).squeeze(-1).numpy()
        teacher_psnr = psnr(target, teacher_pred)
        print(f"  Teacher: PSNR={teacher_psnr:.1f}dB, time={teacher_time:.2f}s, "
              f"params={TEACHER_PARAMS}")

        target_results = {
            'teacher_psnr': teacher_psnr,
            'teacher_params': TEACHER_PARAMS,
            'teacher_time_s': teacher_time,
            'students': []
        }

        for student_hidden in student_hiddens:
            student_params = siren_param_count(student_hidden)
            reduction = TEACHER_PARAMS / student_params

            t0 = time.time()
            student_theta, distill_loss = distill_siren(teacher_model, coords,
                                                        student_hidden,
                                                        epochs=800, lr=1e-3)
            student_time = time.time() - t0
            student_pred = predict_with_params(coords, student_theta, hidden=student_hidden)
            student_psnr_target = psnr(target, student_pred)  # PSNR vs original target
            student_psnr_teacher = psnr(teacher_pred, student_pred)  # PSNR vs teacher

            target_results['students'].append({
                'hidden': student_hidden,
                'params': student_params,
                'reduction_x': reduction,
                'distill_loss': distill_loss,
                'psnr_vs_target': student_psnr_target,
                'psnr_vs_teacher': student_psnr_teacher,
                'time_s': student_time,
            })
            print(f"  Student h={student_hidden}: params={student_params} ({reduction:.1f}x reduction), "
                  f"PSNR vs target={student_psnr_target:.1f}dB, "
                  f"PSNR vs teacher={student_psnr_teacher:.1f}dB, "
                  f"time={student_time:.2f}s")

        all_results[target_name] = target_results

    # ============================================================
    # SUMMARY
    # ============================================================
    print()
    print("=" * 72)
    print("DISTILLATION RESULTS")
    print("=" * 72)
    print()
    print(f"{'Target':<10} {'Teacher':>8} {'Stud h=4':>15} {'Stud h=8':>15} {'Stud h=16':>15} {'Stud h=32':>15}")
    print(f"{'':<10} {'PSNR/params':>15}", end='')
    for h in student_hiddens:
        print(f"  {'PSNR/red/loss':>13}", end='')
    print()

    for tname, tdata in all_results.items():
        print(f"{tname:<10} {tdata['teacher_psnr']:>6.1f}dB/{tdata['teacher_params']:>4d}", end='')
        for s in tdata['students']:
            print(f"  {s['psnr_vs_target']:>5.1f}dB/{s['reduction_x']:>4.1f}x", end='')
        print()

    # ============================================================
    # ANALYSIS
    # ============================================================
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()

    # Find smallest student that achieves >25 dB on each target
    best_students = {}
    for tname, tdata in all_results.items():
        best = None
        for s in sorted(tdata['students'], key=lambda x: x['params']):
            if s['psnr_vs_target'] > 25:
                best = s
                break
        best_students[tname] = best
        if best:
            print(f"  {tname}: smallest student h={best['hidden']} "
                  f"(params={best['params']}, reduction={best['reduction_x']:.1f}x, "
                  f"PSNR={best['psnr_vs_target']:.1f}dB)")
        else:
            print(f"  {tname}: NO student achieved PSNR > 25 dB")

    # Aggregate
    successful = sum(1 for v in best_students.values() if v is not None)
    print()
    print(f"  Successful distillations: {successful}/{len(all_results)}")
    if successful > 0:
        avg_reduction = np.mean([s['reduction_x'] for s in best_students.values() if s])
        avg_psnr = np.mean([s['psnr_vs_target'] for s in best_students.values() if s])
        print(f"  Average parameter reduction: {avg_reduction:.1f}x")
        print(f"  Average PSNR: {avg_psnr:.1f} dB")

    print()
    if successful >= 3:
        verdict = (f"VALIDATED — Knowledge distillation rescues SIREN compression where "
                   f"Phase 80 (linear) and Phase 82 (nonlinear projection) failed. "
                   f"Smallest successful students achieve {avg_reduction:.1f}x parameter "
                   f"reduction at {avg_psnr:.1f} dB PSNR. "
                   "Axiom 11 (Subspace Compression) accepted in DISTILLATION form — "
                   "different mechanism than originally proposed.")
        print("REVISED AXIOM 11 (Distillation Form):")
        print("  SIREN compression is achievable via knowledge distillation:")
        print("    - Train teacher SIREN (large) on target image")
        print("    - Train student SIREN (small) to mimic teacher output")
        print("    - Student size << teacher size, with high PSNR")
        print("  This works because the TARGET (teacher output) is smoother than")
        print("  the original image — easier to fit with small SIREN.")
    elif successful >= 1:
        verdict = (f"PARTIAL — Distillation works for some targets ({successful}/{len(all_results)}) "
                   "but not all. Image complexity matters.")
    else:
        verdict = "INVALID — Distillation also fails. SIREN compression may be fundamentally hard."

    print(f"\nVerdict: {verdict}")

    return {
        'phase': 85,
        'name': 'Knowledge Distillation for SIREN Compression',
        'verdict': verdict,
        'teacher_hidden': TEACHER_HIDDEN,
        'teacher_params': TEACHER_PARAMS,
        'n_targets': len(all_results),
        'n_successful': successful,
        'avg_reduction_x': float(avg_reduction) if successful > 0 else None,
        'avg_psnr_db': float(avg_psnr) if successful > 0 else None,
        'all_results': {k: {
            'teacher_psnr': v['teacher_psnr'],
            'students': v['students'],
        } for k, v in all_results.items()},
    }


if __name__ == '__main__':
    result = run_phase85()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
