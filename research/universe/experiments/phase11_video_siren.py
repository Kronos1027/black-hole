# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 11: Video Compression via 3D SIREN (x, y, t)
====================================================
Tests whether the shared roots principle extends to VIDEO.

CONCEPT:
  A video is a 3D signal: f(x, y, t) → RGB
  Instead of compressing each frame independently, we train ONE SIREN
  that maps (x, y, t) coordinates to pixel values.

  Temporal coherence (frame-to-frame similarity) becomes "shared roots"
  in the time dimension.

HYPOTHESIS:
  A single 3D SIREN compressing 16 video frames will achieve 5-10x
  improvement over compressing 16 separate images, because temporal
  redundancy is captured by the network.

METHOD:
  1. Generate synthetic video (16 frames, moving pattern)
  2. Baseline: 16 separate 2D SIRENs (one per frame)
  3. BHUH: 1 × 3D SIREN (x, y, t → RGB)
  4. Compare compressed sizes

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, SIRENLayer, get_coordinates, measure_model_size_compressed


class VideoSIREN(nn.Module):
    """3D SIREN: f(x, y, t) → RGB. Compresses entire video in one network."""
    def __init__(self, hidden_features=32, hidden_layers=2, omega_0=30.0):
        super().__init__()
        layers = [SIRENLayer(3, hidden_features, is_first=True, omega_0=omega_0)]  # 3D input
        for _ in range(hidden_layers):
            layers.append(SIRENLayer(hidden_features, hidden_features, omega_0=omega_0))
        layers.append(nn.Linear(hidden_features, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def generate_synthetic_video(n_frames=16, size=64, seed=42):
    """Generate synthetic video with moving pattern."""
    rng = np.random.default_rng(seed)
    frames = []

    # Base pattern that moves over time
    freq_x = rng.integers(2, 5)
    freq_y = rng.integers(2, 5)
    speed_x = rng.uniform(0.5, 2.0)
    speed_y = rng.uniform(0.5, 2.0)
    phase = rng.uniform(0, 2 * np.pi)

    for t_idx in range(n_frames):
        t = t_idx / n_frames
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size

        img = np.zeros((size, size, 3), dtype=np.float32)
        for c in range(3):
            # Moving wave pattern
            img[:, :, c] = 80 * np.sin(2 * np.pi * freq_x * (xs + speed_x * t) + phase) * \
                           np.cos(2 * np.pi * freq_y * (ys + speed_y * t) + phase * c)

        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        frames.append(img)

    return frames


def train_video_siren(frames, epochs=150, lr=3e-3, device='cpu', verbose=False):
    """Train 3D SIREN on video frames."""
    n_frames = len(frames)
    size = frames[0].shape[0]

    # Generate 3D coordinates: (x, y, t) for all frames
    coords_list = []
    pixels_list = []

    for t_idx, frame in enumerate(frames):
        t_val = t_idx / n_frames
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
        # (size*size, 3) with (y, x, t)
        frame_coords = np.stack([ys.ravel(), xs.ravel(), np.full(size * size, t_val)], axis=-1)
        frame_pixels = frame.astype(np.float32).reshape(-1, 3) / 255.0

        coords_list.append(frame_coords)
        pixels_list.append(frame_pixels)

    all_coords = torch.from_numpy(np.vstack(coords_list).astype(np.float32)).to(device)
    all_pixels = torch.from_numpy(np.vstack(pixels_list).astype(np.float32)).to(device)

    model = VideoSIREN(hidden_features=32, hidden_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(all_coords)
        loss = F.mse_loss(pred, all_pixels)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 30 == 0:
            print(f"  Video SIREN Epoch {epoch}: loss={loss.item():.6f}")

    return model, loss.item()


def train_separate_sirens(frames, epochs=80, device='cpu'):
    """Train separate 2D SIRENs for each frame (baseline)."""
    from phase1_multi_file_siren import train_single_siren
    total_size = 0
    for frame in frames:
        model, _ = train_single_siren(frame, epochs=epochs, device=device, verbose=False)
        total_size += measure_model_size_compressed(model)
    return total_size


def run_phase11_experiment(verbose=True):
    """Run Phase 11 Video Compression experiment."""
    print("=" * 80)
    print("🧪 Phase 11: Video Compression via 3D SIREN (x, y, t)")
    print("=" * 80)

    device = 'cpu'

    # Generate video
    n_frames = 16
    size = 64
    print(f"\n🎬 Generating synthetic video: {n_frames} frames @ {size}x{size}...")
    frames = generate_synthetic_video(n_frames, size)
    total_raw = sum(f.nbytes for f in frames)
    total_zip = sum(len(zlib.compress(f.tobytes(), 9)) for f in frames)
    print(f"  Total raw: {total_raw:,}B ({total_raw/1024:.1f}KB)")
    print(f"  Total ZIP: {total_zip:,}B ({total_zip/1024:.1f}KB)")

    # Baseline: separate 2D SIRENs
    print(f"\n🔵 Baseline: {n_frames} separate 2D SIRENs (one per frame)...")
    t0 = time.time()
    separate_size = train_separate_sirens(frames, epochs=60, device=device)
    separate_time = time.time() - t0
    print(f"  Total: {separate_size:,}B in {separate_time:.1f}s")

    # BHUH: Single 3D SIREN
    print(f"\n🌌 BHUH: Single 3D SIREN (x, y, t → RGB)...")
    t0 = time.time()
    video_model, video_loss = train_video_siren(frames, epochs=120, device=device, verbose=verbose)
    video_time = time.time() - t0
    video_size = measure_model_size_compressed(video_model)
    print(f"  Total: {video_size:,}B in {video_time:.1f}s")

    # Results
    improvement = separate_size / max(video_size, 1)
    vs_zip = total_zip / max(video_size, 1)

    print(f"\n{'='*80}")
    print("📊 PHASE 11 RESULTS — VIDEO COMPRESSION")
    print(f"{'='*80}")
    print(f"\n  {'Method':<40} {'Size':>10} {'vs Separate':>12} {'vs ZIP':>10}")
    print(f"  {'-'*75}")
    print(f"  {'ZIP (zlib-9)':<40} {total_zip:>9,}B {'-':>11} {'1.00x':>9}")
    print(f"  {'Separate 2D SIRENs (16 models)':<40} {separate_size:>9,}B {'1.00x':>11} {total_zip/separate_size:>9.2f}x")
    print(f"  {'3D SIREN (1 model, x,y,t)':<40} {video_size:>9,}B {improvement:>10.2f}x {vs_zip:>9.2f}x")

    print(f"\n📋 Analysis:")
    print(f"  Video frames: {n_frames}")
    print(f"  Frame size: {size}x{size}x3")
    print(f"  3D SIREN uses temporal coherence as 'shared roots'")
    print(f"  Single network captures motion + appearance")

    if improvement >= 2.0:
        print(f"\n✅ 3D SIREN achieves {improvement:.2f}x over separate SIRENs!")
        print(f"   Temporal redundancy captured successfully!")
    elif improvement >= 1.5:
        print(f"\n⚠️  Moderate improvement ({improvement:.2f}x) — temporal coherence helps")
    else:
        print(f"\n❌ 3D SIREN doesn't improve much ({improvement:.2f}x)")

    return {
        'n_frames': n_frames,
        'separate_size': separate_size,
        'video_size': video_size,
        'improvement': improvement,
        'vs_zip': vs_zip,
        'total_zip': total_zip,
    }


if __name__ == '__main__':
    results = run_phase11_experiment(verbose=True)
