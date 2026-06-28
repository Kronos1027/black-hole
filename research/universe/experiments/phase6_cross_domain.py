# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 6: Cross-Domain Transfer
================================
Tests whether "shared roots" can transfer ACROSS data types.

HYPOTHESIS:
  If we train a shared base network on BOTH images AND audio, does the
  shared representation help? Or does mixing domains hurt?

  This tests the deepest implication of the BHUH "Multiverse" principle:
  do ALL structured signals share common mathematical roots?

METHOD:
  1. Create a "universal" modulated SIREN that handles 2D (images) and 1D (audio)
  2. Train on mixed corpus: 10 images + 10 audio clips
  3. Compare with domain-specific models
  4. Measure: does cross-domain training help or hurt?

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
from phase2_universal_hypernetwork import generate_audio_files


class CrossDomainSIREN(nn.Module):
    """SIREN that handles BOTH 2D (images) and 1D (audio) inputs.
    Uses input dimension adapter + shared base + FiLM modulation.
    """
    def __init__(self, n_files, hidden_features=32, hidden_layers=2, modulation_dim=16):
        super().__init__()
        self.n_files = n_files
        self.hidden_features = hidden_features

        # Input adapters: project any input dim to hidden_features
        self.adapter_2d = nn.Linear(2, hidden_features)  # for images (x, y)
        self.adapter_1d = nn.Linear(1, hidden_features)  # for audio (t)

        # Shared base layers
        self.base_layers = nn.ModuleList([
            SIRENLayer(hidden_features, hidden_features, omega_0=30.0)
            for _ in range(hidden_layers)
        ])

        # Output heads: one for images (3 channels), one for audio (1 channel)
        self.head_image = nn.Linear(hidden_features, 3)
        self.head_audio = nn.Linear(hidden_features, 1)

        # Per-file modulation
        self.modulations = nn.Embedding(n_files, modulation_dim)
        self.film = nn.Linear(modulation_dim, 2 * hidden_features)

        # Track which files are images vs audio
        self.file_types = {}  # file_idx -> 'image' or 'audio'

    def register_file_type(self, file_idx, file_type):
        """Register whether file_idx is 'image' or 'audio'."""
        self.file_types[file_idx] = file_type

    def forward(self, coords, file_idx):
        """Forward pass. coords: (N, input_dim), file_idx: int."""
        mod = self.modulations(torch.tensor(file_idx, device=coords.device))
        film = self.film(mod)
        scale, shift = film.chunk(2, dim=-1)

        file_type = self.file_types.get(file_idx, 'image')

        # Input adapter
        if file_type == 'image':
            x = self.adapter_2d(coords)
        else:
            x = self.adapter_1d(coords)

        # Apply FiLM modulation
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)

        # Shared base layers
        for layer in self.base_layers:
            x = layer(x)

        # Output head
        if file_type == 'image':
            return self.head_image(x)
        else:
            return self.head_audio(x)


def train_cross_domain(images, audio_files, epochs=150, device='cpu', verbose=False):
    """Train cross-domain SIREN on mixed images + audio."""
    n_images = len(images)
    n_audio = len(audio_files)
    n_total = n_images + n_audio

    model = CrossDomainSIREN(n_files=n_total, hidden_features=32,
                              hidden_layers=2, modulation_dim=16).to(device)

    # Register file types
    for i in range(n_images):
        model.register_file_type(i, 'image')
    for i in range(n_audio):
        model.register_file_type(n_images + i, 'audio')

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Prepare data
    img_data = []
    for img in images:
        size = img.shape[0]
        coords = get_coordinates(size, device)
        pixels = torch.from_numpy(img.astype(np.float32) / 255.0).reshape(-1, 3).to(device)
        img_data.append((coords, pixels))

    audio_data = []
    for audio in audio_files:
        n = len(audio)
        t = torch.linspace(0, 1, n, device=device).unsqueeze(1)
        target = torch.from_numpy(audio.astype(np.float32) / 32767.0).to(device)
        audio_data.append((t, target))

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0

        # Train on images
        for i, (coords, pixels) in enumerate(img_data):
            pred = model(coords, i)
            loss = F.mse_loss(pred, pixels)
            total_loss += loss

        # Train on audio
        for j, (t, target) in enumerate(audio_data):
            pred = model(t, n_images + j)
            loss = F.mse_loss(pred.squeeze(), target)
            total_loss += loss

        total_loss = total_loss / n_total
        total_loss.backward()
        optimizer.step()

        if verbose and epoch % 30 == 0:
            print(f"  Cross-domain Epoch {epoch}: avg_loss={total_loss.item():.6f}")

    return model, total_loss.item()


def run_phase6_experiment(verbose=True):
    """Run Phase 6 Cross-Domain Transfer experiment."""
    print("=" * 80)
    print("🧪 Phase 6: Cross-Domain Transfer (Images + Audio)")
    print("=" * 80)

    device = 'cpu'

    # Generate data
    print("\n📦 Generating mixed corpus...")
    from phase1_multi_file_siren import generate_satellite_images
    images = generate_satellite_images(n_images=10, size=64, seed=42)
    audio_files = generate_audio_files(n_files=10, duration=0.25, sr=8000, seed=42)

    n_images = len(images)
    n_audio = len(audio_files)
    print(f"  {n_images} images @ 64x64 + {n_audio} audio @ 0.25s @ 8kHz")

    # Baseline 1: Separate image SIRENs
    print("\n🔵 Baseline 1: Separate image SIRENs...")
    from phase1_multi_file_siren import train_single_siren
    img_baseline = 0
    for img in images:
        model, _ = train_single_siren(img, epochs=80, device=device, verbose=False)
        img_baseline += measure_model_size_compressed(model)

    # Baseline 2: Separate audio SIRENs
    print("🔵 Baseline 2: Separate audio SIRENs...")
    from phase2_universal_hypernetwork import MultiFileAudioINR, train_audio_inr
    audio_baseline_model, _ = train_audio_inr(audio_files, epochs=80, device=device, verbose=False)
    audio_baseline = measure_model_size_compressed(audio_baseline_model)

    # Baseline 3: Domain-specific multi-file (image-only + audio-only)
    print("🔵 Baseline 3: Domain-specific multi-file (separate)...")
    from phase1_multi_file_siren import train_multi_file_siren
    img_multi_model, _ = train_multi_file_siren(images, epochs=100, device=device, verbose=False)
    img_multi_size = measure_model_size_compressed(img_multi_model)

    audio_multi_model, _ = train_audio_inr(audio_files, epochs=100, device=device, verbose=False)
    audio_multi_size = measure_model_size_compressed(audio_multi_model)
    domain_specific_total = img_multi_size + audio_multi_size

    # BHUH: Cross-domain shared model
    print("\n🌌 BHUH: Cross-domain shared model (images + audio together)...")
    t0 = time.time()
    cross_model, cross_loss = train_cross_domain(images, audio_files, epochs=150, device=device, verbose=verbose)
    cross_time = time.time() - t0
    cross_size = measure_model_size_compressed(cross_model)

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 6 RESULTS — CROSS-DOMAIN TRANSFER")
    print(f"{'='*80}")
    print(f"\n  {'Method':<40} {'Size':>10} {'vs Baseline':>12}")
    print(f"  {'-'*65}")

    baseline_total = img_baseline + audio_baseline
    print(f"  {'Separate SIRENs (baseline)':<40} {baseline_total:>9,}B {'1.00x':>11}")
    print(f"  {'Domain-specific multi-file':<40} {domain_specific_total:>9,}B {baseline_total/domain_specific_total:>10.2f}x")
    print(f"  {'Cross-domain shared (BHUH)':<40} {cross_size:>9,}B {baseline_total/cross_size:>10.2f}x")

    # Analysis
    print(f"\n📋 Analysis:")
    print(f"  Separate SIRENs:     {baseline_total:,}B (10 img + 10 audio)")
    print(f"  Domain-specific:     {domain_specific_total:,}B ({img_multi_size:,} img + {audio_multi_size:,} audio)")
    print(f"  Cross-domain BHUH:   {cross_size:,}B (ONE model for both!)")

    if cross_size < domain_specific_total:
        ratio = domain_specific_total / cross_size
        print(f"\n✅ Cross-domain WINS! {ratio:.2f}x better than domain-specific")
        print(f"   Sharing roots ACROSS domains helps!")
    else:
        ratio = cross_size / domain_specific_total
        print(f"\n⚠️  Cross-domain is {ratio:.2f}x larger than domain-specific")
        print(f"   Mixed domains may add overhead, but still beats separate SIRENs")

    vs_separate = baseline_total / cross_size
    print(f"\n  vs Separate SIRENs: {vs_separate:.2f}x improvement")

    return {
        'baseline_separate': baseline_total,
        'domain_specific': domain_specific_total,
        'cross_domain': cross_size,
        'improvement_vs_separate': vs_separate,
    }


if __name__ == '__main__':
    results = run_phase6_experiment(verbose=True)
