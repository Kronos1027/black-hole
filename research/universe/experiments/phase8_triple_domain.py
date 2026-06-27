# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 8: Triple Cross-Domain (Images + Audio + Text)
=====================================================
Extends Phase 6's cross-domain discovery to THREE domains simultaneously.

Phase 6 showed images+audio share roots (6.88x).
Can we add TEXT to the same shared model?

HYPOTHESIS:
  A single cross-domain model handling images (2D), audio (1D), AND
  text (1D sequential) will still beat separate models, though with
  more overhead than 2-domain.

ARCHITECTURE:
  - Input adapters: 2D (images), 1D-time (audio), 1D-position (text)
  - Shared base SIREN
  - Output heads: 3ch (images), 1ch (audio), 128-class (text)
  - FiLM modulation per file

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
from phase3_program_synthesis import generate_log_files


class TripleDomainSIREN(nn.Module):
    """SIREN handling images (2D), audio (1D), and text (1D) simultaneously."""
    def __init__(self, n_files, hidden_features=32, hidden_layers=2, modulation_dim=16):
        super().__init__()
        self.n_files = n_files
        self.hidden_features = hidden_features

        # Input adapters
        self.adapter_2d = nn.Linear(2, hidden_features)   # images
        self.adapter_1d = nn.Linear(1, hidden_features)   # audio/text

        # Shared base layers
        self.base_layers = nn.ModuleList([
            SIRENLayer(hidden_features, hidden_features, omega_0=30.0)
            for _ in range(hidden_layers)
        ])

        # Output heads
        self.head_image = nn.Linear(hidden_features, 3)    # RGB
        self.head_audio = nn.Linear(hidden_features, 1)    # amplitude
        self.head_text = nn.Linear(hidden_features, 128)   # ASCII chars

        # Per-file modulation
        self.modulations = nn.Embedding(n_files, modulation_dim)
        self.film = nn.Linear(modulation_dim, 2 * hidden_features)

        # File type registry
        self.file_types = {}

    def register_file(self, idx, file_type):
        self.file_types[idx] = file_type

    def forward(self, coords, file_idx):
        mod = self.modulations(torch.tensor(file_idx, device=coords.device))
        film = self.film(mod)
        scale, shift = film.chunk(2, dim=-1)

        ft = self.file_types.get(file_idx, 'image')

        # Input adapter
        if ft == 'image':
            x = self.adapter_2d(coords)
        else:
            x = self.adapter_1d(coords)

        x = x * (1 + scale.unsqueeze(0) * 0.5) + shift.unsqueeze(0) * 0.5

        for layer in self.base_layers:
            x = layer(x)

        if ft == 'image':
            return self.head_image(x)
        elif ft == 'audio':
            return self.head_audio(x)
        else:
            return self.head_text(x)


def train_triple_domain(images, audio_files, text_files, epochs=150, device='cpu', verbose=False):
    """Train triple-domain model on images + audio + text."""
    n_img = len(images)
    n_aud = len(audio_files)
    n_txt = len(text_files)
    n_total = n_img + n_aud + n_txt

    model = TripleDomainSIREN(n_files=n_total, hidden_features=32,
                               hidden_layers=2, modulation_dim=16).to(device)

    # Register file types
    for i in range(n_img):
        model.register_file(i, 'image')
    for i in range(n_aud):
        model.register_file(n_img + i, 'audio')
    for i in range(n_txt):
        model.register_file(n_img + n_aud + i, 'text')

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Prepare data
    img_data = []
    for img in images:
        size = img.shape[0]
        coords = get_coordinates(size, device)
        pixels = torch.from_numpy(img.astype(np.float32) / 255.0).reshape(-1, 3).to(device)
        img_data.append((coords, pixels))

    aud_data = []
    for audio in audio_files:
        n = len(audio)
        t = torch.linspace(0, 1, n, device=device).unsqueeze(1)
        target = torch.from_numpy(audio.astype(np.float32) / 32767.0).to(device)
        aud_data.append((t, target))

    txt_data = []
    for text in text_files:
        chars = np.array([min(ord(c), 127) for c in text[:256]], dtype=np.int64)  # limit to 256 chars
        n = len(chars)
        pos = torch.linspace(0, 1, n, device=device).unsqueeze(1)
        targets = torch.from_numpy(chars).to(device)
        txt_data.append((pos, targets))

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0

        # Images
        for i, (coords, pixels) in enumerate(img_data):
            pred = model(coords, i)
            total_loss += F.mse_loss(pred, pixels)

        # Audio
        for j, (t, target) in enumerate(aud_data):
            pred = model(t, n_img + j)
            total_loss += F.mse_loss(pred.squeeze(), target)

        # Text (cross-entropy)
        for k, (pos, targets) in enumerate(txt_data):
            logits = model(pos, n_img + n_aud + k)
            total_loss += F.cross_entropy(logits, targets) * 0.1  # weight down (larger loss)

        total_loss = total_loss / n_total
        total_loss.backward()
        optimizer.step()

        if verbose and epoch % 30 == 0:
            print(f"  Triple-domain Epoch {epoch}: avg_loss={total_loss.item():.6f}")

    return model, total_loss.item()


def run_phase8_experiment(verbose=True):
    """Run Phase 8 Triple Cross-Domain experiment."""
    print("=" * 80)
    print("🧪 Phase 8: Triple Cross-Domain (Images + Audio + Text)")
    print("=" * 80)

    device = 'cpu'

    # Generate data
    print("\n📦 Generating triple-domain corpus...")
    from phase1_multi_file_siren import generate_satellite_images
    images = generate_satellite_images(n_images=5, size=64, seed=42)
    audio_files = generate_audio_files(n_files=5, duration=0.25, sr=8000, seed=42)
    text_files = generate_log_files(n_files=5, n_lines=10, seed=42)

    n_img, n_aud, n_txt = len(images), len(audio_files), len(text_files)
    n_total = n_img + n_aud + n_txt
    print(f"  {n_img} images + {n_aud} audio + {n_txt} text = {n_total} files")

    # Baseline: separate models per domain
    print("\n🔵 Baseline: Separate models per domain...")
    from phase1_multi_file_siren import train_single_siren, train_multi_file_siren
    from phase2_universal_hypernetwork import train_audio_inr

    # Images
    img_baseline = 0
    for img in images:
        m, _ = train_single_siren(img, epochs=60, device=device, verbose=False)
        img_baseline += measure_model_size_compressed(m)

    # Audio
    aud_model, _ = train_audio_inr(audio_files, epochs=80, device=device, verbose=False)
    aud_baseline = measure_model_size_compressed(aud_model)

    # Text (zlib baseline)
    txt_baseline = sum(len(zlib.compress(t.encode(), 9)) for t in text_files)

    baseline_total = img_baseline + aud_baseline + txt_baseline
    print(f"  Images: {img_baseline:,}B, Audio: {aud_baseline:,}B, Text: {txt_baseline:,}B")
    print(f"  Total: {baseline_total:,}B")

    # BHUH: Triple-domain shared model
    print("\n🌌 BHUH: Triple-domain shared model...")
    t0 = time.time()
    triple_model, triple_loss = train_triple_domain(
        images, audio_files, text_files, epochs=120, device=device, verbose=verbose
    )
    triple_time = time.time() - t0
    triple_size = measure_model_size_compressed(triple_model)

    # Text needs separate storage (neural can't compress it well)
    # So total = model + text zlib
    triple_total = triple_size + txt_baseline

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 8 RESULTS — TRIPLE CROSS-DOMAIN")
    print(f"{'='*80}")
    print(f"\n  {'Method':<45} {'Size':>10} {'vs Baseline':>12}")
    print(f"  {'-'*70}")
    print(f"  {'Separate (img SIREN + aud SIREN + txt zlib)':<45} {baseline_total:>9,}B {'1.00x':>11}")
    print(f"  {'Triple-domain BHUH (model + txt fallback)':<45} {triple_total:>9,}B {baseline_total/triple_total:>10.2f}x")
    print(f"  {'  - Shared model (img+aud+txt neural)':<45} {triple_size:>9,}B")
    print(f"  {'  - Text fallback (zlib)':<45} {txt_baseline:>9,}B")

    # Compare with Phase 6 (2-domain)
    print(f"\n📋 Comparison with Phase 6 (2-domain):")
    print(f"  Phase 6 (img+aud):  14,424B for 20 files (6.88x vs separate)")
    print(f"  Phase 8 (img+aud+txt): {triple_total:,}B for {n_total} files ({baseline_total/triple_total:.2f}x vs separate)")

    if triple_total < baseline_total:
        print(f"\n✅ Triple-domain sharing works! {baseline_total/triple_total:.2f}x improvement")
    else:
        print(f"\n⚠️  Text overhead reduces benefit, but images+audio still share")

    return {
        'baseline': baseline_total,
        'triple_total': triple_total,
        'model_size': triple_size,
        'text_fallback': txt_baseline,
        'improvement': baseline_total / triple_total,
    }


if __name__ == '__main__':
    results = run_phase8_experiment(verbose=True)
