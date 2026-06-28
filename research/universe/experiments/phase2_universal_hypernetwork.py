# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 2 Experiment: Universal Hypernetwork
============================================
Tests Principle 4 of the Black Hole Universe Hypothesis:
"Universality — works for any data type via type-specific architectures."

HYPOTHESIS:
  The shared roots principle (validated in Phase 1 for images) generalizes
  to other data types: text, audio, and binary. A single hypernetwork
  with type-specific encoders will achieve compression on all types.

ARCHITECTURE:
  - UniversalHypernetwork: shared base + type-specific input encoders
  - Text encoder: positional + character embedding
  - Audio encoder: STFT coordinates
  - Binary encoder: hash-based positional
  - Image encoder: 2D coordinates (from Phase 1)

METHOD:
  1. Generate synthetic datasets: 20 text files, 20 audio clips, 20 binary files
  2. Train universal hypernetwork on each type
  3. Compare with type-specific compressors (gzip for text, etc)
  4. Measure cross-type transfer (does training on text help audio?)

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
from phase1_multi_file_siren import SIREN, SIRENLayer, ModulatedSIREN, get_coordinates


# ============================================================
# Text Compression via Neural Representation
# ============================================================

class TextINR(nn.Module):
    """Represent text as neural implicit function.
    Input: (position, char_class) -> char_value
    """
    def __init__(self, max_len=1024, vocab_size=128, hidden_features=32, hidden_layers=2):
        super().__init__()
        # Position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(1, hidden_features),
            SIRENLayer(hidden_features, hidden_features, omega_0=30.0),
        )
        # Character decoder
        self.decoder = nn.Sequential(
            SIRENLayer(hidden_features, hidden_features, omega_0=30.0),
            nn.Linear(hidden_features, vocab_size),
        )

    def forward(self, positions):
        """positions: (N, 1) float in [0, 1]"""
        x = self.pos_encoder(positions)
        return self.decoder(x)


class MultiFileTextINR(nn.Module):
    """Multi-file text INR with shared roots (FiLM modulation)."""
    def __init__(self, n_files, max_len=1024, vocab_size=128,
                 hidden_features=32, hidden_layers=2, modulation_dim=16):
        super().__init__()
        self.n_files = n_files
        self.modulation_dim = modulation_dim
        self.vocab_size = vocab_size

        # Base network
        self.pos_encoder = nn.Sequential(
            nn.Linear(1, hidden_features),
            SIRENLayer(hidden_features, hidden_features, omega_0=30.0),
        )
        self.decoder = nn.Sequential(
            SIRENLayer(hidden_features, hidden_features, omega_0=30.0),
            nn.Linear(hidden_features, vocab_size),
        )

        # Per-file modulation
        self.modulations = nn.Embedding(n_files, modulation_dim)
        self.film = nn.Linear(modulation_dim, 2 * hidden_features)

    def forward(self, positions, file_idx):
        mod = self.modulations(torch.tensor(file_idx, device=positions.device))
        film = self.film(mod)
        scale, shift = film.chunk(2, dim=-1)

        x = self.pos_encoder(positions)
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return self.decoder(x)


# ============================================================
# Audio Compression via Neural Representation
# ============================================================

class AudioINR(nn.Module):
    """Represent audio as neural implicit function.
    Input: time -> amplitude
    """
    def __init__(self, hidden_features=32, hidden_layers=2):
        super().__init__()
        self.net = SIREN(in_features=1, hidden_features=hidden_features,
                         hidden_layers=hidden_layers, out_features=1, omega_0=30.0)

    def forward(self, t):
        return self.net(t)


class MultiFileAudioINR(nn.Module):
    """Multi-file audio INR with shared roots."""
    def __init__(self, n_files, hidden_features=32, hidden_layers=2, modulation_dim=16):
        super().__init__()
        self.n_files = n_files
        self.base = AudioINR(hidden_features, hidden_layers)
        self.modulations = nn.Embedding(n_files, modulation_dim)
        self.film = nn.Linear(modulation_dim, 2 * hidden_features)

    def forward(self, t, file_idx):
        mod = self.modulations(torch.tensor(file_idx, device=t.device))
        film = self.film(mod)
        scale, shift = film.chunk(2, dim=-1)

        x = t
        for i, layer in enumerate(self.base.net.net):
            if i == 0:
                x = layer(x)
                x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
            else:
                x = layer(x)
        return x


# ============================================================
# Binary Compression via Neural Representation
# ============================================================

class BinaryINR(nn.Module):
    """Represent binary data as neural implicit function.
    Input: byte_position -> byte_value
    """
    def __init__(self, hidden_features=32, hidden_layers=2):
        super().__init__()
        self.net = SIREN(in_features=1, hidden_features=hidden_features,
                         hidden_layers=hidden_layers, out_features=256, omega_0=30.0)

    def forward(self, positions):
        return self.net(positions)


class MultiFileBinaryINR(nn.Module):
    """Multi-file binary INR with shared roots."""
    def __init__(self, n_files, hidden_features=32, hidden_layers=2, modulation_dim=16):
        super().__init__()
        self.n_files = n_files
        self.base = BinaryINR(hidden_features, hidden_layers)
        self.modulations = nn.Embedding(n_files, modulation_dim)
        self.film = nn.Linear(modulation_dim, 2 * hidden_features)

    def forward(self, positions, file_idx):
        mod = self.modulations(torch.tensor(file_idx, device=positions.device))
        film = self.film(mod)
        scale, shift = film.chunk(2, dim=-1)

        x = positions
        for i, layer in enumerate(self.base.net.net):
            if i == 0:
                x = layer(x)
                x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
            else:
                x = layer(x)
        return x


# ============================================================
# Data Generators
# ============================================================

def generate_text_files(n_files=20, length=512, seed=42):
    """Generate synthetic structured text files (like logs)."""
    rng = np.random.default_rng(seed)
    files = []
    templates = [
        "[INFO] User {user} logged in from {ip} at {time}",
        "[WARN] High memory usage: {mem}% on server {srv}",
        "[ERROR] Connection timeout to {host}:{port}",
        "[DEBUG] Processing request {req} took {ms}ms",
    ]

    for i in range(n_files):
        lines = []
        for j in range(length // 60):  # approx chars per line
            template = templates[j % len(templates)]
            line = template.format(
                user=f"user{rng.integers(1, 100)}",
                ip=f"10.0.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
                time=f"2026-06-{rng.integers(1, 30):02d}",
                mem=rng.integers(50, 99),
                srv=f"srv{rng.integers(1, 20)}",
                host=f"host{rng.integers(1, 50)}",
                port=rng.integers(1000, 9999),
                req=rng.integers(10000, 99999),
                ms=rng.integers(1, 500),
            )
            lines.append(line)
        text = "\n".join(lines)
        files.append(text)
    return files


def generate_audio_files(n_files=20, duration=1.0, sr=8000, seed=42):
    """Generate synthetic audio files (tones + harmonics)."""
    rng = np.random.default_rng(seed)
    files = []
    n = int(duration * sr)

    for i in range(n_files):
        t = np.arange(n) / sr
        freq = rng.integers(100, 1000)
        # Fundamental + harmonics
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
        audio += 0.5 * np.sin(2 * np.pi * 2 * freq * t).astype(np.float32)
        audio += 0.3 * np.sin(2 * np.pi * 3 * freq * t).astype(np.float32)
        # Normalize to int16
        audio = (audio / np.abs(audio).max() * 32767).astype(np.int16)
        files.append(audio)
    return files


def generate_binary_files(n_files=20, size=512, seed=42):
    """Generate synthetic structured binary files."""
    rng = np.random.default_rng(seed)
    files = []

    for i in range(n_files):
        # Structured binary: header + repeated pattern + noise
        header = np.array([0xDE, 0xAD, 0xBE, 0xEF], dtype=np.uint8)
        pattern = rng.integers(0, 256, 16, dtype=np.uint8)
        data = np.tile(pattern, size // 16)
        # Add some noise
        noise_idx = rng.integers(0, len(data), size // 10)
        data[noise_idx] = rng.integers(0, 256, len(noise_idx), dtype=np.uint8)
        full = np.concatenate([header, data])[:size]
        files.append(full)
    return files


# ============================================================
# Training Functions
# ============================================================

def train_text_inr(text_files, epochs=100, lr=3e-3, device='cpu', verbose=False):
    """Train multi-file text INR."""
    n_files = len(text_files)
    max_len = max(len(t) for t in text_files)

    # Encode: chars to ASCII, positions normalized
    all_data = []
    for text in text_files:
        chars = np.array([min(ord(c), 127) for c in text], dtype=np.int64)
        all_data.append(chars)

    model = MultiFileTextINR(n_files=n_files, max_len=max_len, vocab_size=128,
                              hidden_features=32, hidden_layers=2, modulation_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        for i, chars in enumerate(all_data):
            n = len(chars)
            positions = torch.linspace(0, 1, n, device=device).unsqueeze(1)
            targets = torch.from_numpy(chars.astype(np.int64)).to(device)

            logits = model(positions, i)
            loss = F.cross_entropy(logits, targets)
            total_loss += loss

        total_loss = total_loss / n_files
        total_loss.backward()
        optimizer.step()

        if verbose and epoch % 20 == 0:
            print(f"  Text Epoch {epoch}: loss={total_loss.item():.4f}")

    return model, total_loss.item()


def train_audio_inr(audio_files, epochs=100, lr=3e-3, device='cpu', verbose=False):
    """Train multi-file audio INR."""
    n_files = len(audio_files)

    model = MultiFileAudioINR(n_files=n_files, hidden_features=32,
                               hidden_layers=2, modulation_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        for i, audio in enumerate(audio_files):
            n = len(audio)
            t = torch.linspace(0, 1, n, device=device).unsqueeze(1)
            # Normalize to [-1, 1]
            target = torch.from_numpy(audio.astype(np.float32) / 32767.0).to(device)

            pred = model(t, i)
            loss = F.mse_loss(pred.squeeze(), target)
            total_loss += loss

        total_loss = total_loss / n_files
        total_loss.backward()
        optimizer.step()

        if verbose and epoch % 20 == 0:
            print(f"  Audio Epoch {epoch}: loss={total_loss.item():.6f}")

    return model, total_loss.item()


def train_binary_inr(binary_files, epochs=100, lr=3e-3, device='cpu', verbose=False):
    """Train multi-file binary INR."""
    n_files = len(binary_files)

    model = MultiFileBinaryINR(n_files=n_files, hidden_features=32,
                                hidden_layers=2, modulation_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        for i, data in enumerate(binary_files):
            n = len(data)
            positions = torch.linspace(0, 1, n, device=device).unsqueeze(1)
            # One-hot target
            target = torch.from_numpy(data.astype(np.int64)).to(device)

            logits = model(positions, i)
            loss = F.cross_entropy(logits, target)
            total_loss += loss

        total_loss = total_loss / n_files
        total_loss.backward()
        optimizer.step()

        if verbose and epoch % 20 == 0:
            print(f"  Binary Epoch {epoch}: loss={total_loss.item():.4f}")

    return model, total_loss.item()


# ============================================================
# Size Measurement
# ============================================================

def measure_model_size_compressed(model):
    """Measure model size in bytes (with zlib compression)."""
    weights_buffer = bytearray()
    for param in model.parameters():
        weights_buffer.extend(param.detach().cpu().numpy().tobytes())
    return len(zlib.compress(bytes(weights_buffer), 9))


# ============================================================
# Main Experiment
# ============================================================

def run_phase2_experiment(verbose=True):
    """Run Phase 2 Universal Hypernetwork experiment."""
    print("=" * 80)
    print("🧪 Phase 2: Universal Hypernetwork (Text + Audio + Binary)")
    print("=" * 80)

    device = 'cpu'

    # === TEXT ===
    print("\n📝 TEXT COMPRESSION")
    print("-" * 40)
    text_files = generate_text_files(n_files=20, length=512)
    text_total = sum(len(t.encode()) for t in text_files)
    text_zip = sum(len(zlib.compress(t.encode(), 9)) for t in text_files)

    t0 = time.time()
    text_model, text_loss = train_text_inr(text_files, epochs=80, device=device, verbose=verbose)
    text_time = time.time() - t0
    text_size = measure_model_size_compressed(text_model)

    print(f"  Files: {len(text_files)}, Total raw: {text_total:,}B")
    print(f"  ZIP: {text_zip:,}B")
    print(f"  BHUH: {text_size:,}B ({text_time:.1f}s)")
    print(f"  vs ZIP: {text_zip/max(text_size,1):.2f}x")

    # === AUDIO ===
    print("\n🔊 AUDIO COMPRESSION")
    print("-" * 40)
    audio_files = generate_audio_files(n_files=20, duration=0.5, sr=8000)
    audio_total = sum(a.nbytes for a in audio_files)
    audio_zip = sum(len(zlib.compress(a.tobytes(), 9)) for a in audio_files)

    t0 = time.time()
    audio_model, audio_loss = train_audio_inr(audio_files, epochs=80, device=device, verbose=verbose)
    audio_time = time.time() - t0
    audio_size = measure_model_size_compressed(audio_model)

    print(f"  Files: {len(audio_files)}, Total raw: {audio_total:,}B")
    print(f"  ZIP: {audio_zip:,}B")
    print(f"  BHUH: {audio_size:,}B ({audio_time:.1f}s)")
    print(f"  vs ZIP: {audio_zip/max(audio_size,1):.2f}x")

    # === BINARY ===
    print("\n💾 BINARY COMPRESSION")
    print("-" * 40)
    binary_files = generate_binary_files(n_files=20, size=512)
    binary_total = sum(b.nbytes for b in binary_files)
    binary_zip = sum(len(zlib.compress(b.tobytes(), 9)) for b in binary_files)

    t0 = time.time()
    binary_model, binary_loss = train_binary_inr(binary_files, epochs=80, device=device, verbose=verbose)
    binary_time = time.time() - t0
    binary_size = measure_model_size_compressed(binary_model)

    print(f"  Files: {len(binary_files)}, Total raw: {binary_total:,}B")
    print(f"  ZIP: {binary_zip:,}B")
    print(f"  BHUH: {binary_size:,}B ({binary_time:.1f}s)")
    print(f"  vs ZIP: {binary_zip/max(binary_size,1):.2f}x")

    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("📊 PHASE 2 SUMMARY")
    print("=" * 80)
    print(f"\n{'Type':<10} {'Raw':>10} {'ZIP':>10} {'BHUH':>10} {'vs ZIP':>10} {'Time':>8}")
    print("-" * 60)
    print(f"{'Text':<10} {text_total:>10,} {text_zip:>10,} {text_size:>10,} {text_zip/max(text_size,1):>9.2f}x {text_time:>7.1f}s")
    print(f"{'Audio':<10} {audio_total:>10,} {audio_zip:>10,} {audio_size:>10,} {audio_zip/max(audio_size,1):>9.2f}x {audio_time:>7.1f}s")
    print(f"{'Binary':<10} {binary_total:>10,} {binary_zip:>10,} {binary_size:>10,} {binary_zip/max(binary_size,1):>9.2f}x {binary_time:>7.1f}s")

    return {
        'text': {'raw': text_total, 'zip': text_zip, 'bhuh': text_size, 'loss': text_loss},
        'audio': {'raw': audio_total, 'zip': audio_zip, 'bhuh': audio_size, 'loss': audio_loss},
        'binary': {'raw': binary_total, 'zip': binary_zip, 'bhuh': binary_size, 'loss': binary_loss},
    }


if __name__ == '__main__':
    results = run_phase2_experiment(verbose=True)
