# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 26: Time-Series Compression (Sensor/Financial Data)
==========================================================
Tests SIREN for compressing 1D time-series data.

CONCEPT:
  Time-series (sensor readings, stock prices, IoT data) are 1D signals.
  SIREN f(t) → value can represent them. For correlated series
  (e.g., multiple temperature sensors), Multi-File SIREN shares roots.

HYPOTHESIS:
  SIREN will compress smooth time-series 3-10x better than ZIP,
  and Multi-File SIREN will give additional 2-5x for correlated series.

METHOD:
  1. Generate 20 correlated time-series (temperature sensors)
  2. Baseline: ZIP
  3. Separate SIRENs: 20 models
  4. Multi-File SIREN: 1 shared model + 20 modulations
  5. Compare

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, SIRENLayer, measure_model_size_compressed


class TimeSeriesSIREN(nn.Module):
    """1D SIREN: f(t) → value."""
    def __init__(self, hidden_features=16, hidden_layers=2, omega_0=30.0):
        super().__init__()
        layers = [SIRENLayer(1, hidden_features, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers):
            layers.append(SIRENLayer(hidden_features, hidden_features, omega_0=omega_0))
        layers.append(nn.Linear(hidden_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


class MultiTimeSeriesSIREN(nn.Module):
    """Multi-file 1D SIREN with shared roots (FiLM modulation)."""
    def __init__(self, n_files, hidden_features=16, hidden_layers=2, mod_dim=8):
        super().__init__()
        self.n_files = n_files
        self.base = TimeSeriesSIREN(hidden_features, hidden_layers)
        self.modulations = nn.Embedding(n_files, mod_dim)
        self.film = nn.Linear(mod_dim, 2 * hidden_features)

    def forward(self, t, file_idx):
        mod = self.modulations(torch.tensor(file_idx, device=t.device))
        film = self.film(mod)
        scale, shift = film.chunk(2, dim=-1)

        x = t
        for i, layer in enumerate(self.base.net):
            if i == 0:
                x = layer(x)
                x = x * (1 + scale.unsqueeze(0) * 0.5) + shift.unsqueeze(0) * 0.5
            elif i < len(self.base.net) - 1:
                x = layer(x)
            else:
                x = layer(x)
        return x


def generate_sensor_data(n_sensors=20, n_points=256, seed=42):
    """Generate correlated temperature sensor data."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=np.float32) / n_points

    series = []
    # Common base pattern (daily temperature cycle)
    base_freq = rng.uniform(2, 5)
    base_amp = rng.uniform(20, 40)

    for i in range(n_sensors):
        # Each sensor: base + per-sensor variation
        amp = base_amp + rng.uniform(-5, 5)
        freq = base_freq + rng.uniform(-0.5, 0.5)
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, 1, n_points)

        signal = amp * np.sin(2 * np.pi * freq * t + phase) + \
                 10 * np.sin(2 * np.pi * 3 * freq * t + phase * 0.5) + \
                 noise
        # Normalize to int16
        signal = (signal / np.abs(signal).max() * 32767).astype(np.int16)
        series.append(signal)

    return series


def train_timeseries_siren(data, epochs=80, device='cpu', verbose=False):
    """Train multi-file time-series SIREN."""
    n = len(data)
    model = MultiTimeSeriesSIREN(n_files=n, hidden_features=16, hidden_layers=2, mod_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        for i, series in enumerate(data):
            t = torch.linspace(0, 1, len(series), device=device).unsqueeze(1)
            target = torch.from_numpy(series.astype(np.float32) / 32767.0).to(device)
            pred = model(t, i)
            total_loss += F.mse_loss(pred.squeeze(), target)
        total_loss = total_loss / n
        total_loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"  TimeSeries Epoch {epoch}: loss={total_loss.item():.6f}")

    return model, total_loss.item()


def run_phase26_experiment(verbose=True):
    """Run Phase 26 Time-Series experiment."""
    print("=" * 80)
    print("🧪 Phase 26: Time-Series Compression (Sensor Data)")
    print("=" * 80)

    device = 'cpu'
    n_sensors = 20
    n_points = 256

    print(f"\n📦 Generating {n_sensors} correlated sensor series ({n_points} points each)...")
    series = generate_sensor_data(n_sensors, n_points, seed=42)
    total_raw = sum(s.nbytes for s in series)
    total_zip = sum(len(zlib.compress(s.tobytes(), 9)) for s in series)
    print(f"  Raw: {total_raw:,}B, ZIP: {total_zip:,}B")

    # Separate SIRENs
    print(f"\n🔵 Baseline: {n_sensors} separate SIRENs...")
    separate_total = 0
    for s in series:
        model = TimeSeriesSIREN(hidden_features=16, hidden_layers=2).to(device)
        t = torch.linspace(0, 1, n_points, device=device).unsqueeze(1)
        target = torch.from_numpy(s.astype(np.float32) / 32767.0).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(60):
            opt.zero_grad()
            loss = F.mse_loss(model(t).squeeze(), target)
            loss.backward()
            opt.step()
        separate_total += measure_model_size_compressed(model)
    print(f"  Total: {separate_total:,}B")

    # Multi-File SIREN
    print(f"\n🌌 BHUH: Multi-File SIREN (shared roots)...")
    t0 = time.time()
    multi_model, multi_loss = train_timeseries_siren(series, epochs=100, device=device, verbose=verbose)
    multi_time = time.time() - t0
    multi_size = measure_model_size_compressed(multi_model)
    print(f"  Total: {multi_size:,}B in {multi_time:.1f}s")

    # Results
    improvement = separate_total / max(multi_size, 1)
    vs_zip = total_zip / max(multi_size, 1)

    print(f"\n{'='*80}")
    print("📊 PHASE 26 RESULTS — TIME-SERIES COMPRESSION")
    print(f"{'='*80}")
    print(f"\n  {'Method':<40} {'Size':>10} {'vs Separate':>12} {'vs ZIP':>10}")
    print(f"  {'-'*75}")
    print(f"  {'ZIP (zlib-9)':<40} {total_zip:>9,}B {'-':>11} {'1.00x':>9}")
    print(f"  {'Separate SIRENs (20 models)':<40} {separate_total:>9,}B {'1.00x':>11} {total_zip/separate_total:>9.2f}x")
    print(f"  {'Multi-File SIREN (shared)':<40} {multi_size:>9,}B {improvement:>10.2f}x {vs_zip:>9.2f}x")

    if improvement >= 2.0:
        print(f"\n  ✅ Shared roots work for time-series! {improvement:.2f}x improvement")
    elif improvement >= 1.5:
        print(f"\n  ⚠️  Moderate improvement ({improvement:.2f}x)")
    else:
        print(f"\n  ❌ Time-series doesn't benefit from shared roots ({improvement:.2f}x)")

    print(f"\n  📋 Applications:")
    print(f"  - IoT sensor networks (temperature, pressure, humidity)")
    print(f"  - Financial data (stock prices, crypto)")
    print(f"  - Industrial monitoring (vibration, current)")
    print(f"  - Health monitoring (ECG, EEG)")

    return {
        'separate': separate_total,
        'multi': multi_size,
        'improvement': improvement,
        'vs_zip': vs_zip,
        'zip': total_zip,
    }


if __name__ == '__main__':
    results = run_phase26_experiment(verbose=True)
