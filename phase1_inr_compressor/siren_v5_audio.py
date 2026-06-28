# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

"""
siren_v5_audio.py — v5.17 Audio compression via STFT spectrogram INR
=====================================================================
Compresses audio by:
  1. Convert audio → STFT spectrogram (2D magnitude + phase)
  2. Compress spectrogram with SIREN f(x, y) → magnitude
  3. Phase is residual-coded (zlib)
  4. Reconstruct via inverse STFT

The spectrogram is a smooth 2D surface (frequency × time) — exactly
what SIREN is good at. Audio with harmonic structure (music, speech)
has very smooth spectrograms.

Recipe format (.blka):
  [magic 'BLKA'][version][bits][sample_rate][n_samples]
  [n_freq][n_frames][siren_packed + meta]
  [phase_compressed][sha]

Usage:
    comp = AudioCompressor(hidden_features=64, hidden_layers=3)
    res = comp.compress(audio_samples, sample_rate=44100)
    recovered = comp.decompress(res['recipe_bytes'])
"""
from __future__ import annotations
import os
import sys
import io
import time
import zlib
import struct
import hashlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siren_v5_torch import SineLayer, SIREN, quantize_int8, dequantize_int8


MAGIC_AUDIO = b'BLKA'  # BLK Audio
VERSION_AUDIO = 1


class AudioCompressor:
    """
    Audio compression via STFT spectrogram + SIREN.

    Pipeline:
      1. STFT: audio → spectrogram (n_freq × n_frames) complex
      2. Split: magnitude (smooth) + phase (residual)
      3. SIREN: f(freq, time) → magnitude (normalized to [-1,1])
      4. Quantize SIREN weights to INT8
      5. Phase: quantize to 8-bit + zlib
      6. Pack: weights + phase + metadata + SHA-256
    """

    def __init__(self, hidden_features=64, hidden_layers=3, omega_0=30.0,
                 n_fft=512, hop_length=128,
                 device: str | None = None):
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = float(omega_0)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _make_model(self):
        return SIREN(in_features=2, hidden_features=self.hidden_features,
                     hidden_layers=self.hidden_layers, out_features=1,
                     omega_0=self.omega_0).to(self.device)

    def _make_coords(self, H, W):
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij',
        )
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    def _stft(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute STFT, return (magnitude, phase) as float32."""
        from scipy.signal import stft
        # Use scipy STFT
        f, t, Zxx = stft(audio, fs=self.sample_rate,
                          nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
        magnitude = np.abs(Zxx).astype(np.float32)
        phase = np.angle(Zxx).astype(np.float32)
        return magnitude, phase, f, t

    def _istft(self, magnitude: np.ndarray, phase: np.ndarray, n_samples: int) -> np.ndarray:
        """Inverse STFT."""
        from scipy.signal import istft
        Zxx = magnitude * np.exp(1j * phase)
        _, audio = istft(Zxx, fs=self.sample_rate,
                         nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
        # Trim or pad to original length
        if len(audio) > n_samples:
            audio = audio[:n_samples]
        elif len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
        return audio

    def compress(self, audio: np.ndarray, sample_rate: int = 44100,
                 epochs: int = 500, lr: float = 2e-3,
                 bits: int = 8, batch_size: int = 8192,
                 use_amp: bool = True, patience: int = 3,
                 verbose: bool = False) -> dict:
        """Compress audio (1D float32 array) to .blka recipe."""
        assert audio.ndim == 1, "Expected 1D audio array"
        self.sample_rate = sample_rate
        original_bytes = audio.astype(np.float32).tobytes()
        t0 = time.time()

        # 1. STFT
        magnitude, phase, freqs, times = self._stft(audio)
        n_freq, n_frames = magnitude.shape
        if verbose:
            print(f"  STFT: {n_freq} freq bins × {n_frames} frames")

        # 2. Normalize magnitude to [-1, 1] for SIREN
        mag_max = float(magnitude.max()) if magnitude.max() > 0 else 1.0
        mag_norm = magnitude / mag_max * 2.0 - 1.0  # [-1, 1]

        # 3. Train SIREN on magnitude spectrogram
        coords = self._make_coords(n_freq, n_frames)
        values = torch.from_numpy(mag_norm.reshape(-1, 1)).to(self.device)
        N = coords.shape[0]

        model = self._make_model()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        warmup = max(1, epochs // 20)

        amp_dtype = torch.bfloat16 if (use_amp and self.device.type == 'cpu') else (
            torch.float16 if use_amp else None
        )

        model.train()
        history = []
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            if epoch < warmup:
                for g in opt.param_groups:
                    g['lr'] = lr * (epoch + 1) / warmup

            if batch_size < N:
                idx = torch.randint(0, N, (batch_size,), device=self.device)
                xb = coords[idx]; yb = values[idx]
            else:
                xb, yb = coords, values

            if amp_dtype is not None:
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                    pred = model(xb)
                    loss = torch.nn.functional.mse_loss(pred, yb)
            else:
                pred = model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            if epoch >= warmup:
                sched.step()

            cur_loss = float(loss.item())
            if epoch % 50 == 0 or epoch == epochs - 1:
                history.append(cur_loss)
                if patience > 0 and epoch >= warmup:
                    if cur_loss < best_loss - 1e-6:
                        best_loss = cur_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"  early stopping at epoch {epoch}")
                            break

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  audio epoch {epoch}/{epochs}  loss={cur_loss:.6e}")

        train_time = time.time() - t0

        # 4. Quantize weights
        weights_np = model.state_to_numpy()
        packed, packed_meta = quantize_int8(weights_np)
        q_weights = dequantize_int8(packed, packed_meta)
        model.load_from_numpy(q_weights)
        model.eval()

        # 5. Inference → predicted magnitude
        with torch.inference_mode():
            pred_mag = model(coords).cpu().numpy()
        pred_mag = np.clip((pred_mag + 1.0) / 2.0 * mag_max, 0, mag_max).reshape(n_freq, n_frames)

        # 6. Phase encoding: quantize to 8-bit [-pi, pi] → [0, 255]
        phase_uint8 = np.round((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        phase_compressed = zlib.compress(phase_uint8.tobytes(), 9)

        # 7. Magnitude residual: (original - predicted) as uint8
        # Scale magnitude to [0, 255] for residual
        mag_scaled = np.round(magnitude / mag_max * 255).astype(np.uint8)
        pred_scaled = np.round(pred_mag / mag_max * 255).astype(np.uint8)
        mag_residual = ((mag_scaled.astype(np.int16) - pred_scaled.astype(np.int16)) % 256).astype(np.uint8)
        mag_residual_compressed = zlib.compress(mag_residual.tobytes(), 9)

        # 8. SHA-256
        sha = hashlib.sha256(original_bytes).digest()

        # 9. Pack recipe
        recipe = self._pack_recipe(
            bits, packed, packed_meta,
            sample_rate, len(audio),
            n_freq, n_frames, mag_max,
            phase_compressed, mag_residual_compressed, sha
        )

        # 10. Stats
        a = np.frombuffer(original_bytes, dtype=np.uint8)
        # Compare reconstructed audio
        recon_mag = pred_mag + (mag_residual.astype(np.float32) / 255 * mag_max - pred_mag * 0)  # approximate
        # Actually, just use mag_scaled and pred_scaled
        recon_mag_scaled = ((pred_scaled.astype(np.int16) + mag_residual.astype(np.int16)) % 256).astype(np.uint8)
        recon_mag = recon_mag_scaled.astype(np.float32) / 255 * mag_max
        recon_phase = (phase_uint8.astype(np.float32) / 255 * 2 * np.pi - np.pi)
        recon_audio = self._istft(recon_mag, recon_phase, len(audio))
        bit_acc = float(np.mean(np.unpackbits(a) == np.unpackbits(
            np.frombuffer(recon_audio.astype(np.float32).tobytes(), dtype=np.uint8)
        ))) * 100 if len(recon_audio) == len(audio) else 0.0

        # SNR
        if len(recon_audio) == len(audio):
            noise = audio - recon_audio
            signal_power = np.mean(audio ** 2)
            noise_power = np.mean(noise ** 2) + 1e-10
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 0.0

        return {
            'recipe_bytes': recipe,
            'original_size': len(original_bytes),
            'recipe_size': len(recipe),
            'weights_packed_size': len(packed),
            'phase_compressed_size': len(phase_compressed),
            'mag_residual_size': len(mag_residual_compressed),
            'train_time_s': train_time,
            'snr_db': float(snr),
            'bit_accuracy': bit_acc,
            'sha256': sha.hex(),
            'n_freq': n_freq,
            'n_frames': n_frames,
        }

    def _pack_recipe(self, bits, packed, packed_meta,
                     sample_rate, n_samples,
                     n_freq, n_frames, mag_max,
                     phase_compressed, mag_residual_compressed, sha):
        out = bytearray()
        out += MAGIC_AUDIO
        out += struct.pack('<B', VERSION_AUDIO)
        out += struct.pack('<B', bits)
        out += struct.pack('<I', sample_rate)
        out += struct.pack('<Q', n_samples)
        out += struct.pack('<H', n_freq)
        out += struct.pack('<H', n_frames)
        out += struct.pack('<f', float(mag_max))
        out += struct.pack('<H', self.n_fft)
        out += struct.pack('<H', self.hop_length)
        out += struct.pack('<H', self.hidden_features)
        out += struct.pack('<B', self.hidden_layers)
        out += struct.pack('<f', self.omega_0)
        # SIREN weights
        out += struct.pack('<I', len(packed))
        out += packed
        out += struct.pack('<H', len(packed_meta))
        for entry in packed_meta:
            n_bytes, shape, scale, name = entry
            name_b = name.encode('utf-8')
            out += struct.pack('<B', len(name_b)); out += name_b
            out += struct.pack('<I', n_bytes)
            out += struct.pack('<B', len(shape))
            for d in shape: out += struct.pack('<i', int(d))
            out += struct.pack('<d', float(scale))
            out += struct.pack('<I', 0)
        # Phase
        out += struct.pack('<Q', len(phase_compressed))
        out += phase_compressed
        # Magnitude residual
        out += struct.pack('<Q', len(mag_residual_compressed))
        out += mag_residual_compressed
        # SHA
        out += sha
        return bytes(out)

    @staticmethod
    def decompress(recipe_bytes: bytes, device: str | None = None) -> tuple[np.ndarray, dict]:
        buf = recipe_bytes
        off = 0
        if buf[:4] != MAGIC_AUDIO:
            raise ValueError(f"bad magic: {buf[:4]!r}")
        off += 4
        version = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        assert version == VERSION_AUDIO
        bits = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        sample_rate = struct.unpack('<I', buf[off:off+4])[0]; off += 4
        n_samples = struct.unpack('<Q', buf[off:off+8])[0]; off += 8
        n_freq = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        n_frames = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        mag_max = struct.unpack('<f', buf[off:off+4])[0]; off += 4
        n_fft = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hop_length = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        hidden_l = struct.unpack('<B', buf[off:off+1])[0]; off += 1
        omega = struct.unpack('<f', buf[off:off+4])[0]; off += 4

        packed_size = struct.unpack('<I', buf[off:off+4])[0]; off += 4
        packed = buf[off:off+packed_size]; off += packed_size
        n_meta = struct.unpack('<H', buf[off:off+2])[0]; off += 2
        meta = []
        for _ in range(n_meta):
            name_len = struct.unpack('<B', buf[off:off+1])[0]; off += 1
            name = buf[off:off+name_len].decode('utf-8'); off += name_len
            n_bytes = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            ndim = struct.unpack('<B', buf[off:off+1])[0]; off += 1
            shape = tuple(struct.unpack('<' + 'i'*ndim, buf[off:off+4*ndim])); off += 4*ndim
            scale = struct.unpack('<d', buf[off:off+8])[0]; off += 8
            _ = struct.unpack('<I', buf[off:off+4])[0]; off += 4
            meta.append((n_bytes, shape, scale, name))

        phase_size = struct.unpack('<Q', buf[off:off+8])[0]; off += 8
        phase_compressed = buf[off:off+phase_size]; off += phase_size

        resid_size = struct.unpack('<Q', buf[off:off+8])[0]; off += 8
        mag_residual_compressed = buf[off:off+resid_size]; off += resid_size

        sha_expected = buf[off:off+32]; off += 32

        # Dequantize weights
        weights = dequantize_int8(packed, meta)
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = SIREN(in_features=2, hidden_features=hidden,
                       hidden_layers=hidden_l, out_features=1,
                       omega_0=omega).to(dev)
        model.load_from_numpy(weights)
        model.eval()

        # Inference
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, n_freq, device=dev),
            torch.linspace(-1, 1, n_frames, device=dev),
            indexing='ij',
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
        with torch.inference_mode():
            pred = model(coords).cpu().numpy()
        pred_mag = np.clip((pred + 1.0) / 2.0 * mag_max, 0, mag_max).reshape(n_freq, n_frames)

        # Decode phase
        phase_uint8 = np.frombuffer(zlib.decompress(phase_compressed), dtype=np.uint8).reshape(n_freq, n_frames)
        phase = phase_uint8.astype(np.float32) / 255 * 2 * np.pi - np.pi

        # Decode magnitude residual
        mag_residual = np.frombuffer(zlib.decompress(mag_residual_compressed), dtype=np.uint8).reshape(n_freq, n_frames)
        pred_scaled = np.round(pred_mag / mag_max * 255).astype(np.uint8)
        recon_mag_scaled = ((pred_scaled.astype(np.int16) + mag_residual.astype(np.int16)) % 256).astype(np.uint8)
        recon_mag = recon_mag_scaled.astype(np.float32) / 255 * mag_max

        # Inverse STFT
        from scipy.signal import istft
        Zxx = recon_mag * np.exp(1j * phase)
        _, audio = istft(Zxx, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
        if len(audio) > n_samples:
            audio = audio[:n_samples]
        elif len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))

        sha_got = hashlib.sha256(audio.astype(np.float32).tobytes()).digest()

        return audio, {
            'sample_rate': sample_rate,
            'n_samples': n_samples,
            'n_freq': n_freq,
            'n_frames': n_frames,
            'sha256_match': sha_got == sha_expected,
            'mode': 'audio',
        }


def _self_test():
    import zlib as _z
    print(f"[audio] device: {('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Generate synthetic audio: chord (3 sine waves)
    sample_rate = 16000
    duration = 1.0  # 1 second
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t) +
             0.2 * np.sin(2 * np.pi * 440 * t) +
             0.1 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)

    orig_bytes = audio.astype(np.float32).tobytes()
    zip_sz = len(_z.compress(orig_bytes, 9))
    print(f"[audio] Audio: {n_samples} samples, {duration:.1f}s @ {sample_rate}Hz")
    print(f"[audio] Original: {len(orig_bytes):,}B  ZIP: {zip_sz:,}B ({len(orig_bytes)/zip_sz:.2f}x)")

    comp = AudioCompressor(hidden_features=64, hidden_layers=3, omega_0=30.0,
                            n_fft=512, hop_length=128)
    t0 = time.time()
    res = comp.compress(audio, sample_rate=sample_rate, epochs=300, lr=2e-3,
                          bits=8, batch_size=8192, use_amp=True, patience=3, verbose=True)
    dt = time.time() - t0

    print(f"\n[audio] BLKH Audio: {res['recipe_size']:,}B  ({len(orig_bytes)/res['recipe_size']:.2f}x)")
    print(f"  SIREN weights:    {res['weights_packed_size']:,}B")
    print(f"  Phase (zlib):     {res['phase_compressed_size']:,}B")
    print(f"  Mag residual:     {res['mag_residual_size']:,}B")
    print(f"  SNR:              {res['snr_db']:.1f}dB")
    print(f"  Time:             {dt:.1f}s")
    print(f"  vs ZIP:           {zip_sz/res['recipe_size']:.2f}x")

    # Decompress
    rec_audio, meta = AudioCompressor.decompress(res['recipe_bytes'])
    print(f"\n[audio] Decompress: {len(rec_audio)} samples")
    print(f"  SHA-256: {meta['sha256_match']}")
    print(f"  SNR check: {res['snr_db']:.1f}dB (higher = better)")


if __name__ == '__main__':
    _self_test()
