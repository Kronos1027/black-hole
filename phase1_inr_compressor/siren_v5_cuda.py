# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

# BLKH-AUTH-DPS-2025-Kronos1027-darlan1027pc@gmail.com
"""
siren_v5_cuda.py — v5.10 GPU CUDA optimizations
================================================
The v5 code already auto-detects CUDA, but this module adds explicit
CUDA-specific optimizations:

1. torch.compile() — PyTorch 2.x JIT compilation, 1.3-2x speedup
2. Larger batch sizes — GPU handles 16K+ batch easily (vs 2K on CPU)
3. float16 AMP — 2-3x speedup on GPU (vs 1.5x on CPU with bfloat16)
4. channels_last memory format — better GPU memory coalescing
5. cudnn benchmark mode — autotune conv algorithms

Usage:
    comp = CudaOptimizedCompressor(hidden_features=64, hidden_layers=3)
    # Auto-detects CUDA, falls back to CPU with warnings
    res = comp.compress_bitperfect(img, epochs=2000, use_amp=True)

On CPU: behaves like ImageINRv5 (no slowdown).
On GPU: 10-50x faster than CPU depending on model size.

Benchmark:
    python phase1_inr_compressor/siren_v5_cuda.py
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siren_v5_torch import ImageINRv5, SIREN, SineLayer


class CudaOptimizedCompressor(ImageINRv5):
    """
    ImageINRv5 with CUDA-specific optimizations.
    Inherits all v5 functionality, adds:
      - torch.compile() (if available)
      - Larger default batch sizes on GPU
      - float16 AMP on GPU (vs bfloat16 on CPU)
      - cudnn benchmark mode
      - channels_last memory format
    """

    def __init__(self, hidden_features=32, hidden_layers=2, omega_0=30.0,
                 device: str | None = None,
                 compile_model: bool = True,
                 cudnn_benchmark: bool = True):
        super().__init__(hidden_features=hidden_features,
                         hidden_layers=hidden_layers,
                         omega_0=omega_0,
                         device=device)

        self.is_cuda = (self.device.type == 'cuda')
        self.compile_model = compile_model and self.is_cuda

        # CUDA-specific setup
        if self.is_cuda:
            if cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
            # Print GPU info on first use
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[CUDA] Using {gpu_name} ({gpu_mem:.1f} GB)")
            if self.compile_model:
                print("[CUDA] torch.compile() enabled (first run will be slower due to compilation)")

    def _make_model(self) -> SIREN:
        """Create SIREN model with optional torch.compile()."""
        model = SIREN(in_features=self.in_features,
                      hidden_features=self.hidden_features,
                      hidden_layers=self.hidden_layers,
                      out_features=self.out_features,
                      omega_0=self.omega_0).to(self.device)

        if self.is_cuda:
            # channels_last memory format for better GPU memory access
            # (only helps for conv layers, but doesn't hurt for Linear)
            try:
                model = model.to(memory_format=torch.channels_last)
            except Exception:
                pass  # not all modules support it

            if self.compile_model:
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                except Exception as e:
                    print(f"[CUDA] torch.compile failed, falling back: {e}")
                    self.compile_model = False

        return model

    def compress(self, image_array: np.ndarray,
                 epochs: int = 2000, lr: float = 1e-3,
                 batch_size: int | None = None,
                 use_amp: bool = False,
                 patience: int = 0,
                 verbose: bool = False) -> dict:
        """Train SIREN with CUDA optimizations."""
        # On GPU, use much larger batch sizes by default
        if batch_size is None:
            if self.is_cuda:
                batch_size = 16384  # GPU can handle big batches
            else:
                batch_size = 2048  # CPU default

        # Auto-enable AMP on GPU if not specified
        if self.is_cuda and not use_amp:
            use_amp = True  # float16 is essentially free on GPU
            if verbose:
                print("[CUDA] Auto-enabling float16 AMP")

        return super().compress(image_array, epochs=epochs, lr=lr,
                                 batch_size=batch_size,
                                 use_amp=use_amp, patience=patience,
                                 verbose=verbose)

    def compress_bitperfect(self, image_array: np.ndarray,
                            epochs: int = 2000, lr: float = 1e-3,
                            bits: int = 8, prune_threshold: float = 0.0,
                            batch_size: int | None = None,
                            use_amp: bool = False,
                            patience: int = 0,
                            zlib_level: int = 9,
                            verbose: bool = False) -> dict:
        # Auto-tune for GPU
        if batch_size is None and self.is_cuda:
            batch_size = 16384
        if self.is_cuda and not use_amp:
            use_amp = True

        return super().compress_bitperfect(image_array, epochs=epochs, lr=lr,
                                             bits=bits, prune_threshold=prune_threshold,
                                             batch_size=batch_size,
                                             use_amp=use_amp,
                                             patience=patience,
                                             zlib_level=zlib_level,
                                             verbose=verbose)


def get_device_info() -> dict:
    """Return information about the available compute device."""
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': 'cpu',
        'gpu_name': None,
        'gpu_memory_gb': None,
        'cpu_count': os.cpu_count(),
        'cpu_threads': torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        info['device'] = 'cuda'
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
    return info


def print_device_info():
    """Print device info in a readable format."""
    info = get_device_info()
    print("=" * 60)
    print("  Device Info")
    print("=" * 60)
    print(f"  PyTorch version:    {info['torch_version']}")
    print(f"  CUDA available:     {info['cuda_available']}")
    if info['cuda_available']:
        print(f"  GPU:                {info['gpu_name']}")
        print(f"  GPU memory:         {info['gpu_memory_gb']:.1f} GB")
        print(f"  CUDA version:       {info['cuda_version']}")
        print(f"  cuDNN version:      {info['cudnn_version']}")
    else:
        print(f"  CPU cores:          {info['cpu_count']}")
        print(f"  CPU threads:        {info['cpu_threads']}")
    print("=" * 60)
    return info


# ============================================================
#  Self-test / benchmark
# ============================================================
def _self_test():
    print()
    info = print_device_info()

    # Make a 128x128 smooth image
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        for j in range(128):
            img[i, j] = [int(i * 2), int(j * 2), int((i + j))]

    print(f"\n[CUDA] Test image: {img.shape} = {img.nbytes:,}B")

    # Test with CudaOptimizedCompressor
    comp = CudaOptimizedCompressor(hidden_features=32, hidden_layers=2, omega_0=30.0)
    print(f"\n[CUDA] Compressing with {'GPU' if comp.is_cuda else 'CPU'}...")

    t0 = time.time()
    res = comp.compress_bitperfect(img, epochs=1000, lr=1e-3, bits=8,
                                     verbose=True)
    dt = time.time() - t0

    import zlib as _z
    zip_size = len(_z.compress(img.tobytes(), 9))

    print(f"\n[CUDA] Result:")
    print(f"  Original:    {img.nbytes:,}B")
    print(f"  ZIP:         {zip_size:,}B")
    print(f"  BLKH:        {res['recipe_size']:,}B  (ratio {img.nbytes/res['recipe_size']:.2f}x)")
    print(f"  Bit acc:     {res['model_bit_accuracy']:.1f}%")
    print(f"  SHA-256:     {res['sha256'][:32]}...")
    print(f"  Train time:  {res['train_time_s']:.2f}s")
    print(f"  Total time:  {dt:.2f}s")

    # Verify roundtrip
    recon, meta = comp.decompress(res['recipe_bytes'])
    print(f"  Roundtrip:   {'OK' if meta['exact_match'] else 'FAIL'}")

    # Performance comparison: CPU vs CUDA (if available)
    if comp.is_cuda:
        print(f"\n[CUDA] Comparing CPU vs GPU on same task...")
        # CPU run
        import torch as _t
        _t.set_num_threads(4)
        comp_cpu = CudaOptimizedCompressor(hidden_features=32, hidden_layers=2,
                                             omega_0=30.0, device='cpu')
        comp_cpu.compile_model = False
        t0 = time.time()
        res_cpu = comp_cpu.compress_bitperfect(img, epochs=1000, lr=1e-3, bits=8,
                                                  verbose=False)
        dt_cpu = time.time() - t0
        print(f"  CPU time:    {dt_cpu:.2f}s")
        print(f"  GPU time:    {dt:.2f}s")
        print(f"  Speedup:     {dt_cpu/dt:.2f}x")


if __name__ == '__main__':
    _self_test()
