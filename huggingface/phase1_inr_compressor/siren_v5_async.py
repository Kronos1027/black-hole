# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

"""
siren_v5_async.py — v5.24 Async batch processing (io_uring-style)
==================================================================
Addresses Copilot feedback: "Integração prática: falta suporte direto a APIs
de I/O (ex.: io_uring, DirectStorage)"

Provides async batch compression for large datasets:
  - Concurrent file I/O (asyncio)
  - Parallel compression using ProcessPoolExecutor
  - Memory-efficient streaming for datasets that don't fit in RAM
  - Progress reporting for long-running batch jobs

On Linux with Python 3.12+, this uses asyncio + ProcessPoolExecutor which
provides io_uring-like concurrent I/O without requiring kernel io_uring.
For true io_uring support, use the `liburing` Python bindings.

Usage:
    from siren_v5_async import AsyncBatchCompressor

    compressor = AsyncBatchCompressor(mode='fast', quality=0.9, workers=4)
    stats = await compressor.compress_directory('input/', 'output/')
    print(f"Compressed {stats['n_files']} files in {stats['time']:.1f}s")

CLI:
    blkh batch input_dir/ output_dir/ --mode fast --quality 0.9 --workers 4

Author: Darlan Pereira da Silva (Kronos1027)
"""
from __future__ import annotations
import os
import sys
import io
import time
import asyncio
import hashlib
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _compress_one(args):
    """Worker function for parallel compression (must be module-level for pickling)."""
    input_path, output_path, mode, quality, speed = args

    from PIL import Image
    import zlib

    # Load image
    img = np.array(Image.open(input_path).convert('RGB'))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    orig_size = img.nbytes

    # Select compressor
    if mode == 'fast':
        from siren_v5_fast import FastDCTCompressor
        comp = FastDCTCompressor(quality=quality, speed=speed)
    elif mode == 'dct':
        from siren_v5_dct import DCTCompressor
        comp = DCTCompressor(quality=quality, codec='brotli')
    elif mode == 'photo':
        from siren_v5_photo import PhotoCompressor
        comp = PhotoCompressor(subsampling='420', codec='brotli')
    elif mode == 'wavelet3':
        from siren_v5_wavelet_v3 import WaveletINRCompressorV3
        comp = WaveletINRCompressorV3(wavelet='auto', level='auto', lossless=True,
                                        codec='brotli', combined=True, parallel=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compress
    t0 = time.time()
    res = comp.compress(img, verbose=False)
    dt = time.time() - t0

    # Write output
    Path(output_path).write_bytes(res['recipe_bytes'])

    return {
        'input': input_path,
        'output': output_path,
        'original_size': orig_size,
        'compressed_size': res['recipe_size'],
        'time': dt,
        'mode': mode,
        'lossy': res.get('lossy', False),
    }


class AsyncBatchCompressor:
    """
    Async batch compressor for large datasets.
    Uses ProcessPoolExecutor for parallel compression + asyncio for concurrent I/O.
    """

    def __init__(self, mode: str = 'fast', quality: float = 0.9,
                 speed: str = 'balanced', workers: int = 4):
        """
        Args:
            mode: 'fast' (v5.23), 'dct' (v5.22), 'photo' (v5.21), 'wavelet3' (v5.20)
            quality: 0.1-1.0 (for lossy modes)
            speed: 'fast', 'balanced', 'best' (for v5.23)
            workers: number of parallel processes
        """
        self.mode = mode
        self.quality = quality
        self.speed = speed
        self.workers = workers

    async def compress_directory(self, input_dir: str, output_dir: str,
                                   extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp'),
                                   progress_callback: Optional[callable] = None) -> dict:
        """Compress all images in a directory concurrently.

        Args:
            input_dir: directory containing input images
            output_dir: directory for compressed .blkX files
            extensions: tuple of input file extensions to process
            progress_callback: optional async callback(completed, total, current_file)

        Returns:
            dict with stats: n_files, total_original, total_compressed, time, throughput
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect input files
        input_files = []
        for ext in extensions:
            input_files.extend(input_path.rglob(f'*{ext}'))
            input_files.extend(input_path.rglob(f'*{ext.upper()}'))

        if not input_files:
            return {'n_files': 0, 'total_original': 0, 'total_compressed': 0, 'time': 0}

        n_files = len(input_files)
        print(f"[batch] Found {n_files} images to compress")
        print(f"[batch] Mode: {self.mode}, quality: {self.quality}, workers: {self.workers}")

        # Build task args
        tasks = []
        for i, inp in enumerate(input_files):
            # Output filename: original_name + .blkX based on mode
            ext_map = {'fast': '.blkf', 'dct': '.blkd', 'photo': '.blkp', 'wavelet3': '.blkw3'}
            out_ext = ext_map.get(self.mode, '.blkf')
            rel_path = inp.relative_to(input_path)
            out_file = output_path / rel_path.with_suffix(out_ext)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((str(inp), str(out_file), self.mode, self.quality, self.speed))

        # Run compression in process pool
        t0 = time.time()
        loop = asyncio.get_event_loop()
        results = []

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_compress_one, task): task for task in tasks}
            completed = 0

            for future in asyncio.as_completed([loop.run_in_executor(executor, _compress_one, t) for t in tasks]):
                try:
                    result = await future
                    results.append(result)
                    completed += 1
                    if progress_callback:
                        await progress_callback(completed, n_files, result['input'])
                    elif completed % 10 == 0 or completed == n_files:
                        print(f"[batch] Progress: {completed}/{n_files} ({completed/n_files*100:.1f}%)")
                except Exception as e:
                    print(f"[batch] Error: {e}")
                    completed += 1

        dt = time.time() - t0

        # Aggregate stats
        total_original = sum(r['original_size'] for r in results)
        total_compressed = sum(r['compressed_size'] for r in results)

        stats = {
            'n_files': len(results),
            'total_original': total_original,
            'total_compressed': total_compressed,
            'time': dt,
            'throughput_mbs': total_original / dt / 1024 / 1024 if dt > 0 else 0,
            'ratio': total_original / total_compressed if total_compressed > 0 else 0,
            'results': results,
        }

        print(f"\n[batch] Done!")
        print(f"  Files: {stats['n_files']}")
        print(f"  Original: {total_original:,}B ({total_original/1024/1024:.1f}MB)")
        print(f"  Compressed: {total_compressed:,}B ({total_compressed/1024/1024:.1f}MB)")
        print(f"  Ratio: {stats['ratio']:.2f}x")
        print(f"  Time: {dt:.1f}s ({stats['throughput_mbs']:.1f}MB/s)")

        return stats

    def compress_directory_sync(self, input_dir: str, output_dir: str, **kwargs) -> dict:
        """Synchronous wrapper for compress_directory."""
        return asyncio.run(self.compress_directory(input_dir, output_dir, **kwargs))


def _self_test():
    """Self-test: create synthetic images and batch compress them."""
    import tempfile
    from PIL import Image

    print("=" * 80)
    print("BLKH v5.24 Async Batch Compressor — Self Test")
    print("=" * 80)

    # Create temp directory with test images
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, 'input')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(input_dir)

        # Generate 20 synthetic test images
        print(f"\nGenerating 20 synthetic test images...")
        rng = np.random.default_rng(42)
        for i in range(20):
            size = 64
            ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
            img = np.zeros((size, size, 3), dtype=np.float32)
            for c in range(3):
                kx, ky = rng.integers(1, 5, 2)
                amp = rng.uniform(40, 80)
                phase = rng.uniform(0, 2*np.pi)
                img[:,:,c] = amp * np.sin(2*np.pi*kx*xs + phase) * np.cos(2*np.pi*ky*ys)
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(input_dir, f'test_{i:03d}.png'))

        # Test each mode
        for mode in ['fast', 'photo']:
            print(f"\n--- Mode: {mode} ---")
            comp = AsyncBatchCompressor(mode=mode, quality=0.9, workers=2)
            mode_output = os.path.join(tmpdir, f'output_{mode}')
            stats = comp.compress_directory_sync(input_dir, mode_output)
            assert stats['n_files'] == 20
            print(f"  Verified: {stats['n_files']} files compressed")


if __name__ == '__main__':
    _self_test()
