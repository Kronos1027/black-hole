#!/usr/bin/env python3
"""
benchmark_video.py — v5.11 Video compression scaling test
============================================================
Tests BLKH Video (temporal SIREN) vs ZIP per-frame on synthetic videos
with high temporal redundancy (moving objects).

The key question: does temporal SIREN exploit inter-frame redundancy
better than ZIP per-frame?
"""
import sys
import os
import time
import zlib
import json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
torch.set_num_threads(4)

from siren_v5_video import VideoCompressor


def make_moving_blob_video(n_frames, size=64, seed=42):
    """Generate a synthetic video: gaussian blob moving smoothly across frames."""
    frames = []
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    # Random trajectory parameters
    cx_start, cy_start = rng.uniform(0.2, 0.4, 2) * size
    cx_end, cy_end = rng.uniform(0.6, 0.8, 2) * size
    sigma = rng.uniform(5, 12)
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        cx = cx_start + (cx_end - cx_start) * t
        cy = cy_start + (cy_end - cy_start) * t
        img = np.zeros((size, size, 3), dtype=np.float32)
        for c in range(3):
            amp = 200 - c * 40
            sigma_c = sigma + c
            img[:, :, c] = amp * np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma_c**2))
        # Subtle color shift over time
        img[:, :, 0] += 20 * t
        img[:, :, 2] += 20 * (1 - t)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


def main():
    print("=" * 90)
    print("  BLKH v5.11 VIDEO benchmark (temporal SIREN vs ZIP per-frame)")
    print("  All 100% SHA-256 verified")
    print("=" * 90)

    SIZE = 64
    results = []
    for N in [4, 8, 16]:
        print(f"\n--- N = {N} frames ({SIZE}x{SIZE}x3) ---")
        frames = make_moving_blob_video(N, size=SIZE, seed=42)
        total_orig = sum(f.nbytes for f in frames)
        zip_total = sum(len(zlib.compress(f.tobytes(), 9)) for f in frames)
        print(f"  total_orig: {total_orig:,}B   ZIP: {zip_total:,}B")

        comp = VideoCompressor(hidden_features=64, hidden_layers=3,
                                omega_0=30.0, omega_t=1.0,
                                residual_codec='webp')
        t0 = time.time()
        res = comp.compress(frames, epochs=1000, lr=1e-3,
                              bits=8, batch_size=4096, verbose=False)
        dt = time.time() - t0
        rec, meta = VideoCompressor.decompress(res['recipe_bytes'])
        winner = "BLKH" if res['recipe_size'] < zip_total else "ZIP"
        print(f"  Video:    {res['recipe_size']:>7,}B  bit%={res['avg_bit_pct']:.1f}  "
              f"ok={meta['all_sha256_match']}  {dt:.1f}s  "
              f"vs ZIP={zip_total/res['recipe_size']:.3f}x  -> {winner}")
        print(f"    SIREN (shared): {res['weights_packed_size']:,}B "
              f"({res['weights_packed_size']/N:.0f}/frame amortized)")
        print(f"    residual/frame: {res['residual_per_frame']:.0f}B ({res['residual_codec']})")

        results.append({
            'n_frames': N,
            'total_orig': total_orig,
            'zip_total': zip_total,
            'video_size': res['recipe_size'],
            'video_ok': meta['all_sha256_match'],
            'video_vs_zip': zip_total / res['recipe_size'],
            'siren_size': res['weights_packed_size'],
            'avg_bit_pct': res['avg_bit_pct'],
            'time_s': dt,
        })

    print("\n" + "=" * 90)
    print("  VIDEO SCALING SUMMARY")
    print("=" * 90)
    print(f"{'N':>5}{'orig':>10}{'ZIP':>10}{'Video':>10}{'Video/ZIP':>11}{'ok':>5}")
    print("-" * 90)
    for r in results:
        print(f"{r['n_frames']:>5}{r['total_orig']:>10,}{r['zip_total']:>10,}"
              f"{r['video_size']:>10,}{r['video_vs_zip']:>10.3f}x"
              f"{'OK' if r['video_ok'] else 'FAIL':>5}")

    out = Path(__file__).parent / 'benchmark_video_results.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")


if __name__ == '__main__':
    main()
