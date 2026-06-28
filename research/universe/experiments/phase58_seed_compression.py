# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 58: Seed Compression (Compress SIREN Weights Further)
=============================================================
Tests whether the SIREN weights themselves can be compressed further.

CONCEPT:
  SIREN weights are float32 arrays. After INT8 quantization (Phase 41),
  they're still stored as raw bytes. Can we compress these bytes further?

  Approaches:
  1. zstd on INT8 weights (vs zlib)
  2. Brotli on INT8 weights
  3. Weight clustering (k-means → store cluster IDs)
  4. Delta encoding (store deltas from mean)

HYPOTHESIS:
  Weight clustering (k-means with 16 clusters) will achieve 2x additional
  compression because SIREN weights cluster around a few values.

METHOD:
  1. Train SIREN, quantize to INT8
  2. Try: zlib, zstd, brotli on raw INT8 bytes
  3. Try: k-means clustering (16, 32, 64 clusters)
  4. Compare all methods

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, zlib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, get_coordinates, train_single_siren, measure_model_size_compressed

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False

try:
    import brotli
    _HAS_BROTLI = True
except ImportError:
    _HAS_BROTLI = False


def quantize_int8(model):
    """Quantize model weights to INT8."""
    all_weights = []
    for param in model.parameters():
        w = param.detach().cpu().numpy()
        all_weights.extend(w.ravel())

    max_abs = max(np.abs(all_weights).max(), 1e-8)
    scale = max_abs / 127.0

    quantized = {}
    for name, param in model.named_parameters():
        w = param.detach().cpu().numpy()
        q = np.round(w / scale).astype(np.int8)
        quantized[name] = q

    return quantized, scale


def kmeans_compress(weights_int8, n_clusters=16):
    """Compress INT8 weights via k-means clustering."""
    # Flatten all weights
    flat = np.concatenate([w.ravel() for w in weights_int8.values()])

    # Simple k-means (1D — just bin the values)
    unique_vals = np.unique(flat)
    if len(unique_vals) <= n_clusters:
        # Fewer unique values than clusters — no compression needed
        return None, None, 0

    # K-means on 1D data
    from scipy.cluster.vq import kmeans, vq
    centroids, _ = kmeans(flat.astype(np.float64), n_clusters)
    labels, _ = vq(flat.astype(np.float64), centroids)

    # Reconstruct
    reconstructed = centroids[labels].astype(np.int8)

    # Pack: centroids (n_clusters × 1 byte) + labels (n_weights × log2(n_clusters) bits)
    # For 16 clusters: 4 bits per label → 2 labels per byte
    bits_per_label = int(np.ceil(np.log2(n_clusters)))

    # Pack labels
    label_bytes = bytearray()
    if n_clusters <= 16:  # 4 bits per label
        for i in range(0, len(labels), 2):
            byte = labels[i] & 0xF
            if i + 1 < len(labels):
                byte |= (labels[i + 1] & 0xF) << 4
            label_bytes.append(byte)
    elif n_clusters <= 256:  # 8 bits per label
        label_bytes = labels.astype(np.uint8).tobytes()

    # Compress
    import zlib
    centroid_bytes = centroids.astype(np.int8).tobytes()
    compressed = zlib.compress(bytes(label_bytes), 9)
    total = len(compressed) + len(centroid_bytes) + 8  # +8 for scale and n_clusters

    return labels, centroids, total


def run_phase58_experiment(verbose=True):
    print("=" * 80)
    print("🧪 Phase 58: Seed Compression (Compress SIREN Weights Further)")
    print("=" * 80)

    device = 'cpu'
    size = 128

    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(3):
            kx, ky = rng.integers(2, 5, 2)
            img[:, :, c] += 50 * np.sin(2 * np.pi * kx * xs) * np.cos(2 * np.pi * ky * ys)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    model, _ = train_single_siren(img, epochs=100, device=device, verbose=False)

    # Quantize to INT8
    quantized, scale = quantize_int8(model)
    raw_int8 = b''.join(w.tobytes() for w in quantized.values())
    raw_size = len(raw_int8)

    print(f"\n  INT8 weights: {raw_size:,}B (raw)")

    # Method 1: zlib
    zlib_size = len(zlib.compress(raw_int8, 9))

    # Method 2: zstd
    zstd_size = 0
    if _HAS_ZSTD:
        zstd_size = len(zstd.ZstdCompressor(level=22).compress(raw_int8))

    # Method 3: brotli
    brotli_size = 0
    if _HAS_BROTLI:
        brotli_size = len(brotli.compress(raw_int8, quality=11))

    # Method 4: k-means clustering
    kmeans_results = {}
    for n_clusters in [16, 32, 64]:
        _, _, kmeans_size = kmeans_compress(quantized, n_clusters)
        kmeans_results[n_clusters] = kmeans_size

    # Results
    print(f"\n{'Method':<30} {'Size':>8} {'vs raw INT8':>12} {'vs FP32':>10}")
    print("-" * 65)

    fp32_size = sum(p.numel() * 4 for p in model.parameters())  # raw FP32
    print(f"{'FP32 (raw)':<30} {fp32_size:>7,}B {'-':>11} {'1.00x':>9}")
    print(f"{'INT8 (raw)':<30} {raw_size:>7,}B {'1.00x':>11} {fp32_size/raw_size:>8.2f}x")
    print(f"{'INT8 + zlib':<30} {zlib_size:>7,}B {raw_size/zlib_size:>10.2f}x {fp32_size/zlib_size:>8.2f}x")

    if zstd_size:
        print(f"{'INT8 + zstd L22':<30} {zstd_size:>7,}B {raw_size/zstd_size:>10.2f}x {fp32_size/zstd_size:>8.2f}x")
    if brotli_size:
        print(f"{'INT8 + brotli':<30} {brotli_size:>7,}B {raw_size/brotli_size:>10.2f}x {fp32_size/brotli_size:>8.2f}x")

    for n_clusters, k_size in kmeans_results.items():
        if k_size > 0:
            print(f"{'INT8 + kmeans-' + str(n_clusters) + ' + zlib':<30} {k_size:>7,}B {raw_size/k_size:>10.2f}x {fp32_size/k_size:>8.2f}x")

    # Find best
    methods = [
        ('INT8 + zlib', zlib_size),
        ('INT8 + brotli', brotli_size if brotli_size else float('inf')),
        ('INT8 + zstd', zstd_size if zstd_size else float('inf')),
    ]
    for n_clusters, k_size in kmeans_results.items():
        if k_size > 0:
            methods.append((f'kmeans-{n_clusters}', k_size))

    best_name, best_size = min(methods, key=lambda x: x[1])
    best_ratio = fp32_size / max(best_size, 1)

    print(f"\n  🏆 Best: {best_name} = {best_size:,}B ({best_ratio:.2f}x vs FP32)")

    print(f"\n{'='*80}")
    print("📊 PHASE 58 SUMMARY — SEED COMPRESSION")
    print(f"{'='*80}")

    print(f"\n  📋 Compression pipeline:")
    print(f"  FP32 ({fp32_size:,}B) → INT8 ({raw_size:,}B) → {best_name} ({best_size:,}B)")
    print(f"  Total compression: {best_ratio:.2f}x from FP32")

    print(f"\n  📋 Key findings:")
    print(f"  - INT8 quantization: {fp32_size/raw_size:.1f}x compression (4→1 byte)")
    print(f"  - Entropy coding (zlib/zstd/brotli): {raw_size/best_size:.1f}x additional")
    print(f"  - K-means: {'helps' if any(k < zlib_size for k in kmeans_results.values() if k > 0) else 'limited benefit'}")
    print(f"  - Combined: {best_ratio:.1f}x total seed compression")

    print(f"\n  📋 Practical impact:")
    print(f"  - Original seed: ~8.6KB (FP32 + zlib)")
    print(f"  - Optimized seed: ~{best_size/1024:.1f}KB (INT8 + {best_name})")
    print(f"  - Savings: {(1 - best_size/8600)*100:.0f}% smaller seeds!")

    return {
        'fp32_size': fp32_size,
        'int8_size': raw_size,
        'best_method': best_name,
        'best_size': best_size,
        'best_ratio': best_ratio,
    }


if __name__ == '__main__':
    results = run_phase58_experiment(verbose=True)
