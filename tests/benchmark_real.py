#!/usr/bin/env python3
"""
Black Hole - Real-World Benchmark Suite
Tests BLKH against ZIP on real data: images, text, audio, patterns, and random data.
Shows the Kolmogorov limit for high-entropy data.
"""
import sys
import os
import zipfile
import time
import json
import tempfile

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
from siren_core import DataINRCompressor
import numpy as np

def zip_compress(data, level=zipfile.ZIP_DEFLATED):
    """Compress data using ZIP (standard zlib)."""
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
        zip_path = f.name
    with zipfile.ZipFile(zip_path, 'w', compression=level) as zf:
        zf.writestr('data', data)
    with open(zip_path, 'rb') as f:
        compressed = f.read()
    os.unlink(zip_path)
    return compressed

def benchmark_file(filepath, data_type, epochs=3000, lr=1e-4):
    """Benchmark a single file against both ZIP and Black Hole."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {os.path.basename(filepath)} ({data_type})")
    print(f"{'='*70}")
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    original_size = len(data)
    print(f"Original: {original_size} bytes")
    
    # --- ZIP Benchmark ---
    zip_start = time.time()
    zip_data = zip_compress(data)
    zip_time = time.time() - zip_start
    zip_size = len(zip_data)
    zip_ratio = original_size / zip_size if zip_size > 0 else 0
    print(f"\n[ZIP] Size: {zip_size} bytes | Ratio: {zip_ratio:.2f}x | Time: {zip_time:.4f}s")
    
    # --- Black Hole Benchmark ---
    bh_start = time.time()
    compressor = DataINRCompressor(hidden_dim=64, num_layers=3)
    meta = compressor.compress(data, epochs=epochs, lr=lr)
    
    with tempfile.NamedTemporaryFile(suffix='.recipe.json', delete=False) as f:
        recipe_path = f.name
    compressor.save_recipe(recipe_path)
    
    with open(recipe_path, 'rb') as f:
        recipe_data = f.read()
    recipe_size = len(recipe_data)
    bh_time = time.time() - bh_start
    
    # Reconstruct
    reconstructed = compressor.reconstruct(original_size)
    mse = np.mean((reconstructed.astype(np.float32) - np.frombuffer(data, dtype=np.uint8).astype(np.float32)) ** 2)
    psnr = 10 * np.log10(65025.0 / mse) if mse > 0 else float('inf')
    accuracy = np.mean(reconstructed == np.frombuffer(data, dtype=np.uint8)) * 100
    
    bh_ratio = original_size / recipe_size if recipe_size > 0 else 0
    print(f"[BLKH] Recipe: {recipe_size} bytes | Ratio: {bh_ratio:.2f}x | Time: {bh_time:.2f}s")
    print(f"[BLKH] PSNR: {psnr:.2f} dB | Accuracy: {accuracy:.2f}%")
    
    os.unlink(recipe_path)
    
    return {
        'file': os.path.basename(filepath),
        'type': data_type,
        'original_size': original_size,
        'zip_size': zip_size,
        'zip_ratio': zip_ratio,
        'zip_time': zip_time,
        'bh_recipe_size': recipe_size,
        'bh_ratio': bh_ratio,
        'bh_time': bh_time,
        'bh_psnr': psnr,
        'bh_accuracy': accuracy,
    }

def main():
    test_dir = os.path.join(os.path.dirname(__file__), 'real_data')
    
    if not os.path.exists(test_dir):
        print(f"ERROR: Test data directory not found: {test_dir}")
        print("Run this first to generate test data:")
        print("  python -c \"import generate_test_data\" (from generate_test_data.py)")
        sys.exit(1)
    
    results = []
    
    # 1. Structured pattern (predictable, should do well)
    f = os.path.join(test_dir, 'test_pattern.bin')
    if os.path.exists(f):
        results.append(benchmark_file(f, 'Structured Pattern', epochs=3000, lr=1e-4))
    
    # 2. Text (structured, moderate entropy)
    f = os.path.join(test_dir, 'test_text.txt')
    if os.path.exists(f):
        results.append(benchmark_file(f, 'Text Document', epochs=3000, lr=1e-4))
    
    # 3. Image raw (structured 2D data flattened to 1D)
    f = os.path.join(test_dir, 'test_image.raw')
    if os.path.exists(f):
        results.append(benchmark_file(f, 'Image (16x16 RGB flattened)', epochs=3000, lr=1e-4))
    
    # 4. Audio (periodic, SIREN-friendly)
    f = os.path.join(test_dir, 'test_audio.raw')
    if os.path.exists(f):
        results.append(benchmark_file(f, 'Audio (440Hz+880Hz sine)', epochs=3000, lr=1e-4))
    
    # 5. Random (Kolmogorov limit - uncompressible)
    f = os.path.join(test_dir, 'test_random.bin')
    if os.path.exists(f):
        results.append(benchmark_file(f, 'Random (Kolmogorov Limit)', epochs=1000, lr=1e-4))
    
    # --- Summary Table ---
    print(f"\n{'='*90}")
    print(f"{'BENCHMARK SUMMARY':^90}")
    print(f"{'='*90}")
    print(f"{'File':<30} {'Type':<25} {'ZIP':>8} {'BLKH':>8} {'ZIP vs BLKH':>12}")
    print(f"{'-'*90}")
    
    for r in results:
        label = "ZIP wins" if r['zip_ratio'] > r['bh_ratio'] else "BLKH wins"
        print(f"{r['file']:<30} {r['type']:<25} {r['zip_ratio']:>7.2f}x {r['bh_ratio']:>7.2f}x {label:>12}")
    
    print(f"{'='*90}")
    print("\nKEY INSIGHTS:")
    print("- ZIP excels on text and patterns (dictionary-based, well-optimized)")
    print("- BLKH shows potential on periodic/structured signals (SIREN's strength)")
    print("- Random data (Kolmogorov limit): Neither compresses well. This is expected.")
    print("- Current prototype: recipe size > original due to unquantized float32 weights.")
    print("- Roadmap: 8-bit quantization (-4x), meta-learning (-100x) will close the gap.")
    
    # Save report (convert numpy types to Python native)
    report_path = os.path.join(os.path.dirname(__file__), 'benchmark_report.json')
    clean_results = []
    for r in results:
        clean = {}
        for k, v in r.items():
            if hasattr(v, 'item'):  # numpy scalar
                clean[k] = v.item()
            else:
                clean[k] = v
        clean_results.append(clean)
    with open(report_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    main()
