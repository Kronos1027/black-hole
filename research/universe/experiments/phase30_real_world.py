# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 30: Universal File Compressor (Real-World Test)
=======================================================
Final real-world test: compress a mixed directory of REAL files.

CONCEPT:
  All previous phases used synthetic data. This phase tests on
  REAL files from the BLKH repository itself:
  - Python source code (.py)
  - JSON data files (.json)
  - PNG images (.png)
  - Binary data (.bin)
  - Text files (.md, .txt)

  The Universe auto-detects type and picks best method.

METHOD:
  1. Collect real files from the repository
  2. Compress each with Universe auto-selection
  3. Compare total size with ZIP (tar + gzip equivalent)

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time, zlib, io, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))


def collect_real_files():
    """Collect real files from the repository."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
    files = {}

    # Python files
    for fname in ['blkh.py', 'demo.py']:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            files[fname] = open(path, 'rb').read()

    # JSON files
    json_dir = os.path.join(base, 'docs', 'assets')
    if os.path.exists(json_dir):
        for f in os.listdir(json_dir):
            if f.endswith('.json'):
                path = os.path.join(json_dir, f)
                files[f] = open(path, 'rb').read()

    # PNG images
    photos_dir = os.path.join(base, 'docs', 'assets', 'sample_photos')
    if os.path.exists(photos_dir):
        for f in sorted(os.listdir(photos_dir))[:3]:
            if f.endswith('.png'):
                path = os.path.join(photos_dir, f)
                files[f] = open(path, 'rb').read()

    # Markdown
    for fname in ['README.md', 'CONTRIBUTING.md']:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            data = open(path, 'rb').read()[:10240]
            files[fname] = data

    return files


def compress_with_universe(name, data):
    """Compress file with auto-selection."""
    ext = os.path.splitext(name)[1].lower()

    if ext in ('.png', '.jpg', '.jpeg'):
        try:
            from siren_v5_dct import DCTCompressor
            img = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            comp = DCTCompressor(quality=0.9, codec='brotli')
            res = comp.compress(img, verbose=False)
            return res['recipe_size'], 'DCT'
        except Exception:
            return len(zlib.compress(data, 9)), 'zlib'

    elif ext in ('.json',):
        # Try program synthesis
        text = data.decode('utf-8', errors='ignore')
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
        try:
            from phase3_program_synthesis import compress_with_program
            prog_size, method, _ = compress_with_program(text)
            zlib_size = len(zlib.compress(data, 9))
            if prog_size < zlib_size:
                return prog_size, f'program_{method}'
            return zlib_size, 'zlib'
        except Exception:
            return len(zlib.compress(data, 9)), 'zlib'

    else:
        # Text, binary, etc: zlib
        return len(zlib.compress(data, 9)), 'zlib'


def run_phase30_experiment(verbose=True):
    """Run Phase 30 Universal File Compressor experiment."""
    print("=" * 80)
    print("🧪 Phase 30: Universal File Compressor (Real-World Test)")
    print("=" * 80)

    # Collect real files
    print("\n📦 Collecting real files from repository...")
    files = collect_real_files()
    print(f"  Found {len(files)} files")

    # Compress each
    print(f"\n{'File':<30} {'Type':<8} {'Original':>10} {'Universe':>10} {'ZIP':>10} {'Method':<12} {'vs ZIP':>8}")
    print("-" * 95)

    total_original = 0
    total_universe = 0
    total_zip = 0
    results = []

    for name, data in sorted(files.items()):
        ext = os.path.splitext(name)[1].lower().lstrip('.')
        orig = len(data)
        zip_sz = len(zlib.compress(data, 9))
        uni_sz, method = compress_with_universe(name, data)

        total_original += orig
        total_universe += uni_sz
        total_zip += zip_sz

        vs_zip = zip_sz / max(uni_sz, 1)
        type_label = ext[:7] if ext else 'unknown'
        print(f"{name:<30} {type_label:<8} {orig:>9,}B {uni_sz:>9,}B {zip_sz:>9,}B {method:<12} {vs_zip:>7.2f}x")

        results.append({
            'name': name,
            'type': ext,
            'original': orig,
            'universe': uni_sz,
            'zip': zip_sz,
            'method': method,
        })

    # Summary
    print(f"\n{'-'*95}")
    print(f"{'TOTAL':<30} {'':>8} {total_original:>9,}B {total_universe:>9,}B {total_zip:>9,}B {'':>12} {total_zip/max(total_universe,1):>7.2f}x")

    print(f"\n{'='*80}")
    print("📊 PHASE 30 RESULTS — REAL-WORLD UNIVERSAL COMPRESSION")
    print(f"{'='*80}")
    print(f"\n  Files: {len(files)}")
    print(f"  Original: {total_original:,}B ({total_original/1024:.1f}KB)")
    print(f"  ZIP: {total_zip:,}B ({total_zip/1024:.1f}KB)")
    print(f"  Universe: {total_universe:,}B ({total_universe/1024:.1f}KB)")
    print(f"  vs ZIP: {total_zip/max(total_universe,1):.2f}x")

    # Method breakdown
    methods = {}
    for r in results:
        m = r['method']
        if m not in methods:
            methods[m] = {'count': 0, 'orig': 0, 'comp': 0}
        methods[m]['count'] += 1
        methods[m]['orig'] += r['original']
        methods[m]['comp'] += r['universe']

    print(f"\n  📋 Method breakdown:")
    print(f"  {'Method':<15} {'Files':>6} {'Original':>10} {'Compressed':>11} {'Ratio':>8}")
    print(f"  {'-'*55}")
    for m, s in sorted(methods.items()):
        ratio = s['orig'] / max(s['comp'], 1)
        print(f"  {m:<15} {s['count']:>6} {s['orig']:>9,}B {s['comp']:>10,}B {ratio:>7.1f}x")

    return {
        'n_files': len(files),
        'total_original': total_original,
        'total_universe': total_universe,
        'total_zip': total_zip,
        'vs_zip': total_zip / max(total_universe, 1),
    }


if __name__ == '__main__':
    results = run_phase30_experiment(verbose=True)
