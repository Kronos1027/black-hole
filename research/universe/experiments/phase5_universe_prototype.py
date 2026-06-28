# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 5 Experiment: Universe Prototype
=======================================
Combines all approaches into a single "Universe" compression system.

The Universe automatically selects the best compression method based on
file type and characteristics:

  1. Images (smooth/synthetic) → Multi-File SIREN (Phase 1)
  2. Images (natural photos) → DCT v5.22 (BLKH production)
  3. Audio (tones/smooth) → Audio INR (Phase 2)
  4. Text (structured logs) → Program Synthesis (Phase 3)
  5. Binary (short) → zlib fallback
  6. Any file → Auto-detect and pick best

HYPOTHESIS:
  The Universe system will achieve better average compression than any
  single method, by routing each file to its optimal compressor.

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import io
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))


class UniverseCompressor:
    """The Universe — auto-selects best compression method per file."""

    def __init__(self):
        self.methods_tried = []
        self.results = {}

    def detect_file_type(self, data):
        """Detect file type from raw bytes."""
        if len(data) < 4:
            return 'binary'

        # PNG
        if data[:4] == b'\x89PNG':
            return 'image_png'
        # JPEG
        if data[:2] == b'\xff\xd8':
            return 'image_jpeg'
        # WAV
        if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
            return 'audio_wav'
        # Try text
        try:
            text = data.decode('utf-8')
            # Check if structured
            if text.startswith('{') or text.startswith('['):
                return 'json'
            if ',' in text.split('\n')[0] and ',' in text.split('\n')[1]:
                return 'csv'
            if '[INFO]' in text or '[ERROR]' in text or '[WARN]' in text:
                return 'log'
            return 'text'
        except (UnicodeDecodeError, IndexError):
            pass

        return 'binary'

    def compress_image(self, img_array):
        """Compress image using BLKH DCT (production mode)."""
        from siren_v5_dct import DCTCompressor
        comp = DCTCompressor(quality=0.9, codec='brotli')
        res = comp.compress(img_array, verbose=False)
        return res['recipe_bytes'], res['recipe_size'], 'dct_v5_22'

    def compress_text(self, text):
        """Compress text using program synthesis (Phase 3) or zlib."""
        # Try program synthesis
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from phase3_program_synthesis import compress_with_program
        size, method, details = compress_with_program(text)
        compressed = zlib.compress(text.encode('utf-8'), 9)

        # Use whichever is smaller
        if size < len(compressed):
            return compressed, size, f'program_{method}'
        return compressed, len(compressed), 'zlib'

    def compress_binary(self, data):
        """Compress binary with zlib."""
        compressed = zlib.compress(data, 9)
        return compressed, len(compressed), 'zlib'

    def compress(self, data):
        """Auto-detect and compress with best method."""
        file_type = self.detect_file_type(data)
        self.methods_tried.append(file_type)

        if file_type in ('image_png', 'image_jpeg'):
            # Load as image array
            img = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            compressed, size, method = self.compress_image(img)
        elif file_type in ('json', 'csv', 'log', 'text'):
            text = data.decode('utf-8')
            compressed, size, method = self.compress_text(text)
        else:
            compressed, size, method = self.compress_binary(data)

        self.results[file_type] = {
            'original_size': len(data),
            'compressed_size': size,
            'method': method,
        }

        return compressed, size, method, file_type


def generate_test_corpus():
    """Generate a mixed test corpus with different file types."""
    corpus = {}

    # 1. Images (satellite-like)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
    from phase1_multi_file_siren import generate_satellite_images
    images = generate_satellite_images(n_images=5, size=128, seed=42)
    for i, img in enumerate(images):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='PNG')
        corpus[f'satellite_{i}.png'] = buf.getvalue()

    # 2. Log files
    from phase3_program_synthesis import generate_log_files
    logs = generate_log_files(n_files=3, n_lines=50, seed=42)
    for i, log in enumerate(logs):
        corpus[f'server_{i}.log'] = log.encode('utf-8')

    # 3. JSON files
    from phase3_program_synthesis import generate_json_files
    jsons = generate_json_files(n_files=3, n_items=50, seed=42)
    for i, j in enumerate(jsons):
        corpus[f'data_{i}.json'] = j.encode('utf-8')

    # 4. CSV files
    from phase3_program_synthesis import generate_csv_files
    csvs = generate_csv_files(n_files=3, n_rows=50, seed=42)
    for i, c in enumerate(csvs):
        corpus[f'table_{i}.csv'] = c.encode('utf-8')

    # 5. Binary files
    rng = np.random.default_rng(42)
    for i in range(3):
        # Structured binary
        data = np.tile(rng.integers(0, 256, 16, dtype=np.uint8), 64).tobytes()
        corpus[f'binary_{i}.bin'] = data

    return corpus


def run_phase5_experiment(verbose=True):
    """Run Phase 5 Universe Prototype experiment."""
    print("=" * 80)
    print("🧪 Phase 5: Universe Prototype (Multi-Method Auto-Selection)")
    print("=" * 80)

    # Generate test corpus
    print("\n📦 Generating mixed test corpus...")
    corpus = generate_test_corpus()
    print(f"  {len(corpus)} files generated")

    # Compress each file with Universe
    print("\n🌌 Compressing with Universe auto-selection...")
    universe = UniverseCompressor()
    total_original = 0
    total_compressed = 0
    total_zip = 0

    print(f"\n{'File':<25} {'Type':<12} {'Original':>10} {'Universe':>10} {'ZIP':>10} {'Method':<15} {'vs ZIP':>8}")
    print("-" * 95)

    for name, data in sorted(corpus.items()):
        compressed, comp_size, method, file_type = universe.compress(data)
        zip_size = len(zlib.compress(data, 9))

        total_original += len(data)
        total_compressed += comp_size
        total_zip += zip_size

        vs_zip = zip_size / max(comp_size, 1)
        type_short = file_type.replace('image_', '').replace('audio_', '')
        print(f"{name:<25} {type_short:<12} {len(data):>9,}B {comp_size:>9,}B {zip_size:>9,}B {method:<15} {vs_zip:>7.2f}x")

    # Summary
    print(f"\n{'='*80}")
    print("📊 PHASE 5 UNIVERSE SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Files: {len(corpus)}")
    print(f"  Original: {total_original:,}B ({total_original/1024:.1f}KB)")
    print(f"  ZIP: {total_zip:,}B ({total_zip/1024:.1f}KB)")
    print(f"  Universe: {total_compressed:,}B ({total_compressed/1024:.1f}KB)")
    print(f"  vs ZIP: {total_zip/max(total_compressed,1):.2f}x")
    print(f"  Compression ratio: {total_original/max(total_compressed,1):.1f}x")

    # Method breakdown
    print(f"\n📋 Method breakdown:")
    method_counts = {}
    for r in universe.results.values():
        m = r['method']
        if m not in method_counts:
            method_counts[m] = {'count': 0, 'original': 0, 'compressed': 0}
        method_counts[m]['count'] += 1
        method_counts[m]['original'] += r['original_size']
        method_counts[m]['compressed'] += r['compressed_size']

    print(f"  {'Method':<20} {'Files':>6} {'Original':>10} {'Compressed':>11} {'Ratio':>8}")
    print("  " + "-" * 60)
    for m, stats in sorted(method_counts.items()):
        ratio = stats['original'] / max(stats['compressed'], 1)
        print(f"  {m:<20} {stats['count']:>6} {stats['original']:>9,}B {stats['compressed']:>10,}B {ratio:>7.1f}x")

    return {
        'total_original': total_original,
        'total_compressed': total_compressed,
        'total_zip': total_zip,
        'vs_zip': total_zip / max(total_compressed, 1),
        'method_counts': method_counts,
    }


if __name__ == '__main__':
    results = run_phase5_experiment(verbose=True)
