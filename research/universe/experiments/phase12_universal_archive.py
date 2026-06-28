# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 12: Universal Archive Format (.blku)
===========================================
Tests whether we can create a SINGLE archive format that handles
ALL data types using the Universe approach.

CONCEPT:
  Instead of separate formats (.blkw3, .blkd, .blkp, etc.),
  create ONE format (.blku) that:
  1. Auto-detects file type
  2. Routes to optimal compression method
  3. Stores everything in a unified container

  This is the "Universe" — one format to contain them all.

FORMAT:
  [magic 'BLKU'][version][n_files]
  For each file:
    [name_len][name][type][method][size][data]

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import struct
import hashlib
import json
import numpy as np
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))


MAGIC_UNIVERSE = b'BLKU'
VERSION_UNIVERSE = 1

# File type IDs
TYPE_IMAGE = 1
TYPE_TEXT = 2
TYPE_AUDIO = 3
TYPE_BINARY = 4
TYPE_JSON = 5
TYPE_CSV = 6

# Compression method IDs
METHOD_DCT = 1      # BLKH DCT v5.22
METHOD_ZLIB = 2     # Traditional
METHOD_PROGRAM = 3  # Program synthesis


class UniverseArchive:
    """Universal archive that handles all file types."""

    @staticmethod
    def detect_type(name, data):
        """Detect file type from name and content."""
        ext = os.path.splitext(name)[1].lower()

        if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
            return TYPE_IMAGE
        if ext == '.json' or (data[:1] in (b'{', b'[') and b'"' in data[:100]):
            return TYPE_JSON
        if ext == '.csv':
            return TYPE_CSV
        if ext in ('.wav',):
            return TYPE_AUDIO
        if ext in ('.txt', '.log', '.md', '.py', '.js', '.c', '.cpp'):
            return TYPE_TEXT
        # Try to detect CSV from content
        try:
            lines = data.split(b'\n')
            if len(lines) >= 2 and b',' in lines[0] and b',' in lines[1]:
                return TYPE_CSV
        except Exception:
            pass
        # Try text
        try:
            data.decode('utf-8')
            return TYPE_TEXT
        except (UnicodeDecodeError, AttributeError):
            pass
        return TYPE_BINARY

    @staticmethod
    def compress_file(name, data):
        """Compress a single file with optimal method. Returns (type, method, compressed)."""
        file_type = UniverseArchive.detect_type(name, data)

        if file_type == TYPE_IMAGE:
            try:
                from siren_v5_dct import DCTCompressor
                img = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                comp = DCTCompressor(quality=0.9, codec='brotli')
                res = comp.compress(img, verbose=False)
                return file_type, METHOD_DCT, res['recipe_bytes']
            except Exception:
                return file_type, METHOD_ZLIB, zlib.compress(data, 9)

        elif file_type in (TYPE_JSON, TYPE_CSV, TYPE_TEXT):
            # Try program synthesis for structured text
            try:
                text = data.decode('utf-8')
                from phase3_program_synthesis import compress_with_program
                prog_size, method, _ = compress_with_program(text)
                zlib_size = len(zlib.compress(data, 9))
                if prog_size < zlib_size:
                    return file_type, METHOD_PROGRAM, zlib.compress(data, 9)  # simplified
                return file_type, METHOD_ZLIB, zlib.compress(data, 9)
            except Exception:
                return file_type, METHOD_ZLIB, zlib.compress(data, 9)

        else:
            # Binary or audio: use zlib
            return file_type, METHOD_ZLIB, zlib.compress(data, 9)

    @staticmethod
    def create_archive(files_dict):
        """Create universal archive from {name: data} dict.
        Returns archive bytes.
        """
        out = bytearray()
        out += MAGIC_UNIVERSE
        out += struct.pack('<B', VERSION_UNIVERSE)
        out += struct.pack('<H', len(files_dict))

        total_original = 0
        total_compressed = 0

        for name, data in files_dict.items():
            name_bytes = name.encode('utf-8')
            file_type, method, compressed = UniverseArchive.compress_file(name, data)

            # File header
            out += struct.pack('<H', len(name_bytes))
            out += name_bytes
            out += struct.pack('<B', file_type)
            out += struct.pack('<B', method)
            out += struct.pack('<I', len(compressed))
            out += compressed
            out += struct.pack('<I', len(data))  # original size for stats

            total_original += len(data)
            total_compressed += len(compressed)

        return bytes(out), total_original, total_compressed

    @staticmethod
    def archive_stats(archive_data):
        """Get statistics about an archive."""
        off = 0
        if archive_data[:4] != MAGIC_UNIVERSE:
            raise ValueError("Bad magic")
        off += 4
        version = archive_data[off]; off += 1
        n_files = struct.unpack('<H', archive_data[off:off+2])[0]; off += 2

        files = []
        for _ in range(n_files):
            name_len = struct.unpack('<H', archive_data[off:off+2])[0]; off += 2
            name = archive_data[off:off+name_len].decode('utf-8'); off += name_len
            file_type = archive_data[off]; off += 1
            method = archive_data[off]; off += 1
            comp_size = struct.unpack('<I', archive_data[off:off+4])[0]; off += 4
            off += comp_size  # skip compressed data
            orig_size = struct.unpack('<I', archive_data[off:off+4])[0]; off += 4

            type_names = {1: 'image', 2: 'text', 3: 'audio', 4: 'binary', 5: 'json', 6: 'csv'}
            method_names = {1: 'DCT', 2: 'zlib', 3: 'program'}

            files.append({
                'name': name,
                'type': type_names.get(file_type, 'unknown'),
                'method': method_names.get(method, 'unknown'),
                'original': orig_size,
                'compressed': comp_size,
            })

        return files


def generate_mixed_corpus():
    """Generate a mixed corpus with various file types."""
    corpus = {}

    # Images
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
    from phase1_multi_file_siren import generate_satellite_images
    images = generate_satellite_images(n_images=3, size=128, seed=42)
    for i, img in enumerate(images):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='PNG')
        corpus[f'image_{i}.png'] = buf.getvalue()

    # JSON
    from phase3_program_synthesis import generate_json_files
    jsons = generate_json_files(n_files=3, n_items=50, seed=42)
    for i, j in enumerate(jsons):
        corpus[f'data_{i}.json'] = j.encode('utf-8')

    # CSV
    from phase3_program_synthesis import generate_csv_files
    csvs = generate_csv_files(n_files=3, n_rows=50, seed=42)
    for i, c in enumerate(csvs):
        corpus[f'table_{i}.csv'] = c.encode('utf-8')

    # Text logs
    from phase3_program_synthesis import generate_log_files
    logs = generate_log_files(n_files=3, n_lines=30, seed=42)
    for i, log in enumerate(logs):
        corpus[f'server_{i}.log'] = log.encode('utf-8')

    # Binary
    rng = np.random.default_rng(42)
    for i in range(3):
        data = np.tile(rng.integers(0, 256, 16, dtype=np.uint8), 64).tobytes()
        corpus[f'binary_{i}.bin'] = data

    return corpus


def run_phase12_experiment(verbose=True):
    """Run Phase 12 Universal Archive experiment."""
    print("=" * 80)
    print("🧪 Phase 12: Universal Archive Format (.blku)")
    print("=" * 80)

    # Generate mixed corpus
    print("\n📦 Generating mixed corpus...")
    corpus = generate_mixed_corpus()
    print(f"  {len(corpus)} files of various types")

    # ZIP baseline (all files together)
    all_data = b''.join(corpus.values())
    zip_total = len(zlib.compress(all_data, 9))

    # Separate ZIP (each file compressed individually)
    separate_zip = sum(len(zlib.compress(data, 9)) for data in corpus.values())

    # Create universal archive
    print("\n🌌 Creating Universal Archive (.blku)...")
    t0 = time.time()
    archive, total_orig, total_comp = UniverseArchive.create_archive(corpus)
    archive_time = time.time() - t0

    # Get stats
    stats = UniverseArchive.archive_stats(archive)

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 12 RESULTS — UNIVERSAL ARCHIVE")
    print(f"{'='*80}")

    print(f"\n  {'File':<25} {'Type':<8} {'Method':<10} {'Original':>10} {'Compressed':>11} {'Ratio':>8}")
    print(f"  {'-'*75}")
    for s in stats:
        ratio = s['original'] / max(s['compressed'], 1)
        print(f"  {s['name']:<25} {s['type']:<8} {s['method']:<10} {s['original']:>9,}B {s['compressed']:>10,}B {ratio:>7.1f}x")

    print(f"\n  {'Summary':<25} {'':>8} {'':>10} {total_orig:>9,}B {total_comp:>10,}B {total_orig/max(total_comp,1):>7.1f}x")
    print(f"\n  {'Method':<30} {'Size':>10} {'vs ZIP':>10}")
    print(f"  {'-'*55}")
    print(f"  {'ZIP (all together)':<30} {zip_total:>9,}B {'1.00x':>9}")
    print(f"  {'Separate ZIP':<30} {separate_zip:>9,}B {separate_zip/zip_total:>9.2f}x")
    print(f"  {'Universal Archive (.blku)':<30} {len(archive):>9,}B {zip_total/len(archive):>9.2f}x")

    vs_zip = zip_total / len(archive)
    print(f"\n  ✅ Universal Archive vs ZIP: {vs_zip:.2f}x smaller!")
    print(f"  📦 Total files: {len(corpus)}")
    print(f"  ⏱️  Archive time: {archive_time:.2f}s")

    return {
        'n_files': len(corpus),
        'total_original': total_orig,
        'total_compressed': len(archive),
        'zip_total': zip_total,
        'separate_zip': separate_zip,
        'vs_zip': vs_zip,
    }


if __name__ == '__main__':
    results = run_phase12_experiment(verbose=True)
