#!/usr/bin/env python3
"""
blkh — unified Black Hole CLI
=============================
Replaces the legacy compress.py / decompress.py (which used siren_core.py v1,
giving 194KB recipes). This CLI uses v5 (PyTorch backend) by default and
produces bit-perfect recipes.

Usage:
    # Compress any file (image or binary) into a .blkh5 recipe
    python blkh compress input.png output.blkh5

    # Decompress (recover original bytes, SHA-256 verified)
    python blkh decompress output.blkh5 recovered.png

    # Benchmark vs ZIP on a file
    python blkh benchmark input.png

    # Show info about a recipe
    python blkh info output.blkh5

Options:
    --epochs N          training steps (default 1500)
    --bits 4|8          quantization (default 8 = better quality)
    --hidden N          hidden features (default 32)
    --layers N          hidden layers (default 2)
    --omega N           omega_0 (default 30.0)
    --no-bit-perfect    lossy mode (smaller, not exact)
    --batch-size N      mini-batch size (default 2048)
"""
import sys
import os
import zlib
import json
import hashlib
import argparse
import time
from pathlib import Path

# Add phase1 to path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'phase1_inr_compressor'))


def is_image_file(path: str) -> bool:
    """Detect image files by extension."""
    ext = Path(path).suffix.lower()
    return ext in {'.png', '.jpg', '.jpeg', '.bmp', '.pgm', '.ppm', '.tif', '.tiff'}


def load_image(path: str):
    """Load image as (H, W, 3) uint8 numpy array."""
    try:
        from PIL import Image
        img = Image.open(path).convert('RGB')
        import numpy as np
        return np.array(img, dtype=np.uint8)
    except ImportError:
        # Fallback: read raw bytes and reshape assuming square RGB
        import numpy as np
        data = Path(path).read_bytes()
        side = int(len(data) ** 0.5 / 3) ** 0.5
        # Best effort — user should install Pillow for proper image support
        raise SystemExit("Pillow not installed. Run: pip install Pillow")


def load_any(path: str):
    """Load file as either image (H,W,3) uint8 or treat raw bytes as 1D signal."""
    if is_image_file(path):
        return ('image', load_image(path))
    # Treat as 1D byte stream — pack into square image shape for SIREN 2D
    import numpy as np
    data = Path(path).read_bytes()
    # Find smallest square that fits
    n = len(data)
    side = int((n + 2) ** 0.5) + 1  # pad to square
    # Replicate grayscale into 3 channels to fit the model's expected 3-channel output
    arr = np.zeros((side, side, 3), dtype=np.uint8)
    flat = np.frombuffer(data, dtype=np.uint8)
    arr.reshape(-1)[:n] = flat[:, None].repeat(3, axis=1).reshape(-1)[:n]
    return ('binary', (arr, n, data))


def cmd_compress(args):
    import numpy as np
    from siren_v5_torch import ImageINRv5

    kind, payload = load_any(args.input)
    if kind == 'image':
        img = payload
        orig_nbytes = img.nbytes
    else:
        img, orig_nbytes, orig_bytes = payload

    print(f"[BLKH] Input: {args.input}  ({kind}, {orig_nbytes:,} bytes)")
    print(f"[BLKH] Config: epochs={args.epochs} bits={args.bits} "
          f"hidden={args.hidden} layers={args.layers} omega={args.omega}")

    comp = ImageINRv5(hidden_features=args.hidden,
                      hidden_layers=args.layers,
                      omega_0=args.omega)

    t0 = time.time()
    if args.no_bit_perfect:
        # Lossy mode (no residual)
        meta = comp.compress(img, epochs=args.epochs, lr=1e-3,
                             batch_size=args.batch_size, verbose=True)
        # Save just the SIREN weights (need to quantize manually)
        from siren_v5_torch import quantize_int8, quantize_int4
        W = comp._model.state_to_numpy()
        if args.bits == 8:
            packed, pm = quantize_int8(W)
        else:
            packed, pm = quantize_int4(W)
        # Use bit-perfect with empty residual as a way to reuse the format
        sha = hashlib.sha256(img.tobytes()).digest()
        recipe = comp._pack_recipe(args.bits, packed, pm, b'', sha)
        # NOTE: in lossy mode, decompress will produce predicted-only bytes
        # (residual is empty), so the SHA won't match. That's expected.
        print(f"[BLKH] Lossy mode: PSNR={meta['psnr']:.2f} dB  "
              f"(recipe will NOT roundtrip bit-perfect)")
    else:
        res = comp.compress_bitperfect(img, epochs=args.epochs, lr=1e-3,
                                        bits=args.bits, prune_threshold=0.0,
                                        batch_size=args.batch_size, verbose=True)
        recipe = res['recipe_bytes']
        print(f"[BLKH] Bit-perfect: bit acc={res['model_bit_accuracy']:.1f}%  "
              f"PSNR={res['psnr_db']:.2f} dB  SHA-256={res['sha256'][:16]}...")

    dt = time.time() - t0
    Path(args.output).write_bytes(recipe)
    recipe_size = len(recipe)

    # ZIP comparison
    if kind == 'image':
        zip_size = len(zlib.compress(img.tobytes(), 9))
    else:
        zip_size = len(zlib.compress(orig_bytes if kind == 'binary' else img.tobytes(), 9))

    print(f"\n[BLKH] Result:")
    print(f"  Original:    {orig_nbytes:>10,} B")
    print(f"  ZIP (zlib9): {zip_size:>10,} B  (ratio {orig_nbytes/zip_size:.2f}x)")
    print(f"  BLKH:        {recipe_size:>10,} B  (ratio {orig_nbytes/recipe_size:.2f}x)")
    winner = "BLKH" if recipe_size < zip_size else "ZIP"
    print(f"  Winner:      {winner}  (BLKH/ZIP = {recipe_size/zip_size:.3f})")
    print(f"  Time:        {dt:.2f}s")
    print(f"  Output:      {args.output}")


def cmd_decompress(args):
    import numpy as np
    from siren_v5_torch import ImageINRv5

    recipe = Path(args.input).read_bytes()
    t0 = time.time()
    img, meta = ImageINRv5.decompress(recipe)
    dt = time.time() - t0

    if args.output:
        if is_image_file(args.output):
            # Save as proper image file
            try:
                from PIL import Image
                Image.fromarray(img).save(args.output)
            except ImportError:
                Path(args.output).write_bytes(img.tobytes())
        else:
            # Save raw bytes
            Path(args.output).write_bytes(img.tobytes())
        print(f"[BLKH] Decompressed: {args.output}")
    else:
        sys.stdout.buffer.write(img.tobytes())

    print(f"[BLKH] Shape: {img.shape}  Time: {dt*1000:.1f}ms", file=sys.stderr)
    print(f"[BLKH] SHA-256 match: {meta['exact_match']}", file=sys.stderr)
    if not meta['exact_match']:
        print(f"[BLKH] WARNING: roundtrip failed! Original bytes not recovered.", file=sys.stderr)
        sys.exit(1)


def cmd_info(args):
    import struct
    from siren_v5_torch import MAGIC_V5
    recipe = Path(args.input).read_bytes()
    if recipe[:4] == MAGIC_V5:
        print(f"[BLKH] Format: BLK5 (v5 PyTorch)")
        print(f"[BLKH] Size: {len(recipe):,} bytes")
        # Parse header
        version = recipe[4]
        bits = recipe[5]
        in_f = recipe[6]
        hidden = struct.unpack('<H', recipe[7:9])[0]
        hidden_l = recipe[9]
        out_f = recipe[10]
        omega = struct.unpack('<f', recipe[11:15])[0]
        H = struct.unpack('<H', recipe[15:17])[0]
        W = struct.unpack('<H', recipe[17:19])[0]
        C = recipe[19]
        print(f"  version:    {version}")
        print(f"  bits:       {bits}")
        print(f"  arch:       in={in_f} hidden={hidden} layers={hidden_l} out={out_f}")
        print(f"  omega_0:    {omega}")
        print(f"  image:      {H}x{W}x{C}")
    else:
        print(f"[BLKH] Unknown format (magic: {recipe[:4]!r})")


def cmd_benchmark(args):
    import numpy as np
    from siren_v5_torch import ImageINRv5

    kind, payload = load_any(args.input)
    if kind == 'image':
        img = payload
    else:
        img, _, _ = payload

    orig = img.nbytes
    zip_size = len(zlib.compress(img.tobytes(), 9))
    print(f"[BLKH] Benchmark: {args.input}")
    print(f"  Original:    {orig:,} B")
    print(f"  ZIP (zlib9): {zip_size:,} B  (ratio {orig/zip_size:.2f}x)")

    for epochs in [500, 1500, 3000]:
        comp = ImageINRv5()
        t0 = time.time()
        res = comp.compress_bitperfect(img, epochs=epochs, lr=1e-3,
                                        bits=8, batch_size=2048, verbose=False)
        dt = time.time() - t0
        winner = "BLKH" if res['recipe_size'] < zip_size else "ZIP"
        print(f"  BLKH e={epochs:4d}: {res['recipe_size']:>6,} B  "
              f"bit%={res['model_bit_accuracy']:.1f}  "
              f"PSNR={res['psnr_db']:.1f}  {dt:.1f}s  {winner}")


def main():
    p = argparse.ArgumentParser(prog='blkh',
                                 description='Black Hole — Neural Implicit Compression')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_c = sub.add_parser('compress', help='Compress a file into a .blkh5 recipe')
    p_c.add_argument('input')
    p_c.add_argument('output')
    p_c.add_argument('--epochs', type=int, default=1500)
    p_c.add_argument('--bits', type=int, default=8, choices=[4, 8])
    p_c.add_argument('--hidden', type=int, default=32)
    p_c.add_argument('--layers', type=int, default=2)
    p_c.add_argument('--omega', type=float, default=30.0)
    p_c.add_argument('--batch-size', type=int, default=2048)
    p_c.add_argument('--no-bit-perfect', action='store_true',
                     help='Lossy mode (no residual, ~3x smaller but NOT exact)')
    p_c.set_defaults(func=cmd_compress)

    p_d = sub.add_parser('decompress', help='Recover original file from .blkh5')
    p_d.add_argument('input')
    p_d.add_argument('output', nargs='?', default=None,
                     help='Output file (default: stdout)')
    p_d.set_defaults(func=cmd_decompress)

    p_i = sub.add_parser('info', help='Show info about a .blkh5 recipe')
    p_i.add_argument('input')
    p_i.set_defaults(func=cmd_info)

    p_b = sub.add_parser('benchmark', help='Benchmark BLKH vs ZIP on a file')
    p_b.add_argument('input')
    p_b.set_defaults(func=cmd_benchmark)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
