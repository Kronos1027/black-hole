# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

# BLKH-AUTH-DPS-2025-Kronos1027-darlan1027pc@gmail.com
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


def cmd_doctor(args):
    """Diagnose the BLKH environment and show recommendations."""
    import torch
    import platform

    print("=" * 60)
    print("  BLKH Doctor — Environment Diagnostics")
    print("=" * 60)

    # Python
    print(f"\n  Python:       {platform.python_version()}")
    print(f"  Platform:     {platform.platform()}")

    # PyTorch
    print(f"\n  PyTorch:      {torch.__version__}")
    print(f"  CUDA:         {'available' if torch.cuda.is_available() else 'not available (CPU mode)'}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory:   {gpu_mem:.1f} GB")
    print(f"  CPU threads:  {torch.get_num_threads()}")

    # Dependencies
    deps = []
    try:
        import numpy; deps.append(f'numpy {numpy.__version__}')
    except ImportError:
        deps.append('numpy MISSING')
    try:
        from PIL import Image; deps.append('Pillow OK')
    except ImportError:
        deps.append('Pillow MISSING (pip install Pillow)')
    try:
        import scipy; deps.append(f'scipy {scipy.__version__}')
    except ImportError:
        deps.append('scipy MISSING (pip install scipy)')
    try:
        import psutil; deps.append('psutil OK')
    except ImportError:
        deps.append('psutil MISSING (pip install psutil)')
    try:
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new('RGB', (10,10)).save(buf, format='WebP', lossless=True)
        deps.append('WebP support OK')
    except Exception:
        deps.append('WebP support MISSING (install libwebp)')

    print(f"\n  Dependencies:")
    for d in deps:
        print(f"    {d}")

    # Recommendations
    print(f"\n  Recommendations:")
    if not torch.cuda.is_available():
        print(f"    → CPU mode: use --amp for 1.5x speedup")
        print(f"    → Use --patience 5 for 2x speedup (early stopping)")
    else:
        print(f"    → GPU detected: AMP auto-enabled, expect 10-50x speedup")
    print(f"    → Use --auto-tune for optimal SIREN size (recommended)")
    print(f"    → Use --patience 5 to cut encoding time in half")

    # Quick test
    print(f"\n  Quick test (16x16 gradient)...")
    try:
        import numpy as np
        from siren_v5_hybrid import HybridCompressor
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        for i in range(16):
            for j in range(16):
                img[i,j] = [i*15, j*15, (i+j)*8]
        comp = HybridCompressor(hidden_features=16, hidden_layers=1, residual_codec='webp')
        res = comp.compress_bitperfect(img, epochs=100, lr=1e-3, bits=8,
                                         batch_size=64, verbose=False)
        rec, meta = HybridCompressor.decompress(res['recipe_bytes'])
        if meta['exact_match']:
            print(f"    ✓ Roundtrip OK (SHA-256 verified)")
        else:
            print(f"    ✗ Roundtrip FAILED")
    except Exception as e:
        print(f"    ✗ Test failed: {e}")

    print(f"\n{'=' * 60}")
    print(f"  BLKH is {'ready' if all('MISSING' not in d for d in deps) else 'partially ready'}.")
    print(f"{'=' * 60}")


def cmd_compress(args):
    import numpy as np
    from siren_v5_torch import ImageINRv5
    from siren_v5_hybrid import HybridCompressor

    kind, payload = load_any(args.input)
    if kind == 'image':
        img = payload
        orig_nbytes = img.nbytes
    else:
        img, orig_nbytes, orig_bytes = payload

    # Turbo/instant mode overrides
    if getattr(args, 'turbo', False):
        args.epochs = 200
        args.batch_size = 16384
        if not getattr(args, 'auto_tune', False):
            args.auto_tune = True
        if not getattr(args, 'amp', False):
            args.amp = True
        args.lr_override = 3e-3
        args.codec_override = 'webp'
    elif getattr(args, 'instant', False):
        args.epochs = 100
        args.batch_size = 16384
        if not getattr(args, 'auto_tune', False):
            args.auto_tune = True
        if not getattr(args, 'amp', False):
            args.amp = True
        args.lr_override = 4e-3
        args.codec_override = 'png'  # PNG is 16x faster to encode than WebP
    else:
        args.lr_override = None
        args.codec_override = None

    print(f"[BLKH] Input: {args.input}  ({kind}, {orig_nbytes:,} bytes)")
    if getattr(args, 'instant', False):
        print(f"[BLKH] INSTANT mode: 100 epochs, lr=4e-3, bs=16384, PNG residual, AMP (~0.4-0.7s)")
    elif getattr(args, 'turbo', False):
        print(f"[BLKH] TURBO mode: 200 epochs, lr=3e-3, bs=16384, auto-tune, AMP (~1s)")
    elif getattr(args, 'auto_tune', False):
        print(f"[BLKH] Config: epochs={args.epochs} bits={args.bits} "
              f"auto-tune=ON (SIREN size picked from image dims) "
              f"amp={getattr(args, 'amp', False)}")
    else:
        print(f"[BLKH] Config: epochs={args.epochs} bits={args.bits} "
              f"hidden={args.hidden} layers={args.layers} omega={args.omega} "
              f"amp={getattr(args, 'amp', False)}")

    t0 = time.time()
    if args.no_bit_perfect:
        # Lossy mode (no residual)
        comp = ImageINRv5(hidden_features=args.hidden,
                          hidden_layers=args.layers,
                          omega_0=args.omega)
        meta = comp.compress(img, epochs=args.epochs, lr=getattr(args, "lr_override", None) or 1e-3,
                             batch_size=args.batch_size,
                             use_amp=getattr(args, 'amp', False),
                             patience=getattr(args, 'patience', 0),
                             verbose=True)
        from siren_v5_torch import quantize_int8, quantize_int4
        W = comp._model.state_to_numpy()
        if args.bits == 8:
            packed, pm = quantize_int8(W)
        else:
            packed, pm = quantize_int4(W)
        sha = hashlib.sha256(img.tobytes()).digest()
        recipe = comp._pack_recipe(args.bits, packed, pm, b'', sha)
        print(f"[BLKH] Lossy mode: PSNR={meta['psnr']:.2f} dB  "
              f"(recipe will NOT roundtrip bit-perfect)")
    elif getattr(args, 'auto_tune', False) or getattr(args, 'instant', False) or getattr(args, 'turbo', False):
        # Hybrid mode with auto-tune (recommended for images)
        codec = getattr(args, 'codec_override', None) or 'webp'
        comp = HybridCompressor(auto_tune=True, omega_0=args.omega,
                                 residual_codec=codec)
        res = comp.compress_bitperfect(img, epochs=args.epochs, lr=getattr(args, "lr_override", None) or 1e-3,
                                        bits=args.bits, prune_threshold=0.0,
                                        batch_size=args.batch_size,
                                        use_amp=getattr(args, 'amp', False),
                                        patience=getattr(args, 'patience', 0),
                                        verbose=True)
        recipe = res['recipe_bytes']
        print(f"[BLKH] Bit-perfect (hybrid+auto-tune): bit acc={res['model_bit_accuracy']:.1f}%  "
              f"PSNR={res['psnr_db']:.2f} dB  SHA-256={res['sha256'][:16]}...")
    else:
        comp = ImageINRv5(hidden_features=args.hidden,
                          hidden_layers=args.layers,
                          omega_0=args.omega)
        res = comp.compress_bitperfect(img, epochs=args.epochs, lr=getattr(args, "lr_override", None) or 1e-3,
                                        bits=args.bits, prune_threshold=0.0,
                                        batch_size=args.batch_size,
                                        use_amp=getattr(args, 'amp', False),
                                        patience=getattr(args, 'patience', 0),
                                        verbose=True)
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
    from siren_v5_hybrid import HybridCompressor

    recipe = Path(args.input).read_bytes()
    magic = recipe[:4]

    t0 = time.time()
    # Auto-detect format by magic bytes
    if magic == b'BLK8':
        # Hybrid format (.blkh8) — use HybridCompressor
        img, meta = HybridCompressor.decompress(recipe)
    elif magic == b'BLK5':
        # v5 format (.blkh5) — use ImageINRv5
        img, meta = ImageINRv5.decompress(recipe)
    elif magic == b'BLKV':
        # Video format — not supported via CLI decompress (use video-decompress)
        print("[BLKH] Error: this is a video recipe. Use 'blkh video-decompress' instead.")
        sys.exit(1)
    elif magic == b'BLK3':
        # 3D volume format — not supported via CLI decompress (use volume-decompress)
        print("[BLKH] Error: this is a 3D volume recipe. Use 'blkh volume-decompress' instead.")
        sys.exit(1)
    elif magic == b'BLK9':
        # Combo format — not supported via CLI decompress (use combo-decompress)
        print("[BLKH] Error: this is a combo recipe. Use 'blkh combo-decompress' instead.")
        sys.exit(1)
    else:
        # Try ImageINRv5 as fallback
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
    if meta.get('mode') == 'lossy':
        print(f"[BLKH] Mode: LOSSY (no SHA-256 match expected — not bit-perfect)", file=sys.stderr)
    else:
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


def cmd_atlas(args):
    """Compress N similar images into a shared .bla5 atlas recipe."""
    import numpy as np
    from siren_v5_atlas import AtlasCompressor

    inputs = args.inputs
    if len(inputs) < 2:
        print("[BLKH] Atlas needs at least 2 input images.")
        sys.exit(1)

    images = []
    for path in inputs:
        try:
            from PIL import Image
            img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
            images.append(img)
            print(f"[BLKH] Loaded: {path} ({img.shape})")
        except Exception as e:
            print(f"[BLKH] Failed to load {path}: {e}")
            sys.exit(1)

    total_orig = sum(im.nbytes for im in images)
    zip_total = sum(len(zlib.compress(im.tobytes(), 9)) for im in images)
    print(f"\n[BLKH] Atlas: {len(images)} images, total {total_orig:,}B")
    print(f"[BLKH] ZIP per-file total: {zip_total:,}B ({total_orig/zip_total:.2f}x)")
    print(f"[BLKH] Config: epochs={args.epochs} hidden={args.hidden} layers={args.layers}")

    comp = AtlasCompressor(hidden_features=args.hidden,
                            hidden_layers=args.layers,
                            omega_0=args.omega)
    t0 = time.time()
    res = comp.compress(images, epochs=args.epochs, lr=1e-3,
                         bits=args.bits, batch_size=4096, verbose=True)
    dt = time.time() - t0

    Path(args.output).write_bytes(res['recipe_bytes'])
    winner = "BLKH" if res['recipe_size'] < zip_total else "ZIP"
    print(f"\n[BLKH] Atlas result:")
    print(f"  Original:    {total_orig:>10,} B")
    print(f"  ZIP:         {zip_total:>10,} B  (ratio {total_orig/zip_total:.2f}x)")
    print(f"  BLKH Atlas:  {res['recipe_size']:>10,} B  (ratio {total_orig/res['recipe_size']:.2f}x)")
    print(f"    weights (shared): {res['weights_packed_size']:,} B "
          f"-> {res['weights_packed_size']/len(images):.0f} B/image amortized")
    print(f"    residuals:        {res['residual_total']:,} B "
          f"-> {res['residual_per_image']:.0f} B/image")
    print(f"  Bit acc avg: {res['avg_bit_pct']:.1f}%")
    print(f"  Winner: {winner}  (BLKH/ZIP = {res['recipe_size']/zip_total:.3f})")
    print(f"  Time: {dt:.1f}s")
    print(f"  Output: {args.output}")


def cmd_atlas_decompress(args):
    """Decompress a .bla5 atlas recipe into N images."""
    import numpy as np
    from siren_v5_atlas import AtlasCompressor

    recipe = Path(args.input).read_bytes()
    t0 = time.time()
    images, meta = AtlasCompressor.decompress(recipe)
    dt = time.time() - t0
    print(f"[BLKH] Atlas: decompressed {meta['n_images']} images in {dt*1000:.0f}ms")
    print(f"  All SHA-256 match: {meta['all_sha256_match']}")
    if not meta['all_sha256_match']:
        print(f"  WARNING: some images failed SHA-256 verification!")
        sys.exit(1)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            for i, img in enumerate(images):
                Image.fromarray(img).save(out_dir / f"recovered_{i:03d}.png")
            print(f"  Saved {len(images)} images to: {out_dir}")
        except ImportError:
            for i, img in enumerate(images):
                (out_dir / f"recovered_{i:03d}.raw").write_bytes(img.tobytes())
            print(f"  Saved {len(images)} raw files to: {out_dir} (install Pillow for PNG)")


def cmd_combo(args):
    """Compress N similar images using v5.9 Combo (hypernetwork + WebP residual).
    Best mode for datacenter: 2-3x smaller than ZIP on smooth image corpora.
    """
    import numpy as np
    from siren_v5_combo import ComboCompressor

    inputs = args.inputs
    if len(inputs) < 2:
        print("[BLKH] Combo needs at least 2 input images.")
        sys.exit(1)

    images = []
    for path in inputs:
        try:
            from PIL import Image
            img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
            images.append(img)
            print(f"[BLKH] Loaded: {path} ({img.shape})")
        except Exception as e:
            print(f"[BLKH] Failed to load {path}: {e}")
            sys.exit(1)

    total_orig = sum(im.nbytes for im in images)
    zip_total = sum(len(zlib.compress(im.tobytes(), 9)) for im in images)
    print(f"\n[BLKH] Combo: {len(images)} images, total {total_orig:,}B")
    print(f"[BLKH] ZIP per-file total: {zip_total:,}B ({total_orig/zip_total:.2f}x)")
    print(f"[BLKH] Config: latent={args.latent} hidden={args.hidden} layers={args.layers} "
          f"codec={args.codec} base_epochs={args.base_epochs} compress_epochs={args.compress_epochs}")

    comp = ComboCompressor(latent_dim=args.latent,
                            hidden_features=args.hidden,
                            hidden_layers=args.layers,
                            omega_0=args.omega,
                            residual_codec=args.codec)
    t0 = time.time()
    print(f"\n[BLKH] Phase 1: training shared hypernetwork...")
    comp.train_base(images, epochs=args.base_epochs, lr=1e-3,
                     batch_size=2048, verbose=True)
    print(f"\n[BLKH] Phase 2: compressing {len(images)} images (latent + {args.codec} residual)...")
    res = comp.compress_many(images, epochs=args.compress_epochs, lr=3e-3,
                              bits=8, batch_size=2048, verbose=False)
    dt = time.time() - t0

    Path(args.output).write_bytes(res['recipe_bytes'])
    winner = "BLKH" if res['recipe_size'] < zip_total else "ZIP"
    print(f"\n[BLKH] Combo result:")
    print(f"  Original:    {total_orig:>10,} B")
    print(f"  ZIP:         {zip_total:>10,} B  (ratio {total_orig/zip_total:.2f}x)")
    print(f"  BLKH Combo:  {res['recipe_size']:>10,} B  (ratio {total_orig/res['recipe_size']:.2f}x)")
    print(f"    hyper (shared): {res['hyper_size']:,} B "
          f"-> {res['hyper_size']/len(images):.0f} B/image amortized")
    print(f"    latent/img:     {res['latent_per_image']} B (INT8)")
    print(f"    residual/img:   {res['residual_per_image']:.0f} B ({res['residual_codec']})")
    print(f"  Bit acc avg: {res['avg_bit_pct']:.1f}%")
    print(f"  Winner: {winner}  (BLKH/ZIP = {res['recipe_size']/zip_total:.3f})")
    print(f"  Time: {dt:.1f}s ({res['per_image_time_s']:.2f}s/image)")
    print(f"  Output: {args.output}")


def cmd_combo_decompress(args):
    """Decompress a .blkh9 combo recipe into N images."""
    import numpy as np
    from siren_v5_combo import ComboCompressor

    recipe = Path(args.input).read_bytes()
    t0 = time.time()
    images, meta = ComboCompressor.decompress(recipe)
    dt = time.time() - t0
    print(f"[BLKH] Combo: decompressed {meta['n_images']} images in {dt*1000:.0f}ms")
    print(f"  Mode: {meta['mode']}  codec: {meta['residual_codec']}")
    print(f"  All SHA-256 match: {meta['all_sha256_match']}")
    if not meta['all_sha256_match']:
        print(f"  WARNING: some images failed SHA-256 verification!")
        sys.exit(1)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            for i, img in enumerate(images):
                Image.fromarray(img).save(out_dir / f"recovered_{i:03d}.png")
            print(f"  Saved {len(images)} images to: {out_dir}")
        except ImportError:
            for i, img in enumerate(images):
                (out_dir / f"recovered_{i:03d}.raw").write_bytes(img.tobytes())
            print(f"  Saved {len(images)} raw files to: {out_dir}")


def cmd_video(args):
    """Compress N video frames using v5.11 temporal SIREN.
    Input: directory of PNG frames (sorted by name) OR explicit file list.
    """
    import numpy as np
    from siren_v5_video import VideoCompressor

    # Collect input frames
    inputs = args.inputs
    if len(inputs) == 1 and Path(inputs[0]).is_dir():
        # Directory mode: load all PNGs sorted
        d = Path(inputs[0])
        inputs = sorted([str(f) for f in d.glob('*.png')])
        print(f"[BLKH] Video: loaded {len(inputs)} frames from {d}")

    if len(inputs) < 2:
        print("[BLKH] Video needs at least 2 frames (directory or file list).")
        sys.exit(1)

    frames = []
    for path in inputs:
        try:
            from PIL import Image
            img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
            frames.append(img)
        except Exception as e:
            print(f"[BLKH] Failed to load {path}: {e}")
            sys.exit(1)

    total_orig = sum(f.nbytes for f in frames)
    zip_total = sum(len(zlib.compress(f.tobytes(), 9)) for f in frames)
    print(f"[BLKH] Video: {len(frames)} frames, total {total_orig:,}B")
    print(f"[BLKH] ZIP per-frame total: {zip_total:,}B ({total_orig/zip_total:.2f}x)")
    print(f"[BLKH] Config: hidden={args.hidden} layers={args.layers} "
          f"codec={args.codec} epochs={args.epochs}")

    comp = VideoCompressor(hidden_features=args.hidden,
                            hidden_layers=args.layers,
                            omega_0=args.omega,
                            omega_t=args.omega_t,
                            residual_codec=args.codec)
    t0 = time.time()
    res = comp.compress(frames, epochs=args.epochs, lr=1e-3,
                          bits=8, batch_size=4096, verbose=True)
    dt = time.time() - t0

    Path(args.output).write_bytes(res['recipe_bytes'])
    winner = "BLKH" if res['recipe_size'] < zip_total else "ZIP"
    print(f"\n[BLKH] Video result:")
    print(f"  Original:    {total_orig:>10,} B ({len(frames)} frames)")
    print(f"  ZIP:         {zip_total:>10,} B  (ratio {total_orig/zip_total:.2f}x)")
    print(f"  BLKH Video:  {res['recipe_size']:>10,} B  (ratio {total_orig/res['recipe_size']:.2f}x)")
    print(f"    SIREN (shared): {res['weights_packed_size']:,} B "
          f"-> {res['weights_packed_size']/len(frames):.0f} B/frame amortized")
    print(f"    residual/frame: {res['residual_per_frame']:.0f} B ({res['residual_codec']})")
    print(f"  Bit acc avg: {res['avg_bit_pct']:.1f}%")
    print(f"  Winner: {winner}  (BLKH/ZIP = {res['recipe_size']/zip_total:.3f})")
    print(f"  Time: {dt:.1f}s")
    print(f"  Output: {args.output}")


def cmd_video_decompress(args):
    """Decompress a .blkv video recipe into N PNG frames."""
    import numpy as np
    from siren_v5_video import VideoCompressor

    recipe = Path(args.input).read_bytes()
    t0 = time.time()
    frames, meta = VideoCompressor.decompress(recipe)
    dt = time.time() - t0
    print(f"[BLKH] Video: decompressed {meta['n_frames']} frames in {dt*1000:.0f}ms")
    print(f"  Mode: {meta['mode']}  codec: {meta['residual_codec']}")
    print(f"  All SHA-256 match: {meta['all_sha256_match']}")
    if not meta['all_sha256_match']:
        print(f"  WARNING: some frames failed SHA-256 verification!")
        sys.exit(1)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            for i, img in enumerate(frames):
                Image.fromarray(img).save(out_dir / f"frame_{i:04d}.png")
            print(f"  Saved {len(frames)} frames to: {out_dir}")
        except ImportError:
            for i, img in enumerate(frames):
                (out_dir / f"frame_{i:04d}.raw").write_bytes(img.tobytes())
            print(f"  Saved {len(frames)} raw files to: {out_dir}")


def cmd_volume(args):
    """Compress a 3D volume (stack of slices as PNG files) using v5.12 SIREN f(x,y,z).
    Input: directory of PNG slices (sorted by name, all same size).
    The slices are stacked into a 3D volume and compressed as ONE SIREN.
    """
    import numpy as np
    from siren_v5_volume import VolumeCompressor

    d = Path(args.input)
    if not d.is_dir():
        print(f"[BLKH] Volume input must be a directory of PNG slices: {args.input}")
        sys.exit(1)
    slices = sorted(d.glob('*.png'))
    if len(slices) < 2:
        print(f"[BLKH] Need at least 2 PNG slices in {d}")
        sys.exit(1)

    print(f"[BLKH] Loading {len(slices)} slices from {d}...")
    from PIL import Image
    slice_imgs = [np.array(Image.open(s).convert('RGB'), dtype=np.uint8) for s in slices]
    H, W, C = slice_imgs[0].shape
    # Stack into (D, H, W, C)
    volume = np.stack(slice_imgs, axis=0)
    D = volume.shape[0]
    print(f"[BLKH] Volume: {volume.shape} = {volume.nbytes:,}B")

    import zlib as _z
    zip_size = len(_z.compress(volume.tobytes(), 9))
    print(f"[BLKH] ZIP: {zip_size:,}B")

    comp = VolumeCompressor(hidden_features=args.hidden, hidden_layers=args.layers,
                              omega_0=args.omega)
    t0 = time.time()
    res = comp.compress(volume, epochs=args.epochs, lr=1e-3,
                          bits=8, batch_size=8192, verbose=True)
    dt = time.time() - t0

    Path(args.output).write_bytes(res['recipe_bytes'])
    winner = "BLKH" if res['recipe_size'] < zip_size else "ZIP"
    print(f"\n[BLKH] Volume result:")
    print(f"  Original:    {volume.nbytes:>10,} B  ({D}x{H}x{W}x{C})")
    print(f"  ZIP:         {zip_size:>10,} B  (ratio {volume.nbytes/zip_size:.2f}x)")
    print(f"  BLKH Volume: {res['recipe_size']:>10,} B  (ratio {volume.nbytes/res['recipe_size']:.2f}x)")
    print(f"    SIREN: {res['weights_packed_size']:,}B  residual: {res['residual_compressed_size']:,}B")
    print(f"  Bit acc: {res['model_bit_accuracy']:.1f}%")
    print(f"  Winner: {winner}  (BLKH/ZIP = {res['recipe_size']/zip_size:.3f})")
    print(f"  Time: {dt:.1f}s")
    print(f"  Output: {args.output}")


def cmd_volume_decompress(args):
    """Decompress a .blk3 volume recipe into PNG slices."""
    import numpy as np
    from siren_v5_volume import VolumeCompressor

    recipe = Path(args.input).read_bytes()
    t0 = time.time()
    volume, meta = VolumeCompressor.decompress(recipe)
    dt = time.time() - t0
    print(f"[BLKH] Volume: decompressed {meta['shape']} in {dt:.2f}s")
    print(f"  SHA-256 match: {meta['exact_match']}")
    if not meta['exact_match']:
        print(f"  WARNING: roundtrip failed!")
        sys.exit(1)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            D, H, W, C = meta['shape']
            for i in range(D):
                Image.fromarray(volume[i]).save(out_dir / f"slice_{i:04d}.png")
            print(f"  Saved {D} slices to: {out_dir}")
        except ImportError:
            (out_dir / "volume.raw").write_bytes(volume.tobytes())
            print(f"  Saved raw volume to: {out_dir}/volume.raw")


def cmd_lossy(args):
    """Compress with BLKH lossy mode (no residual, smaller recipe, NOT bit-perfect).
    Competes with JPEG and WebP lossy.
    """
    import numpy as np
    from siren_v5_torch import ImageINRv5

    kind, payload = load_any(args.input)
    if kind == 'image':
        img = payload
    else:
        img, _, _ = payload

    orig = img.nbytes
    zip_size = len(zlib.compress(img.tobytes(), 9))
    print(f"[BLKH] Lossy mode (no residual — NOT bit-perfect)")
    print(f"[BLKH] Input: {args.input} ({orig:,} bytes)")
    print(f"[BLKH] Config: epochs={args.epochs} bits={args.bits} "
          f"prune={args.prune} hidden={args.hidden} layers={args.layers}")

    comp = ImageINRv5(hidden_features=args.hidden,
                      hidden_layers=args.layers,
                      omega_0=args.omega)
    t0 = time.time()
    res = comp.compress_lossy(img, epochs=args.epochs, lr=1e-3,
                                bits=args.bits, prune_threshold=args.prune,
                                batch_size=args.batch_size,
                                use_amp=getattr(args, 'amp', False),
                                verbose=True)
    dt = time.time() - t0

    Path(args.output).write_bytes(res['recipe_bytes'])

    # JPEG/WebP comparison (if PIL available)
    jpeg_size = webp_size = None
    try:
        import io
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.fromarray(img).save(buf, format='JPEG', quality=85)
        jpeg_size = len(buf.getvalue())
        buf = io.BytesIO()
        PILImage.fromarray(img).save(buf, format='WebP', lossless=False, quality=85)
        webp_size = len(buf.getvalue())
    except Exception:
        pass

    print(f"\n[BLKH] Result:")
    print(f"  Original:    {orig:>10,} B")
    print(f"  ZIP (zlib9): {zip_size:>10,} B  (ratio {orig/zip_size:.2f}x, lossless)")
    if jpeg_size:
        print(f"  JPEG q=85:   {jpeg_size:>10,} B  (ratio {orig/jpeg_size:.2f}x, lossy)")
    if webp_size:
        print(f"  WebP q=85:   {webp_size:>10,} B  (ratio {orig/webp_size:.2f}x, lossy)")
    print(f"  BLKH lossy:  {res['recipe_size']:>10,} B  (ratio {orig/res['recipe_size']:.2f}x, "
          f"PSNR {res['psnr_db']:.1f} dB)")
    print(f"  Time: {dt:.2f}s")
    print(f"  Output: {args.output}")
    print(f"\n  NOTE: Lossy mode is NOT bit-perfect. Use 'blkh compress' for lossless.")


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
    p_c.add_argument('--amp', action='store_true',
                     help='Use mixed precision (bfloat16 on CPU, ~1.5x faster)')
    p_c.add_argument('--auto-tune', action='store_true',
                     help='Auto-pick SIREN size from image dims + use hybrid WebP residual (recommended)')
    p_c.add_argument('--patience', type=int, default=0,
                     help='Early stopping patience (try 5 for ~2x speedup, 0=disabled)')
    p_c.add_argument('--turbo', action='store_true',
                     help='Turbo mode: 200 epochs, lr=3e-3, WebP residual (~1s encoding)')
    p_c.add_argument('--instant', action='store_true',
                     help='Instant mode: 100 epochs, lr=4e-3, PNG residual (~0.4-0.7s, fastest)')
    p_c.add_argument('--no-bit-perfect', action='store_true',
                     help='Lossy mode (no residual, ~3x smaller but NOT exact)')
    p_c.set_defaults(func=cmd_compress)

    p_doc = sub.add_parser('doctor', help='Diagnose environment and show recommendations')
    p_doc.set_defaults(func=cmd_doctor)

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

    p_a = sub.add_parser('atlas', help='Compress N similar images into a shared .bla5 atlas')
    p_a.add_argument('inputs', nargs='+', help='Input image files (2 or more)')
    p_a.add_argument('output', help='Output .bla5 atlas recipe')
    p_a.add_argument('--epochs', type=int, default=1500)
    p_a.add_argument('--bits', type=int, default=8, choices=[4, 8])
    p_a.add_argument('--hidden', type=int, default=64)
    p_a.add_argument('--layers', type=int, default=3)
    p_a.add_argument('--omega', type=float, default=30.0)
    p_a.set_defaults(func=cmd_atlas)

    p_ad = sub.add_parser('atlas-decompress', help='Decompress a .bla5 atlas into N images')
    p_ad.add_argument('input', help='Input .bla5 atlas recipe')
    p_ad.add_argument('output_dir', nargs='?', default=None,
                      help='Output directory (default: just verify SHA-256)')
    p_ad.set_defaults(func=cmd_atlas_decompress)

    p_l = sub.add_parser('lossy', help='Lossy compression (no residual, smaller recipe, NOT bit-perfect)')
    p_l.add_argument('input')
    p_l.add_argument('output')
    p_l.add_argument('--epochs', type=int, default=1500)
    p_l.add_argument('--bits', type=int, default=4, choices=[4, 8],
                     help='4 = aggressive (default), 8 = high quality')
    p_l.add_argument('--prune', type=float, default=0.005,
                     help='Prune threshold (0.0=none, 0.01=aggressive)')
    p_l.add_argument('--hidden', type=int, default=32)
    p_l.add_argument('--layers', type=int, default=2)
    p_l.add_argument('--omega', type=float, default=30.0)
    p_l.add_argument('--batch-size', type=int, default=2048)
    p_l.add_argument('--amp', action='store_true',
                     help='Use mixed precision (bfloat16 on CPU, ~1.5x faster)')
    p_l.set_defaults(func=cmd_lossy)

    p_cb = sub.add_parser('combo',
                           help='v5.9 Combo: hypernetwork + WebP residual (best for N similar images)')
    p_cb.add_argument('inputs', nargs='+', help='Input image files (2 or more)')
    p_cb.add_argument('output', help='Output .blkh9 combo recipe')
    p_cb.add_argument('--latent', type=int, default=16,
                       help='Latent vector dim per image (default 16 = 16 bytes INT8)')
    p_cb.add_argument('--hidden', type=int, default=16)
    p_cb.add_argument('--layers', type=int, default=1)
    p_cb.add_argument('--omega', type=float, default=30.0)
    p_cb.add_argument('--codec', choices=['webp', 'png', 'zlib'], default='webp',
                       help='Residual codec (webp=best, png=compat, zlib=legacy)')
    p_cb.add_argument('--base-epochs', type=int, default=2000,
                       help='Phase 1: hypernetwork training epochs')
    p_cb.add_argument('--compress-epochs', type=int, default=800,
                       help='Phase 2: per-image latent training epochs')
    p_cb.set_defaults(func=cmd_combo)

    p_cbd = sub.add_parser('combo-decompress', help='Decompress a .blkh9 combo recipe')
    p_cbd.add_argument('input', help='Input .blkh9 combo recipe')
    p_cbd.add_argument('output_dir', nargs='?', default=None,
                        help='Output directory (default: just verify SHA-256)')
    p_cbd.set_defaults(func=cmd_combo_decompress)

    p_v = sub.add_parser('video',
                          help='v5.11 Video: temporal SIREN f(x,y,t) for N frames')
    p_v.add_argument('inputs', nargs='+',
                      help='Input frame images (2+) OR a single directory of PNGs')
    p_v.add_argument('output', help='Output .blkv video recipe')
    p_v.add_argument('--epochs', type=int, default=2000)
    p_v.add_argument('--hidden', type=int, default=64)
    p_v.add_argument('--layers', type=int, default=3)
    p_v.add_argument('--omega', type=float, default=30.0)
    p_v.add_argument('--omega-t', type=float, default=1.0,
                      help='Temporal frequency scale (lower=smoother motion)')
    p_v.add_argument('--codec', choices=['webp', 'png', 'zlib'], default='webp')
    p_v.set_defaults(func=cmd_video)

    p_vd = sub.add_parser('video-decompress', help='Decompress a .blkv video recipe')
    p_vd.add_argument('input', help='Input .blkv video recipe')
    p_vd.add_argument('output_dir', nargs='?', default=None,
                       help='Output directory (default: just verify SHA-256)')
    p_vd.set_defaults(func=cmd_video_decompress)

    p_vol = sub.add_parser('volume',
                            help='v5.12 3D Volume: SIREN f(x,y,z) for MRI/CT stack of slices')
    p_vol.add_argument('input', help='Input directory of PNG slices (sorted by name)')
    p_vol.add_argument('output', help='Output .blk3 volume recipe')
    p_vol.add_argument('--epochs', type=int, default=2000)
    p_vol.add_argument('--hidden', type=int, default=64)
    p_vol.add_argument('--layers', type=int, default=3)
    p_vol.add_argument('--omega', type=float, default=30.0)
    p_vol.set_defaults(func=cmd_volume)

    p_vold = sub.add_parser('volume-decompress', help='Decompress a .blk3 volume recipe into PNG slices')
    p_vold.add_argument('input', help='Input .blk3 volume recipe')
    p_vold.add_argument('output_dir', nargs='?', default=None,
                         help='Output directory (default: just verify SHA-256)')
    p_vold.set_defaults(func=cmd_volume_decompress)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
