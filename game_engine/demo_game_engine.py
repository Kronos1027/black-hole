#!/usr/bin/env python3
"""
demo_game_engine.py — End-to-end demo: compress → serve → fetch → decompress
=============================================================================
Demonstrates the full BLKH game engine pipeline:
  1. Generate a game texture (synthetic skybox)
  2. Compress it with BLKH hybrid mode
  3. Start the BLKH Texture Streaming Server
  4. Fetch the texture via HTTP (simulating a game engine client)
  5. Decompress and verify SHA-256

This is exactly what a Unity/Godot game would do at runtime.
"""
import sys
import os
import time
import zlib
import hashlib
import threading
import urllib.request
import io
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game_engine', 'server'))

from siren_v5_hybrid import HybridCompressor
from blkh_texture_server import BLKHTextureServer
from http.server import HTTPServer


def make_skybox_texture(size=256):
    """Generate a game skybox texture (smooth gradient + sun glow)."""
    rng = np.random.default_rng(42)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Sky gradient
    img[:, :, 2] = 180 + 60 * (1 - ys)  # blue
    img[:, :, 1] = 130 + 50 * (1 - ys)  # green
    img[:, :, 0] = 100 + 40 * (1 - ys)  # red
    # Sun glow
    cx, cy = 0.7, 0.2
    glow = 80 * np.exp(-((xs - cx)**2 + (ys - cy)**2) / 0.05)
    img += glow[:, :, None]
    # Noise (camera/sensor)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    print("=" * 70)
    print("  BLKH Game Engine Demo — Full Pipeline")
    print("=" * 70)

    # Step 1: Generate game texture
    print("\n[1] Generating skybox texture (256x256)...")
    img = make_skybox_texture(256)
    print(f"    Original: {img.nbytes:,}B")
    zip_sz = len(zlib.compress(img.tobytes(), 9))
    print(f"    ZIP:      {zip_sz:,}B")

    # Step 2: Compress with BLKH
    print("\n[2] Compressing with BLKH hybrid (auto-tune + WebP residual)...")
    comp = HybridCompressor(auto_tune=True, residual_codec='webp')
    t0 = time.time()
    res = comp.compress_bitperfect(img, epochs=600, lr=1e-3, bits=8,
                                     batch_size=8192, use_amp=True, verbose=False)
    dt = time.time() - t0
    print(f"    BLKH:     {res['recipe_size']:,}B  (ratio {img.nbytes/res['recipe_size']:.2f}x)")
    print(f"    vs ZIP:   {zip_sz/res['recipe_size']:.2f}x smaller")
    print(f"    SHA-256:  {res['sha256'][:32]}...")
    print(f"    Time:     {dt:.1f}s")

    # Save recipe
    tex_dir = os.path.join(os.path.dirname(__file__), '..', 'game_engine', 'server', 'textures')
    os.makedirs(tex_dir, exist_ok=True)
    recipe_path = os.path.join(tex_dir, 'skybox.blkh8')
    with open(recipe_path, 'wb') as f:
        f.write(res['recipe_bytes'])

    # Step 3: Start server in background thread
    print("\n[3] Starting BLKH Texture Streaming Server on port 8080...")
    server = HTTPServer(('localhost', 8080), BLKHTextureServer)
    server.texture_dir = __import__('pathlib').Path(tex_dir)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)  # Let server start
    print("    Server running at http://localhost:8080")

    try:
        # Step 4: Fetch texture metadata (simulating game engine GET /texture/skybox.blkh8/info)
        print("\n[4] Fetching texture info (GET /texture/skybox.blkh8/info)...")
        resp = urllib.request.urlopen('http://localhost:8080/texture/skybox.blkh8/info')
        info = json.loads(resp.read())
        print(f"    Name:     {info['name']}")
        print(f"    Size:     {info['size']:,}B")
        print(f"    Format:   {info['format']}")

        # Step 5: Fetch decoded PNG (simulating game engine GET /texture/skybox.blkh8/decode)
        print("\n[5] Fetching decoded PNG (GET /texture/skybox.blkh8/decode)...")
        t0 = time.time()
        resp = urllib.request.urlopen('http://localhost:8080/texture/skybox.blkh8/decode')
        png_data = resp.read()
        dt_fetch = time.time() - t0
        print(f"    PNG size: {len(png_data):,}B")
        print(f"    Fetch+decode time: {dt_fetch*1000:.0f}ms")

        # Step 6: Verify the fetched PNG matches the original
        print("\n[6] Verifying SHA-256 match...")
        from PIL import Image
        recovered_img = np.array(Image.open(io.BytesIO(png_data)).convert('RGB'), dtype=np.uint8)
        orig_sha = hashlib.sha256(img.tobytes()).hexdigest()
        rec_sha = hashlib.sha256(recovered_img.tobytes()).hexdigest()
        match = (orig_sha == rec_sha)
        print(f"    Original SHA:    {orig_sha[:32]}...")
        print(f"    Recovered SHA:   {rec_sha[:32]}...")
        print(f"    Match: {'✓ PERFECT' if match else '✗ FAILED'}")

        # Summary
        print("\n" + "=" * 70)
        print("  DEMO SUMMARY")
        print("=" * 70)
        print(f"  Original texture:     {img.nbytes:>8,}B")
        print(f"  ZIP compression:      {zip_sz:>8,}B  ({img.nbytes/zip_sz:.2f}x)")
        print(f"  BLKH recipe:          {res['recipe_size']:>8,}B  ({img.nbytes/res['recipe_size']:.2f}x)")
        print(f"  BLKH vs ZIP:          {zip_sz/res['recipe_size']:.2f}x smaller")
        print(f"  Roundtrip:            {'PERFECT (SHA-256 verified)' if match else 'FAILED'}")
        print(f"  Fetch+decode latency: {dt_fetch*1000:.0f}ms")
        print(f"\n  In a real game:")
        print(f"    1. Compress textures offline with BLKH (saves disk space)")
        print(f"    2. Serve via BLKH Texture Server (or bundle .blkh8 files)")
        print(f"    3. Unity/Godot fetches and decodes at runtime")
        print(f"    4. Game loads faster (smaller files) with identical quality")

    finally:
        server.shutdown()
        print("\n  Server stopped.")

    print("\n  Files created:")
    print(f"    {recipe_path}")
    print(f"    Unity client:   game_engine/unity/BLKHTextureLoader.cs")
    print(f"    Godot client:   game_engine/godot/BLKHTextureLoader.gd")
    print(f"    Server:         game_engine/server/blkh_texture_server.py")


if __name__ == '__main__':
    main()
