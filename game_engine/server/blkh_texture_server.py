# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

"""
blkh_texture_server.py — BLKH Texture Streaming Server
=======================================================
HTTP server that streams BLKH-compressed textures to game engines
(Unity, Unreal, Godot) at runtime.

Architecture:
    Game Engine  <--HTTP-->  BLKH Server  <--reads-->  .blkh8 files on disk

Endpoints:
    GET /texture/<name>          → raw .blkh8 recipe bytes
    GET /texture/<name>/info     → JSON metadata (size, shape, SHA-256)
    GET /texture/<name>/decode   → decoded PNG (for engines without BLKH native support)
    GET /textures                → list all available textures
    POST /compress               → upload PNG, compress to BLKH, return recipe
    GET /health                  → health check

Usage:
    # Start server (serves .blkh8 files from ./textures/ directory)
    python blkh_texture_server.py --port 8080 --textures ./textures/

    # In Unity/Godot, fetch texture:
    # GET http://localhost:8080/texture/skybox.blkh8
    # → raw recipe bytes → decode client-side OR
    # GET http://localhost:8080/texture/skybox.blkh8/decode
    # → decoded PNG (fallback for engines without BLKH support)

The server is lightweight (stdlib http.server, no external deps beyond BLKH).
Designed for LAN deployment in game studios (low latency, high throughput).
"""
from __future__ import annotations
import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add BLKH to path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'phase1_inr_compressor'))


class BLKHTextureServer(BaseHTTPRequestHandler):
    """HTTP handler for BLKH texture streaming."""

    # Silence default logging (override for custom format)
    def log_message(self, format, *args):
        pass

    def log_request_custom(self, method, path, status, size=0):
        print(f"  [{time.strftime('%H:%M:%S')}] {method} {path} → {status} ({size:,}B)")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.strip('/')

        # Health check
        if path == 'health':
            self._send_json(200, {'status': 'ok', 'blkh': 'v5.14'})
            return

        # List all textures
        if path == 'textures':
            tex_dir = self.server.texture_dir
            if not tex_dir.exists():
                self._send_json(200, {'textures': []})
                return
            textures = []
            for f in sorted(tex_dir.glob('*.blkh*')):
                textures.append({
                    'name': f.name,
                    'size': f.stat().st_size,
                    'format': f.suffix,
                })
            self._send_json(200, {'textures': textures, 'count': len(textures)})
            return

        # Texture endpoints: /texture/<name> or /texture/<name>/info or /texture/<name>/decode
        parts = path.split('/')
        if len(parts) >= 2 and parts[0] == 'texture':
            tex_name = parts[1]
            action = parts[2] if len(parts) > 2 else 'raw'

            tex_path = self.server.texture_dir / tex_name
            if not tex_path.exists():
                self._send_json(404, {'error': f'texture not found: {tex_name}'})
                return

            if action == 'raw':
                # Serve raw recipe bytes
                data = tex_path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('X-BLKH-Format', tex_path.suffix)
                self.end_headers()
                self.wfile.write(data)
                self.log_request_custom('GET', path, 200, len(data))

            elif action == 'info':
                # Return JSON metadata
                data = tex_path.read_bytes()
                info = {
                    'name': tex_name,
                    'size': len(data),
                    'format': tex_path.suffix,
                    'path': str(tex_path),
                }
                # Try to parse header for more info
                if data[:4] == b'BLK5' or data[:4] == b'BLK8':
                    import struct
                    version = data[4]
                    bits = data[5]
                    info['version'] = version
                    info['bits'] = bits
                self._send_json(200, info)

            elif action == 'decode':
                # Decode to PNG (fallback for engines without BLKH support)
                try:
                    data = tex_path.read_bytes()
                    magic = data[:4]
                    if magic == b'BLK5':
                        from siren_v5_torch import ImageINRv5
                        img, meta = ImageINRv5.decompress(data)
                    elif magic == b'BLK8':
                        from siren_v5_hybrid import HybridCompressor
                        img, meta = HybridCompressor.decompress(data)
                    else:
                        self._send_json(400, {'error': f'unsupported format: {magic}'})
                        return

                    # Encode as PNG
                    import io
                    from PIL import Image
                    buf = io.BytesIO()
                    Image.fromarray(img).save(buf, format='PNG')
                    png_data = buf.getvalue()

                    self.send_response(200)
                    self.send_header('Content-Type', 'image/png')
                    self.send_header('Content-Length', str(len(png_data)))
                    self.send_header('X-BLKH-SHA256', meta.get('sha256_match', 'unknown'))
                    self.end_headers()
                    self.wfile.write(png_data)
                    self.log_request_custom('GET', path, 200, len(png_data))
                except Exception as e:
                    self._send_json(500, {'error': str(e)})

            else:
                self._send_json(400, {'error': f'unknown action: {action}'})
            return

        # Root
        if path == '' or path == 'index':
            self._send_json(200, {
                'service': 'BLKH Texture Streaming Server',
                'version': 'v5.14',
                'endpoints': [
                    'GET /health',
                    'GET /textures',
                    'GET /texture/<name> (raw recipe bytes)',
                    'GET /texture/<name>/info (JSON metadata)',
                    'GET /texture/<name>/decode (decoded PNG)',
                    'POST /compress (upload PNG, get BLKH recipe)',
                ],
            })
            return

        self._send_json(404, {'error': f'not found: {path}'})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.strip('/')

        if path == 'compress':
            # Upload PNG, compress to BLKH, return recipe
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_json(400, {'error': 'no data'})
                return

            png_data = self.rfile.read(content_length)

            try:
                from PIL import Image
                import io
                import numpy as np
                from siren_v5_hybrid import HybridCompressor

                img = np.array(Image.open(io.BytesIO(png_data)).convert('RGB'), dtype=np.uint8)

                # Compress with auto-tune
                comp = HybridCompressor(auto_tune=True, residual_codec='webp')
                res = comp.compress_bitperfect(img, epochs=800, lr=1e-3,
                                                 bits=8, batch_size=8192,
                                                 use_amp=True, verbose=False)

                # Save recipe if name provided
                qs = parse_qs(parsed.query)
                name = qs.get('name', ['uploaded.blkh8'])[0]
                recipe_path = self.server.texture_dir / name
                recipe_path.parent.mkdir(parents=True, exist_ok=True)
                recipe_path.write_bytes(res['recipe_bytes'])

                self._send_json(200, {
                    'name': name,
                    'original_size': res['original_size'],
                    'recipe_size': res['recipe_size'],
                    'ratio': round(res['original_size'] / res['recipe_size'], 2),
                    'sha256': res['sha256'][:32] + '...',
                    'path': str(recipe_path),
                })
                self.log_request_custom('POST', path, 200, res['recipe_size'])

            except Exception as e:
                self._send_json(500, {'error': str(e)})
            return

        self._send_json(404, {'error': f'not found: {path}'})

    def _send_json(self, status, data):
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description='BLKH Texture Streaming Server')
    parser.add_argument('--port', type=int, default=8080, help='Port (default 8080)')
    parser.add_argument('--host', default='0.0.0.0', help='Host (default 0.0.0.0)')
    parser.add_argument('--textures', default='./textures',
                        help='Directory with .blkh* files (default ./textures/)')
    args = parser.parse_args()

    tex_dir = Path(args.textures)
    tex_dir.mkdir(parents=True, exist_ok=True)

    server = HTTPServer((args.host, args.port), BLKHTextureServer)
    server.texture_dir = tex_dir

    print("=" * 60)
    print("  BLKH Texture Streaming Server v5.14")
    print("=" * 60)
    print(f"  Host:      {args.host}")
    print(f"  Port:      {args.port}")
    print(f"  Textures:  {tex_dir.resolve()}")
    print(f"  URL:       http://localhost:{args.port}")
    print("=" * 60)
    print(f"\n  Endpoints:")
    print(f"    GET  /health              — health check")
    print(f"    GET  /textures            — list all textures")
    print(f"    GET  /texture/<name>      — raw recipe bytes")
    print(f"    GET  /texture/<name>/info — JSON metadata")
    print(f"    GET  /texture/<name>/decode — decoded PNG")
    print(f"    POST /compress?name=X     — upload PNG, compress to BLKH")
    print(f"\n  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


if __name__ == '__main__':
    main()
