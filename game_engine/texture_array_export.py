# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

"""
texture_array_export.py — Export BLKH textures as game engine texture arrays
=============================================================================
Converts a directory of BLKH recipes into a texture array format that
game engines (Unity, Unreal, Godot) can load directly.

Output formats:
  1. .bin + .json manifest (engine-agnostic)
  2. Unity Texture2DArray (.bin + C# loader script)
  3. Godot TextureArray (.bin + GDScript loader)

Usage:
    python texture_array_export.py --input textures/ --output array.bin --format unity
"""
import os
import sys
import json
import struct
import hashlib
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))


def export_texture_array(input_dir: str, output_prefix: str, format: str = 'generic'):
    """
    Export all BLKH recipes in a directory as a texture array.
    
    Args:
        input_dir: Directory with .blkh8/.blkw files
        output_prefix: Output path prefix (e.g. 'textures/array')
        format: 'generic', 'unity', or 'godot'
    """
    input_path = Path(input_dir)
    
    # Find all BLKH recipe files
    recipes = sorted(list(input_path.glob('*.blkh8')) + 
                     list(input_path.glob('*.blkw')) +
                     list(input_path.glob('*.blkh5')))
    
    if not recipes:
        print(f"No BLKH recipes found in {input_dir}")
        return
    
    print(f"Found {len(recipes)} BLKH recipes")
    
    # Decompress all textures
    textures = []
    for recipe_path in recipes:
        data = recipe_path.read_bytes()
        magic = data[:4]
        
        if magic == b'BLK8':
            from siren_v5_hybrid import HybridCompressor
            img, meta = HybridCompressor.decompress(data)
        elif magic == b'BLKW':
            from siren_v5_wavelet import WaveletINRCompressor
            img, meta = WaveletINRCompressor.decompress(data)
        elif magic == b'BLK5':
            from siren_v5_torch import ImageINRv5
            img, meta = ImageINRv5.decompress(data)
        else:
            print(f"  Skip {recipe_path.name}: unknown format {magic}")
            continue
        
        textures.append({
            'name': recipe_path.stem,
            'image': img,
            'shape': img.shape,
            'sha256': hashlib.sha256(img.tobytes()).hexdigest()[:16],
        })
        print(f"  Loaded {recipe_path.name}: {img.shape}")
    
    if not textures:
        print("No valid textures to export")
        return
    
    # Check all same size
    shapes = set(t['shape'] for t in textures)
    if len(shapes) > 1:
        print(f"WARNING: textures have different sizes: {shapes}")
        print("Texture arrays require uniform size. Resizing to smallest...")
        min_h = min(t['shape'][0] for t in textures)
        min_w = min(t['shape'][1] for t in textures)
        from PIL import Image as PILImage
        for t in textures:
            if t['shape'][:2] != (min_h, min_w):
                pil = PILImage.fromarray(t['image']).resize((min_w, min_h), PILImage.BILINEAR)
                t['image'] = np.array(pil, dtype=np.uint8)
                t['shape'] = (min_h, min_w, t['shape'][2])
    
    H, W, C = textures[0]['shape']
    N = len(textures)
    
    # Pack as texture array: N × H × W × C (interleaved)
    array_data = np.stack([t['image'] for t in textures], axis=0)
    
    # Write binary
    bin_path = f"{output_prefix}.bin"
    array_data.tobytes()
    with open(bin_path, 'wb') as f:
        f.write(array_data.tobytes())
    
    bin_size = os.path.getsize(bin_path)
    
    # Write manifest
    manifest = {
        'format': 'blkh_texture_array',
        'version': 1,
        'count': N,
        'width': W,
        'height': H,
        'channels': C,
        'binary_file': os.path.basename(bin_path),
        'binary_size': bin_size,
        'textures': [
            {
                'index': i,
                'name': t['name'],
                'sha256': t['sha256'],
            }
            for i, t in enumerate(textures)
        ],
    }
    
    manifest_path = f"{output_prefix}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nExported texture array:")
    print(f"  Binary:   {bin_path} ({bin_size:,}B)")
    print(f"  Manifest: {manifest_path}")
    print(f"  Count:    {N} textures")
    print(f"  Size:     {W}x{H}x{C}")
    
    # Format-specific output
    if format == 'unity':
        _export_unity_loader(textures, output_prefix, H, W, C, N)
    elif format == 'godot':
        _export_godot_loader(textures, output_prefix, H, W, C, N)
    
    # ZIP comparison
    import zlib
    zip_total = sum(len(zlib.compress(t['image'].tobytes(), 9)) for t in textures)
    print(f"\n  ZIP (per-texture): {zip_total:,}B")
    print(f"  Raw array:         {bin_size:,}B")
    print(f"  Note: Use BLKH recipes for storage, this array is for runtime loading")


def _export_unity_loader(textures, prefix, H, W, C, N):
    """Generate Unity C# loader script."""
    cs_code = f"""// SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
// BLKH Texture Array Loader for Unity
// Auto-generated by BLKH texture_array_export.py

using UnityEngine;
using System.IO;
using System.Collections.Generic;

public class BLKHTextureArrayLoader : MonoBehaviour
{{
    public string binaryPath = "array.bin";
    public string manifestPath = "array.json";

    public Texture2DArray LoadTextureArray()
    {{
        // Load manifest
        string manifestJson = File.ReadAllText(manifestPath);
        var manifest = JsonUtility.FromJson<BLKHManifest>(manifestJson);

        int count = manifest.count;
        int width = manifest.width;
        int height = manifest.height;

        // Load binary
        byte[] data = File.ReadAllBytes(binaryPath);

        // Create Texture2DArray
        var format = TextureFormat.RGB24;
        var texArray = new Texture2DArray(width, height, count, format, false);

        int textureSize = width * height * 3;
        for (int i = 0; i < count; i++)
        {{
            texArray.SetPixelData(data, i, 0, i * textureSize);
        }}
        texArray.Apply();

        Debug.Log($"[BLKH] Loaded texture array: {{count}} textures, {{width}}x{{height}}");
        return texArray;
    }}
}}

[System.Serializable]
public class BLKHManifest
{{
    public int count;
    public int width;
    public int height;
    public int channels;
}}
"""
    cs_path = f"{prefix}_unity_loader.cs"
    with open(cs_path, 'w') as f:
        f.write(cs_code)
    print(f"  Unity C#: {cs_path}")


def _export_godot_loader(textures, prefix, H, W, C, N):
    """Generate Godot GDScript loader."""
    gd_code = f"""# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# BLKH Texture Array Loader for Godot 4
# Auto-generated by BLKH texture_array_export.py

extends Node

## Loads a BLKH texture array from binary + manifest
## Usage: var tex_array = load_texture_array("array.bin", "array.json")

func load_texture_array(bin_path: String, manifest_path: String) -> Texture2DArray:
    # Load manifest
    var file = FileAccess.open(manifest_path, FileAccess.READ)
    var manifest_text = file.get_as_text()
    file.close()
    
    var json = JSON.new()
    json.parse(manifest_text)
    var manifest = json.data
    
    var count: int = manifest.count
    var width: int = manifest.width
    var height: int = manifest.height
    
    # Load binary
    var bin_file = FileAccess.open(bin_path, FileAccess.READ)
    var data = bin_file.get_buffer(bin_file.get_length())
    bin_file.close()
    
    # Create Texture2DArray
    var tex_array = Texture2DArray.new()
    # Godot 4 API for texture arrays
    var image = Image.create_from_data(width, height, false, Image.FORMAT_RGB8, data)
    tex_array.create_from_image(image)
    
    print("[BLKH] Loaded texture array: %d textures, %dx%d" % [count, width, height])
    return tex_array
"""
    gd_path = f"{prefix}_godot_loader.gd"
    with open(gd_path, 'w') as f:
        f.write(gd_code)
    print(f"  Godot GD: {gd_path}")


def main():
    import argparse
    p = argparse.ArgumentParser(description='Export BLKH textures as game engine texture array')
    p.add_argument('--input', required=True, help='Directory with BLKH recipes')
    p.add_argument('--output', required=True, help='Output prefix (e.g. textures/array)')
    p.add_argument('--format', choices=['generic', 'unity', 'godot'], default='generic')
    args = p.parse_args()
    
    export_texture_array(args.input, args.output, args.format)


if __name__ == '__main__':
    main()
