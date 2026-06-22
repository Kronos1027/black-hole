#!/usr/bin/env python3
"""
Black Hole - Phase 1: INR Decompressor
Reconstruct a file from a SIREN neural recipe.
Usage: python decompress.py <recipe.json> <output_file> <original_size>
"""
import sys
import os
from siren_core import DataINRCompressor

def main():
    if len(sys.argv) < 4:
        print("Usage: python decompress.py <recipe.json> <output_file> <original_size>")
        sys.exit(1)
    
    recipe_file = sys.argv[1]
    output_file = sys.argv[2]
    original_size = int(sys.argv[3])
    
    if not os.path.exists(recipe_file):
        print(f"Error: recipe not found: {recipe_file}")
        sys.exit(1)
    
    print(f"[Black Hole] Loading recipe: {recipe_file}")
    compressor = DataINRCompressor(hidden_dim=64, num_layers=3)
    compressor.load_recipe(recipe_file)
    
    print(f"[Black Hole] Ejecting {original_size} bytes...")
    data = compressor.reconstruct(original_size)
    
    with open(output_file, 'wb') as f:
        f.write(data.tobytes())
    
    print(f"[Black Hole] Ejection complete: {output_file}")
    print(f"  Bytes: {len(data)}")

if __name__ == '__main__':
    main()
