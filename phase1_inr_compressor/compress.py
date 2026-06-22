#!/usr/bin/env python3
"""
Black Hole - Phase 1: INR Compressor
Compress any file into a SIREN neural recipe.
Usage: python compress.py <input_file> <output_recipe.json>
"""
import sys
import os
import time
from siren_core import DataINRCompressor

def main():
    if len(sys.argv) < 3:
        print("Usage: python compress.py <input_file> <output_recipe.json> [epochs] [lr]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_recipe = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
    
    if not os.path.exists(input_file):
        print(f"Error: file not found: {input_file}")
        sys.exit(1)
    
    print(f"[Black Hole] Ingesting: {input_file}")
    print(f"[Black Hole] Singularity activation...")
    
    with open(input_file, 'rb') as f:
        data = f.read()
    
    original_size = len(data)
    print(f"[Black Hole] Original entropy: {original_size} bytes")
    
    compressor = DataINRCompressor(hidden_dim=64, num_layers=3)
    
    start = time.time()
    meta = compressor.compress(data, epochs=epochs, lr=lr)
    elapsed = time.time() - start
    
    compressor.save_recipe(output_recipe)
    recipe_size = os.path.getsize(output_recipe)
    
    ratio = original_size / recipe_size if recipe_size > 0 else float('inf')
    
    print(f"\n[Black Hole] Ingestion complete.")
    print(f"  Original:   {original_size:>10} bytes")
    print(f"  Recipe:     {recipe_size:>10} bytes")
    print(f"  Ratio:      {ratio:>10.2f}x")
    print(f"  MSE:        {meta['mse']:>10.6e}")
    print(f"  PSNR:       {meta['psnr']:>10.2f} dB")
    print(f"  Time:       {elapsed:>10.2f}s")
    print(f"  Recipe:     {output_recipe}")

if __name__ == '__main__':
    main()
