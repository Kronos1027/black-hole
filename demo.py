#!/usr/bin/env python3
"""
Black Hole - Unified Demo
Runs the full pipeline: generate signal -> compress -> reconstruct -> visualize.
"""
import sys
import os

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase2_opportunistic_daemon'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase3_ejection_engine'))

from siren_core import DataINRCompressor
from ejector import EjectionEngine
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def main():
    print("=" * 60)
    print("  BLACK HOLE - UNIFIED DEMONSTRATION")
    print("=" * 60)
    
    # 1. Generate a synthetic 1D periodic signal
    N = 512
    t = np.linspace(0, 1, N)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 13 * t) + 0.2 * np.cos(2 * np.pi * 29 * t)
    y = ((signal + 1.7) / 3.4 * 255).astype(np.uint8)
    
    print(f"\n[1] Signal generated: {N} bytes (1D periodic)")
    
    # 2. Phase 1: Compress into Singularity
    print("\n[2] Phase 1: Singularity (SIREN INR Compression)")
    compressor = DataINRCompressor(hidden_dim=64, num_layers=3)
    meta = compressor.compress(y.tobytes(), epochs=3000, lr=1e-4)
    print(f"    MSE: {meta['mse']:.6e} | PSNR: {meta['psnr']:.2f} dB")
    
    # 3. Phase 2: Save recipe (represents the "Horizon" pre-computation)
    recipe_path = 'demo.recipe.json'
    compressor.save_recipe(recipe_path)
    recipe_size = os.path.getsize(recipe_path)
    print(f"\n[3] Phase 2: Recipe saved ({recipe_size} bytes)")
    print(f"    Ratio: {recipe_size/N:.1f}x (prototype; target with quantization: <0.1x)")
    
    # 4. Phase 3: Eject directly to memory
    print("\n[4] Phase 3: Ejection Engine")
    engine = EjectionEngine()
    data, eject_time = engine.eject(recipe_path, N)
    print(f"    Ejected in {eject_time*1000:.2f} ms. Zero disk write.")
    
    # 5. Validate
    match = np.mean(data == y)
    mse = np.mean((data.astype(np.float32) - y.astype(np.float32)) ** 2)
    psnr = 10 * np.log10(65025.0 / mse) if mse > 0 else float('inf')
    print(f"\n[5] Validation: Exact match: {match*100:.1f}% | PSNR: {psnr:.2f} dB")
    
    # 6. Visualize
    if HAS_MATPLOTLIB:
        print("\n[6] Generating visualization...")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        axes[0].plot(t, y, 'b-', linewidth=1.5)
        axes[0].set_title('Original Signal')
        axes[0].set_xlabel('Coordinate')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t, data, 'r-', linewidth=1.5)
        axes[1].set_title('Reconstructed from Recipe')
        axes[1].set_xlabel('Coordinate')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)
        
        error = np.abs(data.astype(np.float32) - y.astype(np.float32))
        axes[2].plot(t, error, 'g-', linewidth=1.5)
        axes[2].set_title(f'Error (PSNR: {psnr:.1f} dB)')
        axes[2].set_xlabel('Coordinate')
        axes[2].set_ylabel('|Error|')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Black Hole Prototype: Neural Compression & Ejection', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('black_hole_demo.png', dpi=150, bbox_inches='tight')
        print("    Saved: black_hole_demo.png")
    
    # Cleanup
    os.unlink(recipe_path)
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE. The singularity is stable.")
    print("=" * 60)

if __name__ == '__main__':
    main()
