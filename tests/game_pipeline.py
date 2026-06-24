#!/usr/bin/env python3
"""
Black Hole - Game Texture Pipeline v4
Demonstrates multi-texture compression for game engines.
A game level with 6 textures compressed simultaneously.
"""
import sys
import os
import tempfile
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
from siren_v4 import ImageINRV4, MetaImageCompressorV4

import numpy as np
import matplotlib.pyplot as plt

def create_skybox(size=128):
    """Create a sky gradient texture."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            b = 150 + int((i/size) * 105)
            g = 100 + int((i/size) * 80)
            r = 50 + int((i/size) * 60)
            img[i, j] = [r, g, b]
    return img

def create_ground(size=128):
    """Create a ground/dirt texture."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            r = 120 + np.random.randint(-30, 30)
            g = 80 + np.random.randint(-20, 20)
            b = 40 + np.random.randint(-10, 10)
            img[i, j] = [r, g, b]
    return img

def create_water(size=128):
    """Create a water surface texture."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            wave = int(np.sin(i * 0.2) * 20 + np.sin(j * 0.3) * 15)
            r = 20 + wave
            g = 50 + wave
            b = 150 + wave
            img[i, j] = [np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)]
    return img

def create_brick(size=64):
    """Create a brick wall texture."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            brick_y = i // 8
            offset = 4 if brick_y % 2 == 0 else 0
            is_mortar = (j + offset) % 8 == 0 or i % 8 == 0
            if is_mortar:
                img[i, j] = [180, 180, 180]
            else:
                r = 120 + np.random.randint(-20, 20)
                g = 60 + np.random.randint(-10, 10)
                b = 40 + np.random.randint(-10, 10)
                img[i, j] = [r, g, b]
    return img

def create_grass(size=64):
    """Create a grass texture."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            g = 100 + int((i/size) * 80) + np.random.randint(-10, 10)
            r = 30 + int((j/size) * 40)
            b = 20 + np.random.randint(-5, 5)
            img[i, j] = [np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)]
    return img

def create_wood(size=64):
    """Create a wood plank texture."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            grain = int(np.sin(j * 0.3) * 20 + np.sin(i * 0.1) * 10)
            r = 120 + grain + np.random.randint(-5, 5)
            g = 80 + grain + np.random.randint(-5, 5)
            b = 40 + grain + np.random.randint(-5, 5)
            img[i, j] = [np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)]
    return img

def zip_size(data):
    import zipfile
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
        zp = f.name
    with zipfile.ZipFile(zp, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('data', data)
    with open(zp, 'rb') as f:
        sz = len(f.read())
    os.unlink(zp)
    return sz

def main():
    print("="*70)
    print("  BLACK HOLE — GAME TEXTURE PIPELINE v4")
    print("  6 textures compressed for a game level")
    print("="*70)
    
    textures = {
        'skybox': create_skybox(128),
        'ground': create_ground(128),
        'water': create_water(128),
        'brick': create_brick(64),
        'grass': create_grass(64),
        'wood': create_wood(64),
    }
    
    total_original = 0
    total_zip = 0
    total_blkh = 0
    results = []
    
    for name, tex in textures.items():
        orig = tex.tobytes()
        orig_sz = len(orig)
        zip_sz = zip_size(orig)
        
        compressor = ImageINRV4(hidden_dim=32, num_layers=2)
        t0 = time.time()
        compressor.compress(tex, epochs=2000, lr=1e-3)
        
        with tempfile.NamedTemporaryFile(suffix='.blkh', delete=False) as f:
            rp = f.name
        compressor.save_recipe(rp, bits=4, prune_threshold=0.01)
        
        with open(rp, 'rb') as f:
            blkh_sz = len(f.read())
        t1 = time.time()
        
        recon = compressor.reconstruct()
        mse = np.mean((recon.astype(np.float32) - tex.astype(np.float32)) ** 2)
        psnr = 10 * np.log10(65025.0 / mse) if mse > 0 else 0
        match = np.mean(recon == tex) * 100
        
        os.unlink(rp)
        
        total_original += orig_sz
        total_zip += zip_sz
        total_blkh += blkh_sz
        
        results.append({
            'name': name, 'size': tex.shape[0],
            'orig': orig_sz, 'zip': zip_sz, 'blkh': blkh_sz,
            'zip_ratio': orig_sz / zip_sz, 'blkh_ratio': orig_sz / blkh_sz,
            'psnr': psnr, 'match': match, 'time': t1 - t0
        })
        
        print(f"\n[{name}] {tex.shape[0]}x{tex.shape[0]}x3")
        print(f"  Original: {orig_sz} bytes | ZIP: {zip_sz} ({orig_sz/zip_sz:.2f}x)")
        print(f"  BLKH v4:  {blkh_sz} bytes ({orig_sz/blkh_sz:.2f}x) | {t1-t0:.1f}s")
        print(f"  PSNR: {psnr:.1f} dB | Match: {match:.1f}%")
    
    print(f"\n{'='*70}")
    print("  TOTALS")
    print(f"{'='*70}")
    print(f"  Original: {total_original:,} bytes")
    print(f"  ZIP:      {total_zip:,} bytes ({total_original/total_zip:.2f}x)")
    print(f"  BLKH v4:  {total_blkh:,} bytes ({total_original/total_blkh:.2f}x)")
    print(f"  Space saved vs ZIP: {total_zip - total_blkh:,} bytes ({(1-total_blkh/total_zip)*100:.1f}%)")
    
    # Save JSON
    clean_results = []
    for r in results:
        c = {}
        for k, v in r.items():
            if hasattr(v, 'item'):
                c[k] = v.item()
            else:
                c[k] = v
        clean_results.append(c)
    
    with open('game_pipeline_results.json', 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Generate chart
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0d0d1a')
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    w = 0.35
    
    zr = [r['zip_ratio'] for r in results]
    br = [r['blkh_ratio'] for r in results]
    
    ax.bar(x - w/2, zr, w, label='ZIP', color='#4a4a6a', alpha=0.8)
    ax.bar(x + w/2, br, w, label='BLKH v4', color='#7b2cbf', alpha=0.8)
    
    ax.set_ylabel('Compression Ratio', color='white', fontsize=12)
    ax.set_title('Game Texture Pipeline: 6 Textures (ZIP vs BLKH v4)', color='white', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, color='white', fontsize=10)
    ax.legend(labelcolor='white', facecolor='#1a1a2e', edgecolor='#7b2cbf')
    ax.set_facecolor('#0d0d1a')
    ax.tick_params(colors='white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#555577')
    ax.grid(True, alpha=0.2, color='#555577')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('game_pipeline.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    
    print(f"\n  Saved: game_pipeline.png")
    print(f"  Saved: game_pipeline_results.json")
    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
