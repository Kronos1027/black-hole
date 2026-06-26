#!/usr/bin/env python3
"""
demo_visual.py — Generate side-by-side visual comparison for README
====================================================================
Creates a single image showing:
  [Original]  [BLKH Reconstructed]  [Difference (amplified)]
for 3 sample images (sky, wood, water).
Saves to docs/assets/v5_visual_demo.png
"""
import sys
import os
import zlib
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_inr_compressor'))
torch.set_num_threads(4)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from siren_v5_torch import ImageINRv5
# Reuse photo generators
from benchmark_real_photos import make_sky_photo, make_wood_photo, make_water_photo

plt.rcParams.update({
    'figure.facecolor': '#0d0d1a',
    'axes.facecolor': '#0d0d1a',
    'axes.edgecolor': '#555577',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'font.size': 10,
})


def amplify_diff(orig, recon, amp=10):
    """Show pixel differences amplified for visibility."""
    diff = np.abs(orig.astype(np.int16) - recon.astype(np.int16))
    return np.clip(diff * amp, 0, 255).astype(np.uint8)


def main():
    print("Generating visual demo...")

    photos = [
        ('Sky',   make_sky_photo(128, seed=1)),
        ('Wood',  make_wood_photo(128, seed=2)),
        ('Water', make_water_photo(128, seed=3)),
    ]

    fig, axes = plt.subplots(len(photos), 3, figsize=(12, 4 * len(photos)))
    if len(photos) == 1:
        axes = axes.reshape(1, -1)

    for row, (name, img) in enumerate(photos):
        print(f"  compressing {name}...")
        orig_bytes = img.tobytes()
        zip_size = len(zlib.compress(orig_bytes, 9))

        comp = ImageINRv5(hidden_features=32, hidden_layers=2, omega_0=30.0)
        res = comp.compress_bitperfect(img, epochs=1500, lr=1e-3,
                                         bits=8, batch_size=2048, verbose=False)
        recon, meta = ImageINRv5.decompress(res['recipe_bytes'])
        assert meta['exact_match'], f"SHA-256 failed for {name}"

        # Visualizations
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f'{name} — Original ({img.nbytes:,}B)',
                                color='white', fontsize=11)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(recon)
        axes[row, 1].set_title(f'BLKH Reconstructed ({res["recipe_size"]:,}B, '
                                f'bit acc {res["model_bit_accuracy"]:.1f}%)',
                                color='#00d9c0', fontsize=11)
        axes[row, 1].axis('off')

        # Difference (amplified 10x)
        diff = amplify_diff(img, recon, amp=10)
        axes[row, 2].imshow(diff)
        n_diff_pixels = int(np.sum(np.any(img != recon, axis=-1)))
        pct_diff = 100 * n_diff_pixels / (img.shape[0] * img.shape[1])
        axes[row, 2].set_title(f'Difference (10x amplified)\n'
                                f'{pct_diff:.1f}% pixels differ, '
                                f'residual covers ALL → 100% bit-perfect',
                                color='#ff006e', fontsize=10)
        axes[row, 2].axis('off')

        print(f"    {name}: orig={img.nbytes:,}B  zip={zip_size:,}B  "
              f"blkh={res['recipe_size']:,}B  bit%={res['model_bit_accuracy']:.1f}  "
              f"diff_pixels={pct_diff:.1f}%  SHA-256=OK")

    plt.suptitle('BLKH v5 — Visual Roundtrip Demo (100% bit-perfect, SHA-256 verified)',
                  color='white', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..',
                        'docs', 'assets', 'v5_visual_demo.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
