"""Generate image formats comparison chart."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.facecolor': '#0d0d1a',
    'axes.facecolor': '#0d0d1a',
    'axes.edgecolor': '#555577',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'axes.grid': True,
    'grid.color': '#33334a',
    'grid.alpha': 0.5,
    'font.size': 11,
})

with open(os.path.join(os.path.dirname(__file__), 'benchmark_vs_image_formats_results.json')) as f:
    results = json.load(f)

COLOR_PNG = '#4a90e2'
COLOR_WEBP = '#00d9c0'
COLOR_ZIP = '#4a4a6a'
COLOR_BLKH = '#7b2cbf'
COLOR_JPEG = '#ff8c42'
COLOR_WEBP_LOSSY = '#ff6b9d'

fig, ax = plt.subplots(figsize=(14, 7))
labels = [r['name'] for r in results]
x = np.arange(len(labels))
n = len(labels)
# 4 lossless + 2 lossy = 6 bars per photo
w = 0.13

# Lossless (left to right)
ax.bar(x - 2.5*w, [r['png_lossless'] for r in results], w,
       label='PNG (lossless)', color=COLOR_PNG, alpha=0.9)
ax.bar(x - 1.5*w, [r['webp_lossless'] for r in results], w,
       label='WebP (lossless)', color=COLOR_WEBP, alpha=0.9)
ax.bar(x - 0.5*w, [r['zip_lossless'] for r in results], w,
       label='ZIP (lossless)', color=COLOR_ZIP, alpha=0.9)
ax.bar(x + 0.5*w, [r['blkh_lossless'] for r in results], w,
       label='BLKH v5 (lossless, bit-perfect)', color=COLOR_BLKH, alpha=0.9)
# Lossy (context only)
ax.bar(x + 1.5*w, [r['jpeg_lossy'] for r in results], w,
       label='JPEG q=85 (LOSSY)', color=COLOR_JPEG, alpha=0.6, hatch='//')
ax.bar(x + 2.5*w, [r['webp_lossy'] for r in results], w,
       label='WebP q=85 (LOSSY)', color=COLOR_WEBP_LOSSY, alpha=0.6, hatch='//')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Compressed size (bytes)')
ax.set_title('BLKH v5 vs Image Formats — Real Photos (128x128 RGB)\n'
             'Lossless (solid) vs Lossy (hatched, NOT comparable to BLKH)',
             fontweight='bold')
ax.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white',
          loc='upper right', fontsize=9, ncol=2)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'v5_vs_image_formats.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print(f"Saved: {out}")
