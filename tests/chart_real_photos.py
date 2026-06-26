"""Generate real photos chart for README."""
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

with open(os.path.join(os.path.dirname(__file__), 'benchmark_real_photos_results.json')) as f:
    results = json.load(f)

COLOR_ZIP = '#4a4a6a'
COLOR_BLKH = '#00d9c0'
COLOR_ACCENT = '#ff006e'

fig, ax = plt.subplots(figsize=(12, 6))
labels = [r['name'].replace('_128', '').capitalize() for r in results]
x = np.arange(len(labels))
w = 0.35

ax.bar(x - w/2, [r['zip_size'] for r in results], w,
       label='ZIP (zlib-9)', color=COLOR_ZIP, alpha=0.9)
ax.bar(x + w/2, [r['blkh_size'] for r in results], w,
       label='BLKH v5 (bit-perfect)', color=COLOR_BLKH, alpha=0.9)

# Highlight winners
for i, r in enumerate(results):
    color = COLOR_ACCENT if r['winner'] == 'ZIP' else '#00ff88'
    ax.annotate(r['winner'], xy=(i, max(r['zip_size'], r['blkh_size']) * 1.04),
                ha='center', color=color, fontsize=11, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Compressed size (bytes)')
ax.set_title('Real Photos Benchmark — BLKH v5 vs ZIP (128x128 RGB, all SHA-256 verified)',
             fontweight='bold')
ax.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white',
          loc='upper right')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Annotate ratios
for i, r in enumerate(results):
    ax.annotate(f"{r['blkh_vs_zip']:.2f}x",
                xy=(i + w/2, r['blkh_size']),
                xytext=(0, -20), textcoords='offset points',
                ha='center', color=COLOR_ACCENT, fontsize=10, fontweight='bold')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'v5_real_photos.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print(f"Saved: {out}")
