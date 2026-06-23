"""Generate large image scaling chart."""
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

with open(os.path.join(os.path.dirname(__file__), 'benchmark_large_results.json')) as f:
    results = json.load(f)

COLOR_ZIP = '#4a4a6a'
COLOR_BLKH = '#00d9c0'
COLOR_ACCENT = '#ff006e'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sizes = sorted(set(r['size'] for r in results))
gradient_zip = [next(r['zip_size'] for r in results if r['name'] == f'gradient_{s}') for s in sizes]
gradient_blkh = [next(r['blkh_size'] for r in results if r['name'] == f'gradient_{s}') for s in sizes]
blobs_zip = [next(r['zip_size'] for r in results if r['name'] == f'blobs_{s}') for s in sizes]
blobs_blkh = [next(r['blkh_size'] for r in results if r['name'] == f'blobs_{s}') for s in sizes]

x = np.arange(len(sizes))
w = 0.35

# Panel 1: gradient
ax1.bar(x - w/2, gradient_zip, w, label='ZIP (zlib-9)',
        color=COLOR_ZIP, alpha=0.9)
ax1.bar(x + w/2, gradient_blkh, w, label='BLKH v5 (bit-perfect)',
        color=COLOR_BLKH, alpha=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{s}x{s}' for s in sizes])
ax1.set_xlabel('Image size')
ax1.set_ylabel('Compressed size (bytes, log scale)')
ax1.set_yscale('log')
ax1.set_title('Pure Gradient — BLKH wins big at scale', fontweight='bold')
ax1.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white')
for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)
# annotate ratios
for i, s in enumerate(sizes):
    ratio = gradient_zip[i] / gradient_blkh[i]
    ax1.annotate(f'{ratio:.1f}x\nsmaller', xy=(i, gradient_blkh[i]),
                 xytext=(0, -30), textcoords='offset points',
                 ha='center', color=COLOR_ACCENT, fontweight='bold', fontsize=9)

# Panel 2: blobs
ax2.bar(x - w/2, blobs_zip, w, label='ZIP (zlib-9)',
        color=COLOR_ZIP, alpha=0.9)
ax2.bar(x + w/2, blobs_blkh, w, label='BLKH v5 (bit-perfect)',
        color=COLOR_BLKH, alpha=0.9)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{s}x{s}' for s in sizes])
ax2.set_xlabel('Image size')
ax2.set_ylabel('Compressed size (bytes, log scale)')
ax2.set_yscale('log')
ax2.set_title('Gaussian Blobs — BLKH wins at every size', fontweight='bold')
ax2.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white')
for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)
for i, s in enumerate(sizes):
    ratio = blobs_zip[i] / blobs_blkh[i]
    ax2.annotate(f'{ratio:.1f}x\nsmaller', xy=(i, blobs_blkh[i]),
                 xytext=(0, -30), textcoords='offset points',
                 ha='center', color=COLOR_ACCENT, fontweight='bold', fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'v5_large_scaling.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print(f"Saved: {out}")
