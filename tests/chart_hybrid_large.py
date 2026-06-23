"""Generate large hybrid benchmark chart."""
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

with open(os.path.join(os.path.dirname(__file__), 'benchmark_large_hybrid_results.json')) as f:
    results = json.load(f)

COLOR_ZIP = '#4a4a6a'
COLOR_V5 = '#5e60ce'
COLOR_HYBRID = '#00d9c0'
COLOR_ACCENT = '#ff006e'

fig, ax = plt.subplots(figsize=(13, 7))
labels = [r['name'] for r in results]
x = np.arange(len(labels))
w = 0.27

ax.bar(x - w, [r['zip'] for r in results], w,
       label='ZIP (zlib-9)', color=COLOR_ZIP, alpha=0.9)
ax.bar(x, [r['v5'] for r in results], w,
       label='BLKH v5 (XOR+zlib residual)', color=COLOR_V5, alpha=0.9)
ax.bar(x + w, [r['hybrid'] for r in results], w,
       label='BLKH v5.8 hybrid (WebP residual)', color=COLOR_HYBRID, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=10)
ax.set_ylabel('Compressed size (bytes, log scale)')
ax.set_yscale('log')
ax.set_title('BLKH v5.8 Hybrid vs v5 vs ZIP — Large Images (all 100% SHA-256 verified)',
             fontweight='bold')
ax.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white', loc='upper left')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Annotate ratios
for i, r in enumerate(results):
    ratio = r['zip'] / r['hybrid']
    ax.annotate(f'{ratio:.1f}x\nsmaller',
                xy=(i + w, r['hybrid']),
                xytext=(0, -35), textcoords='offset points',
                ha='center', color=COLOR_ACCENT, fontweight='bold', fontsize=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'v5_hybrid_large.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print(f"Saved: {out}")
