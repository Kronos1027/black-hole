"""Generate atlas scaling chart for README."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

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

# Load atlas results
with open(os.path.join(os.path.dirname(__file__), 'benchmark_atlas_results.json')) as f:
    results = json.load(f)

COLOR_ZIP = '#4a4a6a'
COLOR_ATLAS = '#00d9c0'
COLOR_ACCENT = '#ff006e'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: total size comparison
N = [r['n'] for r in results]
w = 0.35
x = np.arange(len(N))
ax1.bar(x - w/2, [r['zip_total'] for r in results], w,
        label='ZIP per-file (zlib-9)', color=COLOR_ZIP, alpha=0.9)
ax1.bar(x + w/2, [r['atlas_total'] for r in results], w,
        label='BLKH Atlas (bit-perfect)', color=COLOR_ATLAS, alpha=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels([f'N={n}' for n in N])
ax1.set_ylabel('Total compressed size (bytes)')
ax1.set_title('Atlas Total Size vs ZIP', fontweight='bold')
ax1.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white')
for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)

# Panel 2: atlas vs ZIP ratio + bit accuracy
ax2_twin = ax2.twinx()
ratios = [r['atlas_vs_zip'] for r in results]
bit_pcts = [r['avg_bit_pct'] for r in results]
ax2.bar(x, ratios, 0.5, label='Atlas/ZIP ratio (>1 = BLKH wins)',
        color=COLOR_ATLAS, alpha=0.85)
ax2.axhline(y=1.0, color=COLOR_ACCENT, linestyle='--', alpha=0.5,
            label='Break-even (1.0)')
ax2.set_xticks(x)
ax2.set_xticklabels([f'N={n}' for n in N])
ax2.set_ylabel('ZIP / Atlas ratio (>1 = BLKH wins)', color=COLOR_ATLAS)
ax2_twin.plot(x, bit_pcts, color=COLOR_ACCENT, marker='o', linewidth=2,
              markersize=10, label='Bit accuracy (%)')
ax2_twin.set_ylabel('Model bit accuracy (%)', color=COLOR_ACCENT)
ax2_twin.set_ylim(0, 100)
ax2.set_title('Atlas Scaling — Win Margin vs Bit Accuracy',
              fontweight='bold')
ax2_twin.tick_params(axis='y', labelcolor=COLOR_ACCENT)
for spine in ['top']:
    ax2.spines[spine].set_visible(False)
ax2_twin.spines['top'].set_visible(False)

# combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2,
           facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white',
           loc='lower left', fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'v5_atlas_scaling.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print(f"Saved: {out}")
