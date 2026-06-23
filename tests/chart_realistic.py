"""Generate realistic data benchmark chart."""
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

with open(os.path.join(os.path.dirname(__file__), 'benchmark_realistic_results.json')) as f:
    results = json.load(f)

COLOR_ZIP = '#4a4a6a'
COLOR_BLKH = '#00d9c0'
COLOR_ACCENT = '#ff006e'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
labels = [r['name'].replace('_128', '').replace('_', ' ') for r in results]
x = np.arange(len(labels))
w = 0.35

# Panel 1: sizes
ax1.bar(x - w/2, [r['zip_size'] for r in results], w,
        label='ZIP (zlib-9)', color=COLOR_ZIP, alpha=0.9)
ax1.bar(x + w/2, [r['blkh_size'] for r in results], w,
        label='BLKH v5 (bit-perfect)', color=COLOR_BLKH, alpha=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15, ha='right')
ax1.set_ylabel('Compressed size (bytes)')
ax1.set_title('Realistic Data — BLKH v5 vs ZIP (128x128 RGB)',
              fontweight='bold')
ax1.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white')
for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)
# annotate winners
for i, r in enumerate(results):
    color = COLOR_BLKH if r['winner'] == 'BLKH' else COLOR_ACCENT
    ax1.annotate(r['winner'], xy=(i, max(r['zip_size'], r['blkh_size']) * 1.05),
                 ha='center', color=color, fontsize=10, fontweight='bold')

# Panel 2: bit accuracy + PSNR
ax2_twin = ax2.twinx()
bit_pcts = [r['bit_pct'] for r in results]
psnrs = [r['psnr_db'] for r in results]
ax2.bar(x - w/4, bit_pcts, w / 2, label='Bit accuracy (%)',
        color=COLOR_BLKH, alpha=0.85)
ax2.bar(x + w/4, [p / 60 * 100 for p in psnrs], w / 2,
        label='PSNR (scaled, /60*100)', color=COLOR_ACCENT, alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha='right')
ax2.set_ylabel('Bit accuracy (%)')
ax2.set_ylim(0, 100)
ax2.axhline(y=50, color='white', linestyle='--', alpha=0.3, label='Random baseline')
ax2.set_title('Quality Metrics — Bit Accuracy & PSNR', fontweight='bold')
ax2.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white',
           loc='lower right', fontsize=9)
for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'v5_realistic_data.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print(f"Saved: {out}")
