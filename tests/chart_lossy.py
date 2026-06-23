"""Generate lossy mode benchmark chart for README."""
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

with open(os.path.join(os.path.dirname(__file__), 'benchmark_lossy_results.json')) as f:
    results = json.load(f)

# Get only the BLKH INT4 balanced entries (one per photo)
balanced = [r for r in results if r['mode'] == 'BLKH INT4 balanced']
if not balanced:
    balanced = [r for r in results if r['mode'] == 'BLKH INT4 aggressive'][:5]

COLOR_ZIP = '#4a4a6a'
COLOR_JPEG = '#ff8c42'
COLOR_WEBP = '#00d9c0'
COLOR_BLKH = '#7b2cbf'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

labels = [r['name'] for r in balanced]
x = np.arange(len(labels))
w = 0.2

# Panel 1: sizes
ax1.bar(x - 1.5*w, [r['zip_size'] for r in balanced], w,
        label='ZIP (lossless)', color=COLOR_ZIP, alpha=0.9)
ax1.bar(x - 0.5*w, [r['jpeg_q85_size'] for r in balanced], w,
        label='JPEG q=85 (lossy)', color=COLOR_JPEG, alpha=0.9)
ax1.bar(x + 0.5*w, [r['webp_q85_size'] for r in balanced], w,
        label='WebP q=85 (lossy)', color=COLOR_WEBP, alpha=0.9)
ax1.bar(x + 1.5*w, [r['blkh_size'] for r in balanced], w,
        label='BLKH lossy (INT4)', color=COLOR_BLKH, alpha=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel('Compressed size (bytes)')
ax1.set_title('Lossy Mode: BLKH vs JPEG vs WebP\n(smaller is better)', fontweight='bold')
ax1.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white', fontsize=9)
for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)

# Annotate winners
for i, r in enumerate(balanced):
    sizes = {'ZIP': r['zip_size'], 'JPEG': r['jpeg_q85_size'],
             'WebP': r['webp_q85_size'], 'BLKH': r['blkh_size']}
    winner = min(sizes, key=sizes.get)
    color = '#00ff88' if winner == 'BLKH' else '#ff006e'
    ax1.annotate(winner, xy=(i, min(sizes.values()) * 0.85),
                 ha='center', color=color, fontsize=10, fontweight='bold')

# Panel 2: PSNR (quality)
ax2.bar(x - 0.5*w, [r['jpeg_q85_psnr'] for r in balanced], w,
        label='JPEG q=85', color=COLOR_JPEG, alpha=0.9)
ax2.bar(x + 0.5*w, [r['blkh_psnr_db'] for r in balanced], w,
        label='BLKH lossy', color=COLOR_BLKH, alpha=0.9)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel('PSNR (dB, higher is better)')
ax2.set_title('Quality: BLKH vs JPEG\n(BLKH trades quality for smaller size)',
              fontweight='bold')
ax2.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white', fontsize=9)
for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)
ax2.axhline(y=30, color='white', linestyle='--', alpha=0.3, label='30 dB (good)')

# Annotate PSNR values
for i, r in enumerate(balanced):
    ax2.annotate(f"{r['blkh_psnr_db']:.0f}",
                 xy=(i + 0.5*w, r['blkh_psnr_db']),
                 xytext=(0, 3), textcoords='offset points',
                 ha='center', color='white', fontsize=9)
    ax2.annotate(f"{r['jpeg_q85_psnr']:.0f}",
                 xy=(i - 0.5*w, r['jpeg_q85_psnr']),
                 xytext=(0, 3), textcoords='offset points',
                 ha='center', color='white', fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'v5_lossy_vs_jpeg_webp.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print(f"Saved: {out}")
