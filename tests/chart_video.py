"""Generate video benchmark chart for README."""
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

# Synthetic video results
with open(os.path.join(os.path.dirname(__file__), 'benchmark_video_results.json')) as f:
    synth = json.load(f)

COLOR_ZIP = '#4a4a6a'
COLOR_VIDEO = '#00d9c0'
COLOR_ACCENT = '#ff006e'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: sizes
N_vals = [r['n_frames'] for r in synth]
x = np.arange(len(N_vals))
w = 0.35

ax1.bar(x - w/2, [r['zip_total'] for r in synth], w,
        label='ZIP per-frame (zlib-9)', color=COLOR_ZIP, alpha=0.9)
ax1.bar(x + w/2, [r['video_size'] for r in synth], w,
        label='BLKH Video (temporal SIREN)', color=COLOR_VIDEO, alpha=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{n} frames' for n in N_vals])
ax1.set_xlabel('Number of frames (64x64x3 each)')
ax1.set_ylabel('Total compressed size (bytes)')
ax1.set_title('Synthetic Video (moving blob) — ZIP wins\n(smooth content, LZ77 captures redundancy)',
              fontweight='bold', fontsize=11)
ax1.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white')
for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)

# Panel 2: realistic content (from manual benchmark)
N_real = [8, 16]
zip_real = [90969, 181946]
video_real = [59792, 106314]
x2 = np.arange(len(N_real))

ax2.bar(x2 - w/2, zip_real, w,
        label='ZIP per-frame', color=COLOR_ZIP, alpha=0.9)
ax2.bar(x2 + w/2, video_real, w,
        label='BLKH Video', color=COLOR_VIDEO, alpha=0.9)
ax2.set_xticks(x2)
ax2.set_xticklabels([f'{n} frames' for n in N_real])
ax2.set_xlabel('Number of frames (64x64x3 each, realistic content)')
ax2.set_ylabel('Total compressed size (bytes)')
ax2.set_title('Realistic Video (gradient+noise+motion) — BLKH wins\n1.5-1.7x smaller than ZIP',
              fontweight='bold', fontsize=11, color='#00ff88')
ax2.legend(facecolor='#1a1a2e', edgecolor='#7b2cbf', labelcolor='white')
for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)

# Annotate ratios
for i, (z, v) in enumerate(zip(zip_real, video_real)):
    ratio = z / v
    ax2.annotate(f'{ratio:.2f}x\nsmaller',
                 xy=(i + w/2, v),
                 xytext=(0, -30), textcoords='offset points',
                 ha='center', color=COLOR_ACCENT, fontweight='bold', fontsize=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'v5_video_benchmark.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()
print(f"Saved: {out}")
