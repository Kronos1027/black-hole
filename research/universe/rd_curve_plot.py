#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Generate RD plot: BHUH Hierarchical curve vs COIN point."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except: pass
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Data from experiments 8, 10, 12, 13
# BHUH Hierarchical curve (K=1 to K=50)
bhuh_k = [1, 5, 10, 15, 20, 25, 50]
bhuh_bytes = [2884, 13999, 16848, 22407, 25199, 25146, 56450]
bhuh_psnr = [17.11, 22.43, 23.24, 23.78, 24.43, 24.95, 27.13]

# COIN single point
coin_bytes = 85601
coin_psnr = 28.10

# Convert to bits per pixel (N=100 images, 64x64 each)
N_images = 100
pixels_per_image = 64 * 64
total_pixels = N_images * pixels_per_image

bhuh_bpp = [b * 8 / total_pixels for b in bhuh_bytes]
coin_bpp = coin_bytes * 8 / total_pixels

fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

# BHUH curve
ax.plot(bhuh_bpp, bhuh_psnr, 'b-o', linewidth=2.5, markersize=10,
        label='BHUH Hierarchical (tunable K)', zorder=5)

# Annotate K values
for i, k in enumerate(bhuh_k):
    ax.annotate(f'K={k}', (bhuh_bpp[i], bhuh_psnr[i]),
                textcoords="offset points", xytext=(10, 5),
                fontsize=9, fontweight='bold')

# COIN point
ax.scatter([coin_bpp], [coin_psnr], color='red', s=200, marker='D',
           zorder=5, label='COIN (separate SIRENs)')
ax.annotate('COIN', (coin_bpp, coin_psnr),
            textcoords="offset points", xytext=(10, 5),
            fontsize=11, fontweight='bold', color='red')

# Labels
ax.set_xlabel('Bits per pixel (bpp)', fontsize=13)
ax.set_ylabel('PSNR (dB)', fontsize=13)
ax.set_title('BHUH Hierarchical vs COIN — Rate-Distortion Trade-off\n'
             '(N=100 real photographs, 64×64 grayscale)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=12)

# Set axis limits
ax.set_xlim(0, max(coin_bpp, max(bhuh_bpp)) * 1.15)
ax.set_ylim(15, 30)

# Add annotation about the curve
ax.text(0.02, 0.98, 'BHUH provides a TUNABLE curve\n'
        'COIN is a single point\n'
        'K controls the trade-off',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig('research/universe/rd_curve_hierarchical.png', dpi=150, bbox_inches='tight')
plt.close()
print("RD plot saved to research/universe/rd_curve_hierarchical.png")
