#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""Generate scaling law chart for Phase 1 results."""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # Load results
    results_path = os.path.join(os.path.dirname(__file__), 'phase1_scaling_results.json')
    with open(results_path) as f:
        data = json.load(f)

    results = data['results']
    n_values = [r['n_images'] for r in results]
    baseline_sizes = [r['baseline_size'] for r in results]
    bhuh_sizes = [r['bhuh_size'] for r in results]
    zip_sizes = [r['zip_size'] for r in results]
    improvements = [r['improvement'] for r in results]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Left: Size comparison (log scale)
    ax1.plot(n_values, baseline_sizes, 'ro-', label='Separate SIRENs (baseline)', linewidth=2, markersize=8)
    ax1.plot(n_values, bhuh_sizes, 'bs-', label='BHUH (shared roots)', linewidth=2, markersize=8)
    ax1.plot(n_values, zip_sizes, 'g^-', label='ZIP (zlib-9)', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Images', fontsize=12)
    ax1.set_ylabel('Total Compressed Size (bytes)', fontsize=12)
    ax1.set_title('Phase 1: Compression Size vs Number of Images', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_values)

    # Right: Improvement factor
    ax2.bar(range(len(n_values)), improvements, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
            edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(n_values)))
    ax2.set_xticklabels([f'N={n}' for n in n_values], fontsize=11)
    ax2.set_ylabel('Improvement (x vs separate SIRENs)', fontsize=12)
    ax2.set_title('Phase 1: Shared Roots Improvement Factor', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(improvements):
        ax2.text(i, v + 0.3, f'{v:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add annotation
    ax2.annotate('Scaling Law:\nImprovement ~ N/3',
                xy=(3, improvements[-1]),
                xytext=(1.5, improvements[-1] + 2),
                fontsize=11, fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Black Hole Universe — Phase 1: Multi-File SIREN Scaling Law',
                fontsize=16, fontweight='bold', y=1.02)

    output_path = os.path.join(os.path.dirname(__file__), 'phase1_scaling_chart.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
