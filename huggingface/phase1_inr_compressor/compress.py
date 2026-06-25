#!/usr/bin/env python3
"""
Black Hole - Phase 1: Compress (v5 PyTorch backend)
====================================================
Replaces the legacy v1 compress.py (which used siren_core.py with 194KB recipes).
Uses siren_v5_torch.py — PyTorch backend, bit-perfect, 5-12x faster than v4.

Usage:
    python compress.py <input_file> <output_recipe.blkh5> [epochs] [lr]

Examples:
    python compress.py image.png out.blkh5
    python compress.py image.png out.blkh5 2000 1e-3
"""
import sys
import os
import time
from pathlib import Path

# Add parent dir to path so we can use the unified CLI
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Reuse the unified blkh CLI
from blkh import cmd_compress
import argparse


def main():
    if len(sys.argv) < 3:
        print("Usage: python compress.py <input_file> <output_recipe.blkh5> [epochs] [lr]")
        print("\nExamples:")
        print("  python compress.py image.png out.blkh5")
        print("  python compress.py image.png out.blkh5 2000 1e-3")
        sys.exit(1)

    args = argparse.Namespace(
        input=sys.argv[1],
        output=sys.argv[2],
        epochs=int(sys.argv[3]) if len(sys.argv) > 3 else 1500,
        lr=float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3,
        bits=8,
        hidden=32,
        layers=2,
        omega=30.0,
        batch_size=2048,
        no_bit_perfect=False,
    )
    cmd_compress(args)


if __name__ == '__main__':
    main()
