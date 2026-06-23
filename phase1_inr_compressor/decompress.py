#!/usr/bin/env python3
"""
Black Hole - Phase 1: Decompress (v5 PyTorch backend)
======================================================
Replaces the legacy v1 decompress.py. Uses siren_v5_torch.py.
SHA-256 verified roundtrip — guarantees 100% bit-perfect recovery.

Usage:
    python decompress.py <recipe.blkh5> <output_file>

Examples:
    python decompress.py out.blkh5 recovered.png
    python decompress.py out.blkh5 recovered.raw
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from blkh import cmd_decompress
import argparse


def main():
    if len(sys.argv) < 3:
        print("Usage: python decompress.py <recipe.blkh5> <output_file>")
        print("\nExamples:")
        print("  python decompress.py out.blkh5 recovered.png")
        sys.exit(1)

    args = argparse.Namespace(
        input=sys.argv[1],
        output=sys.argv[2],
    )
    cmd_decompress(args)


if __name__ == '__main__':
    main()
