# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

"""
blkh_web_demo.py — Interactive BLKH Web Demo (Gradio)
======================================================
Optimized web interface for BLKH compression:
  - Turbo mode by default (sub-1s on 256x256)
  - Side-by-side original vs reconstructed
  - BLKH vs ZIP vs PNG vs WebP comparison
  - SHA-256 verification
  - Download .blkh8 recipe

Run:
    python blkh_web_demo.py
    # Opens at http://localhost:7860
"""
import sys
import os
import io
import time
import zlib
import hashlib
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1_inr_compressor'))

import numpy as np
import torch
torch.set_num_threads(max(4, torch.get_num_threads()))

import gradio as gr
from PIL import Image

from siren_v5_hybrid import HybridCompressor


def compress_image(input_image, mode, use_amp):
    """Compress uploaded image with BLKH hybrid mode."""
    if input_image is None:
        return None, None, "Please upload an image first.", "", "", "", ""

    # Convert to numpy uint8 RGB
    if isinstance(input_image, str):
        img = np.array(Image.open(input_image).convert('RGB'), dtype=np.uint8)
    elif isinstance(input_image, np.ndarray):
        img = input_image
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = img.astype(np.uint8)
    else:
        img = np.array(input_image.convert('RGB'), dtype=np.uint8)

    orig_size = img.nbytes

    # ZIP baseline
    zip_size = len(zlib.compress(img.tobytes(), 9))

    # PNG baseline
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='PNG', optimize=True)
    png_size = len(buf.getvalue())

    # WebP lossless baseline
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='WebP', lossless=True)
    webp_size = len(buf.getvalue())

    # Configure based on mode
    if mode == "Turbo (~1s)":
        epochs, lr, bs, patience = 200, 3e-3, 16384, 3
    elif mode == "Fast (~3s)":
        epochs, lr, bs, patience = 400, 2e-3, 16384, 5
    elif mode == "Quality (~6s)":
        epochs, lr, bs, patience = 800, 1e-3, 8192, 5
    else:
        epochs, lr, bs, patience = 200, 3e-3, 16384, 3

    # BLKH compress
    comp = HybridCompressor(auto_tune=True, residual_codec='webp')
    t0 = time.time()
    try:
        res = comp.compress_bitperfect(
            img, epochs=epochs, lr=lr, bits=8,
            batch_size=bs, use_amp=bool(use_amp),
            patience=patience, verbose=False
        )
    except Exception as e:
        return None, None, f"Error: {str(e)}", "", "", "", ""

    dt = time.time() - t0

    # Decompress and verify
    recon, meta = HybridCompressor.decompress(res['recipe_bytes'])
    sha_ok = meta['exact_match']

    # Save recipe to temp file for download
    tmp = tempfile.NamedTemporaryFile(suffix='.blkh8', delete=False)
    tmp.write(res['recipe_bytes'])
    tmp.close()

    # Format output
    blkh_size = res['recipe_size']
    ratio = orig_size / blkh_size
    vs_zip = zip_size / blkh_size if blkh_size > 0 else 0
    vs_png = png_size / blkh_size if blkh_size > 0 else 0
    vs_webp = webp_size / blkh_size if blkh_size > 0 else 0

    # Determine winner
    formats = {'ZIP': zip_size, 'PNG': png_size, 'WebP-L': webp_size, 'BLKH': blkh_size}
    winner = min(formats, key=formats.get)

    result_text = f"""BLKH Compression Results
{'=' * 50}
  Original:        {orig_size:>10,} bytes
  ZIP (zlib-9):    {zip_size:>10,} bytes  ({orig_size/zip_size:.2f}x)
  PNG (lossless):  {png_size:>10,} bytes  ({orig_size/png_size:.2f}x)
  WebP (lossless): {webp_size:>10,} bytes  ({orig_size/webp_size:.2f}x)
  BLKH (hybrid):   {blkh_size:>10,} bytes  ({ratio:.2f}x)

  BLKH vs ZIP:     {vs_zip:.2f}x {'✓ BLKH wins' if vs_zip > 1 else '✗ ZIP wins'}
  BLKH vs PNG:     {vs_png:.2f}x {'✓ BLKH wins' if vs_png > 1 else '✗ PNG wins'}
  BLKH vs WebP:    {vs_webp:.2f}x {'✓ BLKH wins' if vs_webp > 1 else '✗ WebP wins'}

  Best format:     {winner}
  Bit accuracy:    {res['model_bit_accuracy']:.1f}%
  SHA-256:         {'✓ VERIFIED' if sha_ok else '✗ FAILED'}
  Encoding time:   {dt:.2f}s
  Mode:            {mode}
"""

    return (
        Image.fromarray(img),
        Image.fromarray(recon),
        result_text,
        f"{blkh_size:,} B ({ratio:.2f}x)",
        f"{vs_zip:.2f}x",
        f"{dt:.2f}s",
        tmp.name,
    )


# Gradio UI
with gr.Blocks(title="Black Hole (BLKH) — Neural Compression") as demo:
    gr.Markdown("""
    # 🕳️ Black Hole (BLKH) — Neural Implicit Compression

    Compress images with SIREN + hybrid WebP residual. **100% bit-perfect** (SHA-256 verified).

    Upload an image, choose speed mode, and click **Compress**.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Image", type='pil')
            mode = gr.Radio(
                ["Turbo (~1s)", "Fast (~3s)", "Quality (~6s)"],
                value="Turbo (~1s)",
                label="Speed Mode"
            )
            use_amp = gr.Checkbox(value=True, label="AMP (mixed precision)")
            compress_btn = gr.Button("🚀 Compress with BLKH", variant='primary', size='lg')

        with gr.Column(scale=2):
            result_text = gr.Textbox(label="Results", lines=20, max_lines=25)
            with gr.Row():
                orig_display = gr.Image(label="Original")
                recon_display = gr.Image(label="Reconstructed (BLKH)")

    with gr.Row():
        blkh_size_out = gr.Textbox(label="BLKH Size", scale=1)
        vs_zip_out = gr.Textbox(label="vs ZIP", scale=1)
        time_out = gr.Textbox(label="Encoding Time", scale=1)
        recipe_file = gr.File(label="Download .blkh8", scale=1)

    compress_btn.click(
        fn=compress_image,
        inputs=[input_img, mode, use_amp],
        outputs=[orig_display, recon_display, result_text,
                 blkh_size_out, vs_zip_out, time_out, recipe_file]
    )

    gr.Markdown("""
    ---
    **Black Hole (BLKH)** v5.14 | Created by **Darlan Pereira da Silva** | [GitHub](https://github.com/Kronos1027/black-hole)
    Commercial use requires a separate license. Contact: darlan1027pc@gmail.com
    """)


if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
