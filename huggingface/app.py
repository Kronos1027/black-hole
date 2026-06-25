# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# HuggingFace Spaces app.py — BLKH v5.27 Web Demo
# Deploy: Upload this file + requirements.txt to a new Gradio Space
"""
🕳️ Black Hole (BLKH) — Neural Implicit Compression v5.27
==========================================================
Compress images with 6 modes:
- Instant/Turbo/Quality: Hybrid SIREN + WebP residual (bit-perfect, slow)
- Wavelet v3: TRUE bit-perfect wavelet+float16+brotli (fast, smooth images)
- Photo v5.21: YCbCr 4:2:0 + brotli (lossy, 2-2.7x smaller than PNG)
- DCT v5.22: JPEG-like DCT + brotli (lossy, 20-50x smaller than PNG)

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import io
import time
import zlib
import tempfile

# Add phase1_inr_compressor to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phase1_inr_compressor'))

import numpy as np
import torch
torch.set_num_threads(max(2, min(4, torch.get_num_threads())))

import gradio as gr
from PIL import Image

# Lazy imports for compression modes
_HYBRID_IMPORTED = False
def _ensure_hybrid():
    global _HYBRID_IMPORTED
    if not _HYBRID_IMPORTED:
        global HybridCompressor
        from siren_v5_hybrid import HybridCompressor
        _HYBRID_IMPORTED = True


def compress_image(input_image, mode, use_amp):
    """Compress uploaded image with selected BLKH mode."""
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

    # Baselines
    zip_size = len(zlib.compress(img.tobytes(), 9))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='PNG', optimize=True)
    png_size = len(buf.getvalue())
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='WebP', lossless=True)
    webp_size = len(buf.getvalue())

    # Mode selection
    if mode == "Wavelet v3 (lossless, fast)":
        from siren_v5_wavelet_v3 import WaveletINRCompressorV3
        t0 = time.time()
        try:
            comp = WaveletINRCompressorV3(wavelet='auto', level='auto', lossless=True,
                                            codec='brotli', combined=True, parallel=False)
            res = comp.compress(img, verbose=False)
            recon, meta = WaveletINRCompressorV3.decompress(res['recipe_bytes'])
            sha_ok = meta['sha256_match']
        except Exception as e:
            return None, None, "Error: %s" % str(e), "", "", "", ""
        dt = time.time() - t0
        tmp = tempfile.NamedTemporaryFile(suffix='.blkw3', delete=False)
        tmp.write(res['recipe_bytes'])
        tmp.close()
        blkh_size = res['recipe_size']
        ratio = orig_size / blkh_size
        vs_zip = zip_size / blkh_size
        vs_png = png_size / blkh_size
        vs_webp = webp_size / blkh_size
        formats = {'ZIP': zip_size, 'PNG': png_size, 'WebP-L': webp_size, 'BLKH': blkh_size}
        winner = min(formats, key=formats.get)
        result_text = """BLKH Wavelet v3 (Lossless) Results
==================================================
  Original:        %10d bytes
  ZIP (zlib-9):    %10d bytes  (%.2fx)
  PNG (lossless):  %10d bytes  (%.2fx)
  WebP (lossless): %10d bytes  (%.2fx)
  BLKH v5.20:      %10d bytes  (%.2fx)

  BLKH vs ZIP:     %.2fx %s
  BLKH vs PNG:     %.2fx %s
  BLKH vs WebP:    %.2fx %s

  Best format:     %s
  Wavelet:         %s L%d
  SHA-256:         %s
  Encoding time:   %.2fs
  Mode:            %s
""" % (orig_size, zip_size, orig_size/zip_size, png_size, orig_size/png_size,
       webp_size, orig_size/webp_size, blkh_size, ratio,
       vs_zip, '✓ BLKH wins' if vs_zip > 1 else '✗ ZIP wins',
       vs_png, '✓ BLKH wins' if vs_png > 1 else '✗ PNG wins',
       vs_webp, '✓ BLKH wins' if vs_webp > 1 else '✗ WebP wins',
       winner, res['wavelet'], res['level'],
       '✓ VERIFIED (bit-perfect)' if sha_ok else '✗ FAILED',
       dt, mode)
        return (Image.fromarray(img), Image.fromarray(recon), result_text,
                "%d B (%.2fx)" % (blkh_size, ratio), "%.2fx" % vs_zip,
                "%.2fs" % dt, tmp.name)

    elif mode == "Photo v5.21 (lossy, beats PNG)":
        from siren_v5_photo import PhotoCompressor
        t0 = time.time()
        try:
            comp = PhotoCompressor(subsampling='420', codec='brotli')
            res = comp.compress(img, verbose=False)
            recon, meta = PhotoCompressor.decompress(res['recipe_bytes'])
        except Exception as e:
            return None, None, "Error: %s" % str(e), "", "", "", ""
        dt = time.time() - t0
        mse = np.mean((img.astype(float) - recon.astype(float))**2)
        psnr = 10*np.log10(255**2 / max(mse, 1e-10))
        tmp = tempfile.NamedTemporaryFile(suffix='.blkp', delete=False)
        tmp.write(res['recipe_bytes'])
        tmp.close()
        blkh_size = res['recipe_size']
        ratio = orig_size / blkh_size
        vs_zip = zip_size / blkh_size
        vs_png = png_size / blkh_size
        vs_webp = webp_size / blkh_size
        formats = {'ZIP': zip_size, 'PNG': png_size, 'WebP-L': webp_size, 'BLKH': blkh_size}
        winner = min(formats, key=formats.get)
        result_text = """BLKH Photo v5.21 (Lossy) Results
==================================================
  Original:        %10d bytes
  ZIP (zlib-9):    %10d bytes  (%.2fx)
  PNG (lossless):  %10d bytes  (%.2fx)
  WebP (lossless): %10d bytes  (%.2fx)
  BLKH photo:      %10d bytes  (%.2fx)

  BLKH vs ZIP:     %.2fx %s
  BLKH vs PNG:     %.2fx %s
  BLKH vs WebP:    %.2fx %s

  Best format:     %s
  PSNR:            %.1f dB (lossy)
  Encoding time:   %.2fs
  Mode:            %s
""" % (orig_size, zip_size, orig_size/zip_size, png_size, orig_size/png_size,
       webp_size, orig_size/webp_size, blkh_size, ratio,
       vs_zip, '✓ BLKH wins' if vs_zip > 1 else '✗ ZIP wins',
       vs_png, '✓ BLKH wins' if vs_png > 1 else '✗ PNG wins',
       vs_webp, '✓ BLKH wins' if vs_webp > 1 else '✗ WebP wins',
       winner, psnr, dt, mode)
        return (Image.fromarray(img), Image.fromarray(recon), result_text,
                "%d B (%.2fx)" % (blkh_size, ratio), "%.2fx" % vs_zip,
                "%.2fs" % dt, tmp.name)

    elif mode == "DCT v5.22 (max compression)":
        from siren_v5_dct import DCTCompressor
        t0 = time.time()
        try:
            comp = DCTCompressor(quality=0.9, codec='brotli')
            res = comp.compress(img, verbose=False)
            recon, meta = DCTCompressor.decompress(res['recipe_bytes'])
        except Exception as e:
            return None, None, "Error: %s" % str(e), "", "", "", ""
        dt = time.time() - t0
        mse = np.mean((img.astype(float) - recon.astype(float))**2)
        psnr = 10*np.log10(255**2 / max(mse, 1e-10))
        tmp = tempfile.NamedTemporaryFile(suffix='.blkd', delete=False)
        tmp.write(res['recipe_bytes'])
        tmp.close()
        blkh_size = res['recipe_size']
        ratio = orig_size / blkh_size
        vs_zip = zip_size / blkh_size
        vs_png = png_size / blkh_size
        vs_webp = webp_size / blkh_size
        formats = {'ZIP': zip_size, 'PNG': png_size, 'WebP-L': webp_size, 'BLKH': blkh_size}
        winner = min(formats, key=formats.get)
        result_text = """BLKH DCT v5.22 (Maximum Compression, Lossy) Results
==================================================
  Original:        %10d bytes
  ZIP (zlib-9):    %10d bytes  (%.2fx)
  PNG (lossless):  %10d bytes  (%.2fx)
  WebP (lossless): %10d bytes  (%.2fx)
  BLKH DCT:        %10d bytes  (%.2fx)

  BLKH vs ZIP:     %.2fx %s
  BLKH vs PNG:     %.2fx %s
  BLKH vs WebP:    %.2fx %s

  Best format:     %s
  PSNR:            %.1f dB (lossy, quality=0.9)
  Encoding time:   %.2fs
  Mode:            %s
""" % (orig_size, zip_size, orig_size/zip_size, png_size, orig_size/png_size,
       webp_size, orig_size/webp_size, blkh_size, ratio,
       vs_zip, '✓ BLKH wins' if vs_zip > 1 else '✗ ZIP wins',
       vs_png, '✓ BLKH wins' if vs_png > 1 else '✗ PNG wins',
       vs_webp, '✓ BLKH wins' if vs_webp > 1 else '✗ WebP wins',
       winner, psnr, dt, mode)
        return (Image.fromarray(img), Image.fromarray(recon), result_text,
                "%d B (%.2fx)" % (blkh_size, ratio), "%.2fx" % vs_zip,
                "%.2fs" % dt, tmp.name)

    elif mode == "Fast v5.23 (fastest)":
        from siren_v5_fast import FastDCTCompressor
        t0 = time.time()
        try:
            comp = FastDCTCompressor(quality=0.9, speed='fast')
            res = comp.compress(img, verbose=False)
            recon, meta = FastDCTCompressor.decompress(res['recipe_bytes'])
        except Exception as e:
            return None, None, "Error: %s" % str(e), "", "", "", ""
        dt = time.time() - t0
        mse = np.mean((img.astype(float) - recon.astype(float))**2)
        psnr = 10*np.log10(255**2 / max(mse, 1e-10))
        tmp = tempfile.NamedTemporaryFile(suffix='.blkf', delete=False)
        tmp.write(res['recipe_bytes'])
        tmp.close()
        blkh_size = res['recipe_size']
        ratio = orig_size / blkh_size
        vs_zip = zip_size / blkh_size
        vs_png = png_size / blkh_size
        vs_webp = webp_size / blkh_size
        formats = {'ZIP': zip_size, 'PNG': png_size, 'WebP-L': webp_size, 'BLKH': blkh_size}
        winner = min(formats, key=formats.get)
        throughput = orig_size / dt / 1024 / 1024 if dt > 0 else 0
        result_text = """BLKH Fast v5.23 (Speed-Optimized, Lossy) Results
==================================================
  Original:        %10d bytes
  ZIP (zlib-9):    %10d bytes  (%.2fx)
  PNG (lossless):  %10d bytes  (%.2fx)
  WebP (lossless): %10d bytes  (%.2fx)
  BLKH fast:       %10d bytes  (%.2fx)

  BLKH vs ZIP:     %.2fx %s
  BLKH vs PNG:     %.2fx %s
  BLKH vs WebP:    %.2fx %s

  Best format:     %s
  PSNR:            %.1f dB (lossy, quality=0.9)
  Encoding time:   %.2fs (%.1f MB/s)
  Mode:            %s
""" % (orig_size, zip_size, orig_size/zip_size, png_size, orig_size/png_size,
       webp_size, orig_size/webp_size, blkh_size, ratio,
       vs_zip, '✓ BLKH wins' if vs_zip > 1 else '✗ ZIP wins',
       vs_png, '✓ BLKH wins' if vs_png > 1 else '✗ PNG wins',
       vs_webp, '✓ BLKH wins' if vs_webp > 1 else '✗ WebP wins',
       winner, psnr, dt, throughput, mode)
        return (Image.fromarray(img), Image.fromarray(recon), result_text,
                "%d B (%.2fx)" % (blkh_size, ratio), "%.2fx" % vs_zip,
                "%.2fs" % dt, tmp.name)

    else:
        # Hybrid modes (Instant/Turbo/Quality)
        _ensure_hybrid()
        if mode == "Instant (~0.5s)":
            epochs, lr, bs, patience, codec = 100, 4e-3, 16384, 3, 'png'
        elif mode == "Turbo (~1s)":
            epochs, lr, bs, patience, codec = 200, 3e-3, 16384, 3, 'webp'
        elif mode == "Quality (~6s)":
            epochs, lr, bs, patience, codec = 800, 1e-3, 8192, 5, 'webp'
        else:
            epochs, lr, bs, patience, codec = 100, 4e-3, 16384, 3, 'png'

        comp = HybridCompressor(auto_tune=True, residual_codec=codec)
        t0 = time.time()
        try:
            res = comp.compress_bitperfect(
                img, epochs=epochs, lr=lr, bits=8,
                batch_size=bs, use_amp=bool(use_amp),
                patience=patience, verbose=False
            )
        except Exception as e:
            return None, None, "Error: %s" % str(e), "", "", "", ""
        dt = time.time() - t0

        recon, meta = HybridCompressor.decompress(res['recipe_bytes'])
        sha_ok = meta['exact_match']

        tmp = tempfile.NamedTemporaryFile(suffix='.blkh8', delete=False)
        tmp.write(res['recipe_bytes'])
        tmp.close()

        blkh_size = res['recipe_size']
        ratio = orig_size / blkh_size
        vs_zip = zip_size / blkh_size if blkh_size > 0 else 0
        vs_png = png_size / blkh_size if blkh_size > 0 else 0
        vs_webp = webp_size / blkh_size if blkh_size > 0 else 0

        formats = {'ZIP': zip_size, 'PNG': png_size, 'WebP-L': webp_size, 'BLKH': blkh_size}
        winner = min(formats, key=formats.get)

        result_text = """BLKH Hybrid (Bit-Perfect) Results
==================================================
  Original:        %10d bytes
  ZIP (zlib-9):    %10d bytes  (%.2fx)
  PNG (lossless):  %10d bytes  (%.2fx)
  WebP (lossless): %10d bytes  (%.2fx)
  BLKH (hybrid):   %10d bytes  (%.2fx)

  BLKH vs ZIP:     %.2fx %s
  BLKH vs PNG:     %.2fx %s
  BLKH vs WebP:    %.2fx %s

  Best format:     %s
  Bit accuracy:    %.1f%%
  SHA-256:         %s
  Encoding time:   %.2fs
  Mode:            %s
""" % (orig_size, zip_size, orig_size/zip_size, png_size, orig_size/png_size,
       webp_size, orig_size/webp_size, blkh_size, ratio,
       vs_zip, '✓ BLKH wins' if vs_zip > 1 else '✗ ZIP wins',
       vs_png, '✓ BLKH wins' if vs_png > 1 else '✗ PNG wins',
       vs_webp, '✓ BLKH wins' if vs_webp > 1 else '✗ WebP wins',
       winner, res['model_bit_accuracy'],
       '✓ VERIFIED' if sha_ok else '✗ FAILED',
       dt, mode)

        return (
            Image.fromarray(img),
            Image.fromarray(recon),
            result_text,
            "%d B (%.2fx)" % (blkh_size, ratio),
            "%.2fx" % vs_zip,
            "%.2fs" % dt,
            tmp.name,
        )


# Gradio UI
with gr.Blocks(title="Black Hole (BLKH) — Neural Compression v5.27", theme=gr.Soft()) as demo:
    gr.Markdown("""
    # 🕳️ Black Hole (BLKH) v5.27 — Neural Implicit Compression

    Compress images with BLKH. Choose from 6 modes:
    - **Instant/Turbo/Quality**: Hybrid SIREN + WebP residual (bit-perfect, slow)
    - **Wavelet v3**: TRUE bit-perfect wavelet+float16+brotli (fast, smooth images)
    - **Photo v5.21**: YCbCr 4:2:0 + brotli (lossy, 2-2.7x smaller than PNG on photos)
    - **DCT v5.22**: JPEG-like DCT + brotli (lossy, 20-50x smaller than PNG, max compression)
    - **Fast v5.23**: Speed-optimized (3x faster than ZIP!)

    Upload an image, choose mode, and click **Compress**.

    *Author: Darlan Pereira da Silva (Kronos1027) — [GitHub](https://github.com/Kronos1027/black-hole)*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Image", type='pil')
            mode = gr.Radio(
                ["Fast v5.23 (fastest)",
                 "DCT v5.22 (max compression)",
                 "Photo v5.21 (lossy, beats PNG)",
                 "Wavelet v3 (lossless, fast)",
                 "Instant (~0.5s)",
                 "Turbo (~1s)",
                 "Quality (~6s)"],
                value="Fast v5.23 (fastest)",
                label="Compression Mode"
            )
            use_amp = gr.Checkbox(value=True, label="AMP (mixed precision, hybrid modes only)")
            compress_btn = gr.Button("🚀 Compress with BLKH", variant='primary', size='lg')

        with gr.Column(scale=2):
            result_text = gr.Textbox(label="Results", lines=22, max_lines=30)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Original")
            orig_display = gr.Image(label="Original", show_label=False)
        with gr.Column():
            gr.Markdown("### Reconstructed")
            recon_display = gr.Image(label="Reconstructed", show_label=False)

    with gr.Row():
        blkh_size_out = gr.Textbox(label="BLKH Size", interactive=False)
        vs_zip_out = gr.Textbox(label="vs ZIP", interactive=False)
        time_out = gr.Textbox(label="Time", interactive=False)
        download_file = gr.File(label="Download Recipe")

    compress_btn.click(
        compress_image,
        inputs=[input_img, mode, use_amp],
        outputs=[orig_display, recon_display, result_text,
                 blkh_size_out, vs_zip_out, time_out, download_file]
    )

    gr.Markdown("""
    ---
    ### Performance Comparison

    | Mode | Use Case | Compression | Speed |
    |------|----------|-------------|-------|
    | Fast v5.23 | Real-time | 14-50x vs ZIP | 3-4x faster than ZIP |
    | DCT v5.22 | Max compression | 20-50x vs PNG | 0.04s |
    | Photo v5.21 | Photos visually lossless | 2-4x vs PNG | 0.02s |
    | Wavelet v3 | Smooth lossless | 2-3x vs ZIP | 0.4-2s |
    | Hybrid | Bit-perfect any content | varies | 0.5-6s |

    ### License
    MIT (research/education) + commercial license required.
    """)


if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
