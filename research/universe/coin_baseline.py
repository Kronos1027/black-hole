# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
COIN baseline — Compression with Implicit Neural Representations
=================================================================
Reference implementation of COIN (Dupont et al., 2021) for fair
comparison with BLKH.

COIN strategy:
  1. Train one SIREN per image (no residual, no meta-learning)
  2. Quantize weights to INT8
  3. Compress quantized weights with zlib

This is the SIMPLEST neural compression baseline. BLKH should beat
COIN because:
  - BLKH uses hybrid residual coding (bit-perfect)
  - BLKH has distillation (smaller student)
  - BLKH has INT4 quantization (vs COIN's INT8)

But we MUST verify this empirically, not assume it.

Reference: Dupont et al., "COIN: Compression with Implicit Neural
Representations", arXiv 2103.03123, 2021.
"""
import numpy as np
import time
import io
import zlib
import struct
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1_inr_compressor'))


def compute_psnr(orig, recon):
    err = orig.astype(np.float64) - recon.astype(np.float64)
    mse = float(np.mean(err ** 2))
    if mse == 0:
        return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))


class COINCompressor:
    """Reference COIN implementation (Dupont et al., 2021).

    Single SIREN per image, INT8 quantization, zlib compression.
    No residual coding, no meta-learning.
    """

    def __init__(self, hidden=64, n_layers=3, omega=30.0,
                 epochs=2000, lr=1e-3, quant_bits=8):
        self.hidden = hidden
        self.n_layers = n_layers
        self.omega = omega
        self.epochs = epochs
        self.lr = lr
        self.quant_bits = quant_bits

    def compress(self, image):
        """Compress RGB image using COIN method."""
        import torch
        import torch.nn as nn

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        H, W, C = image.shape
        # COIN trains separate SIREN for each channel
        # For fair comparison, we use grayscale conversion (Y)
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        gray_norm = (gray / 255.0).astype(np.float32)

        coords = np.stack(np.meshgrid(
            np.linspace(-1, 1, W),
            np.linspace(-1, 1, H)
        ), axis=-1).reshape(-1, 2)

        torch.manual_seed(0)
        # COIN uses larger network and more epochs
        class SirenCOIN(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                d = 2
                for k in range(n_layers_arg - 1):
                    lin = nn.Linear(d, hidden_arg)
                    bound = 1.0 / d if k == 0 else np.sqrt(6.0 / hidden_arg) / omega_arg
                    lin.weight.data.uniform_(-bound, bound)
                    lin.bias.data.uniform_(-bound, bound)
                    self.layers.append(lin)
                    d = hidden_arg
                self.head = nn.Linear(hidden_arg, 1)
                bound = np.sqrt(6.0 / hidden_arg) / omega_arg
                self.head.weight.data.uniform_(-bound, bound)
                self.head.bias.data.uniform_(-bound, bound)
                self.omega = omega_arg

            def forward(self, x):
                h = x
                for layer in self.layers:
                    h = torch.sin(self.omega * layer(h))
                return self.head(h)

        hidden_arg = self.hidden
        n_layers_arg = self.n_layers
        omega_arg = self.omega

        model = SirenCOIN()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        xt = torch.tensor(coords, dtype=torch.float32)
        yt = torch.tensor(gray_norm.flatten(), dtype=torch.float32)

        for ep in range(self.epochs):
            opt.zero_grad()
            pred = model(xt).squeeze(-1)
            loss = ((pred - yt) ** 2).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            recon = model(xt).squeeze(-1).numpy()
        recon_gray = (recon * 255).clip(0, 255).astype(np.uint8).reshape(H, W)
        recon_rgb = np.stack([recon_gray] * 3, axis=-1)
        psnr = compute_psnr(image, recon_rgb)

        # Quantize to INT8
        params = []
        for p in model.parameters():
            params.append(p.detach().numpy().flatten())
        params_flat = np.concatenate(params)

        levels = 2 ** self.quant_bits - 1
        max_val = np.abs(params_flat).max()
        scale = max_val / (levels / 2)
        quantized = np.round(params_flat / scale).astype(np.int8)
        quantized = np.clip(quantized, -levels // 2, levels // 2)

        # zlib compress
        raw_bytes = quantized.tobytes()
        compressed = zlib.compress(raw_bytes, 9)

        # Recipe: [H][W][scale][n_params][compressed_data]
        recipe = bytearray()
        recipe.extend(struct.pack('<H', H))
        recipe.extend(struct.pack('<H', W))
        recipe.extend(struct.pack('<f', float(scale)))
        recipe.extend(struct.pack('<I', len(params_flat)))
        recipe.extend(struct.pack('<I', len(compressed)))
        recipe.extend(compressed)

        return {
            'recipe_bytes': bytes(recipe),
            'original_size': H * W * C,
            'compression_ratio': (H * W * C) / len(recipe),
            'psnr_db': psnr,
            'n_params': len(params_flat),
        }


def run_comparison():
    """Run rigorous comparison: BLKH vs COIN vs JPEG vs WebP on real images."""
    print("=" * 72)
    print("RIGOROUS COMPARISON: BLKH vs COIN vs JPEG vs WebP")
    print("=" * 72)
    print()

    import torch
    from PIL import Image

    # Load real images
    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = repo_root / 'tests' / 'kodak_real' / 'skimage_dataset'
    print(f"Looking for images in: {dataset_dir}")
    print(f"Exists: {dataset_dir.exists()}")
    if dataset_dir.exists():
        print(f"Files: {list(dataset_dir.glob('*.png'))[:5]}")

    images = {}
    for img_path in sorted(dataset_dir.glob('*.png'))[:3]:  # Limit to 3 for speed
        img = np.array(Image.open(img_path).convert('RGB'))
        # Resize to common size for fair comparison
        if img.shape[0] != 256 or img.shape[1] != 256:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((256, 256), Image.LANCZOS)
            img = np.array(pil_img)
        images[img_path.stem] = img

    print(f"Loaded {len(images)} real images at 256×256")
    print()

    # Add path for BLKH modules
    sys.path.insert(0, str(repo_root / 'phase1_inr_compressor'))
    from siren_v5_dct import DCTCompressor
    from siren_v5_distill import DistillCompressor

    codecs = {
        'JPEG q=85': lambda img: _jpeg_compress(img, 85),
        'WebP q=80': lambda img: _webp_compress(img, 80),
        'COIN (h=32)': lambda img: COINCompressor(hidden=32, epochs=500).compress(img),
        'BLKH DCT q=0.9': lambda img: _blkh_dct(img, 0.9),
        'BLKH Distill': lambda img: DistillCompressor(
            teacher_hidden=32, student_hidden=8,
            teacher_epochs=200, student_epochs=400
        ).compress(img),
    }

    results = []
    for img_name, img in images.items():
        print(f"--- {img_name} ({img.shape}) ---")
        for codec_name, codec_fn in codecs.items():
            try:
                t0 = time.time()
                result = codec_fn(img)
                t = time.time() - t0
                if isinstance(result, dict):
                    size = len(result.get('recipe_bytes', b''))
                    psnr = result.get('psnr_db', 0)
                else:
                    size = result
                    psnr = None
                ratio = (img.shape[0] * img.shape[1] * 3) / max(size, 1)
                print(f"  {codec_name:<20}: {size:>6}B ({ratio:>6.1f}×) "
                      f"PSNR={psnr:>5.1f}dB" if psnr else
                      f"  {codec_name:<20}: {size:>6}B ({ratio:>6.1f}×)")
                results.append({
                    'image': img_name,
                    'codec': codec_name,
                    'size_bytes': size,
                    'compression_ratio': ratio,
                    'psnr_db': psnr,
                    'time_s': t,
                })
            except Exception as e:
                print(f"  {codec_name:<20}: ERROR — {type(e).__name__}: {str(e)[:60]}")
                results.append({
                    'image': img_name,
                    'codec': codec_name,
                    'error': str(e)[:100],
                })

    # Summary
    print()
    print("=" * 72)
    print("SUMMARY (averaged across all images)")
    print("=" * 72)
    print(f"{'Codec':<22} {'Avg Size':>10} {'Avg Ratio':>10} {'Avg PSNR':>10} {'Avg Time':>10}")
    codec_names = list(codecs.keys())
    for codec_name in codec_names:
        codec_results = [r for r in results if r['codec'] == codec_name and 'error' not in r]
        if not codec_results:
            continue
        avg_size = np.mean([r['size_bytes'] for r in codec_results])
        avg_ratio = np.mean([r['compression_ratio'] for r in codec_results])
        psnrs = [r['psnr_db'] for r in codec_results if r['psnr_db'] is not None]
        avg_psnr = np.mean(psnrs) if psnrs else 0
        avg_time = np.mean([r['time_s'] for r in codec_results])
        print(f"{codec_name:<22} {avg_size:>9.0f}B {avg_ratio:>9.1f}× "
              f"{avg_psnr:>9.1f}dB {avg_time:>9.2f}s")

    return results


def _jpeg_compress(img, quality):
    from PIL import Image as PILImage
    import io
    buf = io.BytesIO()
    PILImage.fromarray(img).save(buf, format='JPEG', quality=quality)
    return buf.tell()


def _webp_compress(img, quality):
    from PIL import Image as PILImage
    import io
    buf = io.BytesIO()
    PILImage.fromarray(img).save(buf, format='WebP', quality=quality)
    return buf.tell()


def _blkh_dct(img, quality):
    from siren_v5_dct import DCTCompressor
    comp = DCTCompressor()
    # Set quality if possible
    if hasattr(comp, 'quality'):
        comp.quality = quality
    result = comp.compress(img)
    return result


if __name__ == '__main__':
    results = run_comparison()
    import json
    print("\n--- JSON ---")
    print(json.dumps(results, indent=2, default=str))
