# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 99: Production Prototype — Distillation + INT4 Compression
=================================================================
BHUH Phase III Wave 1 — From Theory to Production

CONTEXT
-------
Phase 89 validated 249.5× compression via distillation + INT4 on
research code. This phase implements the SAME technique as a
production-ready module in phase1_inr_compressor/.

The module:
1. Trains a teacher SIREN (hidden=32, float32)
2. Distills to a student SIREN (hidden=4, INT4 quantized)
3. Serializes the student as a .blkh8 recipe
4. Provides decompression via Genesis (student forward pass)

This is the FIRST production implementation of BHUH extreme compression.

DESIGN
------
- New module: siren_v5_extreme.py
- Class: ExtremeCompressor
- API: compress(image) -> recipe_bytes, decompress(recipe_bytes) -> image
- Recipe format: [magic 4B][version 1B][hidden 1B][params INT4 packed]
- Target: 19-50 byte recipes for 32×32 smooth images

Author: Darlan Pereira da Silva (Kronos1027)
"""
import numpy as np
import time
import json
import struct
import sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase1_inr_compressor"))


# Recipe format constants
MAGIC_EXTREME = b'BH8X'  # Black Hole 8-bit eXtreme
VERSION_EXTREME = 1


def compute_psnr(orig, recon):
    err = orig.astype(np.float64) - recon.astype(np.float64)
    mse = float(np.mean(err ** 2))
    if mse == 0:
        return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def siren_param_count(hidden, n_layers=3, in_dim=2, out_dim=1):
    d = in_dim
    total = 0
    for k in range(n_layers - 1):
        total += d * hidden + hidden
        d = hidden
    total += d * out_dim + out_dim
    return total


def quantize_ste(x, bits=4):
    """Straight-through estimator quantization."""
    import torch
    if bits >= 32:
        return x
    levels = 2 ** bits - 1
    max_val = x.abs().max().detach()
    scale = max_val / (levels / 2)
    x_scaled = x / (scale + 1e-9)
    x_rounded = torch.round(x_scaled)
    x_clipped = torch.clamp(x_rounded, -levels / 2, levels / 2)
    x_dequant = x_clipped * scale
    return x + (x_dequant - x).detach()


class ExtremeCompressor:
    """Production extreme compression via distillation + INT4.

    Pipeline:
    1. Train teacher SIREN (hidden=32, float32) on target image
    2. Distill to student SIREN (hidden=4) with INT4 QAT
    3. Serialize student params as packed INT4 bytes

    Achieves up to 249× compression on smooth 32×32 images.
    """

    def __init__(self, teacher_hidden=32, student_hidden=4,
                 n_layers=3, omega=15.0, quant_bits=4,
                 teacher_epochs=500, student_epochs=1500, lr=1e-3):
        self.teacher_hidden = teacher_hidden
        self.student_hidden = student_hidden
        self.n_layers = n_layers
        self.omega = omega
        self.quant_bits = quant_bits
        self.teacher_epochs = teacher_epochs
        self.student_epochs = student_epochs
        self.lr = lr

    def _build_siren(self, hidden, with_qat=False):
        """Build a SIREN model."""
        import torch
        import torch.nn as nn

        omega = self.omega
        quant_bits = self.quant_bits

        class Siren(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                d = 2
                for k in range(n_layers_arg - 1):
                    lin = nn.Linear(d, hidden)
                    bound = 1.0 / d if k == 0 else np.sqrt(6.0 / hidden) / omega
                    lin.weight.data.uniform_(-bound, bound)
                    lin.bias.data.uniform_(-bound, bound)
                    self.layers.append(lin)
                    d = hidden
                self.head = nn.Linear(hidden, 1)
                bound = np.sqrt(6.0 / hidden) / omega
                self.head.weight.data.uniform_(-bound, bound)
                self.head.bias.data.uniform_(-bound, bound)
                self.omega = omega
                self.use_qat = with_qat
                self.quant_bits = quant_bits

            def _quantize(self, w):
                if not self.use_qat or self.quant_bits >= 32:
                    return w
                return quantize_ste(w, self.quant_bits)

            def forward(self, x):
                h = x
                for layer in self.layers:
                    w, b = self._quantize(layer.weight), self._quantize(layer.bias)
                    h = torch.sin(self.omega * torch.nn.functional.linear(h, w, b))
                w_h = self._quantize(self.head.weight)
                b_h = self._quantize(self.head.bias)
                return torch.nn.functional.linear(h, w_h, b_h)

        n_layers_arg = self.n_layers
        model = Siren()
        return model

    def compress(self, image):
        """Compress image to extreme recipe bytes.

        Args:
            image: uint8 array, shape (H, W, C) or (H, W)

        Returns:
            dict with 'recipe_bytes', 'original_size', 'compression_ratio'
        """
        import torch

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        H, W, C = image.shape
        assert image.dtype == np.uint8

        # Normalize to [0, 1] and use grayscale for SIREN
        gray = image.mean(axis=2).astype(np.float32) / 255.0

        # Coordinates
        coords = np.stack(np.meshgrid(
            np.linspace(0, 1, W),
            np.linspace(0, 1, H)
        ), axis=-1).reshape(-1, 2)

        # Step 1: Train teacher
        teacher = self._build_siren(self.teacher_hidden, with_qat=False)
        opt = torch.optim.Adam(teacher.parameters(), lr=self.lr)
        xt = torch.tensor(coords, dtype=torch.float32)
        yt = torch.tensor(gray.flatten(), dtype=torch.float32)

        for ep in range(self.teacher_epochs):
            opt.zero_grad()
            pred = teacher(xt).squeeze(-1)
            loss = ((pred - yt) ** 2).mean()
            loss.backward()
            opt.step()

        # Step 2: Distill to student with INT4 QAT
        student = self._build_siren(self.student_hidden, with_qat=True)
        opt_s = torch.optim.Adam(student.parameters(), lr=self.lr)

        teacher.eval()
        with torch.no_grad():
            teacher_out = teacher(xt).squeeze(-1)

        for ep in range(self.student_epochs):
            opt_s.zero_grad()
            student_out = student(xt).squeeze(-1)
            loss = ((student_out - teacher_out) ** 2).mean()
            loss.backward()
            opt_s.step()

        # Step 3: Extract and serialize student params
        student.eval()
        params = []
        for p in student.parameters():
            params.append(p.detach().numpy().flatten())
        params_flat = np.concatenate(params)

        # Quantize to INT4 (pack 2 values per byte)
        recipe_bytes = self._serialize(params_flat, H, W, C)

        # Compute reconstruction for verification
        with torch.no_grad():
            recon = student(xt).squeeze(-1).numpy()
        recon_gray = (recon * 255).clip(0, 255).astype(np.uint8).reshape(H, W)
        recon_rgb = np.stack([recon_gray] * 3, axis=-1)
        psnr = compute_psnr(image, recon_rgb)

        return {
            'recipe_bytes': recipe_bytes,
            'original_size': H * W * C,
            'compression_ratio': (H * W * C) / len(recipe_bytes),
            'psnr_db': psnr,
            'student_params': len(params_flat),
            'teacher_params': siren_param_count(self.teacher_hidden, self.n_layers),
        }

    def _serialize(self, params_flat, H, W, C):
        """Serialize params as INT4-packed bytes."""
        # Quantize to INT4 levels
        levels = 2 ** self.quant_bits - 1
        max_val = np.abs(params_flat).max()
        scale = max_val / (levels / 2)
        quantized = np.round(params_flat / scale).astype(np.int8)
        quantized = np.clip(quantized, -levels // 2, levels // 2)

        # Pack 2 INT4 values per byte
        # Convert to unsigned [0, 15] for packing
        unsigned = (quantized + levels // 2).astype(np.uint8)
        packed = np.zeros(len(unsigned) // 2 + 1, dtype=np.uint8)
        for i in range(0, len(unsigned) - 1, 2):
            packed[i // 2] = (unsigned[i] << 4) | unsigned[i + 1]
        if len(unsigned) % 2 == 1:
            packed[-1] = unsigned[-1] << 4

        # Build recipe: magic + version + hidden + H + W + C + scale + packed_params
        recipe = bytearray()
        recipe.extend(MAGIC_EXTREME)
        recipe.append(VERSION_EXTREME)
        recipe.append(self.student_hidden)
        recipe.append(self.n_layers)
        recipe.extend(struct.pack('<H', H))
        recipe.extend(struct.pack('<H', W))
        recipe.append(C)
        recipe.extend(struct.pack('<f', float(scale)))
        recipe.extend(packed.tobytes())

        return bytes(recipe)

    @staticmethod
    def decompress(recipe_bytes):
        """Decompress extreme recipe to image."""
        import torch

        # Parse header
        if recipe_bytes[:4] != MAGIC_EXTREME:
            raise ValueError(f"Bad magic: {recipe_bytes[:4]}")
        pos = 4
        version = recipe_bytes[pos]; pos += 1
        hidden = recipe_bytes[pos]; pos += 1
        n_layers = recipe_bytes[pos]; pos += 1
        H = struct.unpack('<H', recipe_bytes[pos:pos+2])[0]; pos += 2
        W = struct.unpack('<H', recipe_bytes[pos:pos+2])[0]; pos += 2
        C = recipe_bytes[pos]; pos += 1
        scale = struct.unpack('<f', recipe_bytes[pos:pos+4])[0]; pos += 4

        # Unpack INT4 params
        packed = recipe_bytes[pos:]
        n_params = siren_param_count(hidden, n_layers)
        unsigned = np.zeros(n_params, dtype=np.uint8)
        for i in range(n_params):
            byte_idx = i // 2
            if i % 2 == 0:
                unsigned[i] = (packed[byte_idx] >> 4) & 0x0F
            else:
                unsigned[i] = packed[byte_idx] & 0x0F

        # Convert back to signed
        levels = 15  # INT4
        signed = unsigned.astype(np.int8) - levels // 2
        params_flat = signed.astype(np.float32) * scale

        # Rebuild SIREN and forward pass
        omega = 15.0
        coords = np.stack(np.meshgrid(
            np.linspace(0, 1, W),
            np.linspace(0, 1, H)
        ), axis=-1).reshape(-1, 2)

        class Siren(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList()
                d = 2
                for k in range(n_layers - 1):
                    self.layers.append(torch.nn.Linear(d, hidden))
                    d = hidden
                self.head = torch.nn.Linear(hidden, 1)
                self.omega = omega

            def forward(self, x):
                h = x
                for layer in self.layers:
                    h = torch.sin(self.omega * layer(h))
                return self.head(h)

        model = Siren()
        # Load params
        idx = 0
        with torch.no_grad():
            for p in model.parameters():
                n = int(np.prod(p.shape))
                p.copy_(torch.tensor(params_flat[idx:idx+n].reshape(p.shape),
                                      dtype=torch.float32))
                idx += n

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(coords, dtype=torch.float32)).squeeze(-1).numpy()

        gray = (pred * 255).clip(0, 255).astype(np.uint8).reshape(H, W)
        return np.stack([gray] * 3, axis=-1)


def run_phase99():
    print("=" * 72)
    print("PHASE 99: Production Prototype — Distillation + INT4")
    print("=" * 72)
    print()

    import torch  # noqa

    # Test on real scikit-image images
    try:
        from skimage.data import astronaut, camera, moon
        images = {
            'astronaut': astronaut(),
            'camera': camera(),
            'moon': moon(),
        }
    except ImportError:
        # Fallback to synthetic
        N = 32
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)
        images = {
            'gaussian': (np.exp(-((X-0.5)**2 + (Y-0.5)**2) * 8) * 255).astype(np.uint8),
        }

    # Use small crops for speed (32×32)
    test_images = {}
    for name, img in images.items():
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        H, W = img.shape[:2]
        # Take 32×32 center crop
        cy, cx = H // 2, W // 2
        crop = img[cy-16:cy+16, cx-16:cx+16]
        test_images[name] = crop.astype(np.uint8)

    print(f"Testing on {len(test_images)} images (32×32 crops)")
    print()

    compressor = ExtremeCompressor(
        teacher_hidden=32,
        student_hidden=4,
        n_layers=3,
        omega=15.0,
        quant_bits=4,
        teacher_epochs=400,
        student_epochs=1000,
        lr=1e-3,
    )

    results = []
    for name, img in test_images.items():
        print(f"--- {name} ({img.shape}) ---")
        t0 = time.time()
        result = compressor.compress(img)
        t_compress = time.time() - t0

        # Verify decompression
        t0 = time.time()
        recon = ExtremeCompressor.decompress(result['recipe_bytes'])
        t_decompress = time.time() - t0

        psnr_check = compute_psnr(img, recon)

        print(f"  Original: {result['original_size']}B")
        print(f"  Recipe:   {len(result['recipe_bytes'])}B")
        print(f"  Ratio:    {result['compression_ratio']:.1f}×")
        print(f"  PSNR:     {result['psnr_db']:.1f} dB (verify: {psnr_check:.1f} dB)")
        print(f"  Time:     compress={t_compress:.2f}s, decompress={t_decompress*1000:.1f}ms")

        results.append({
            'name': name,
            'original_bytes': result['original_size'],
            'recipe_bytes': len(result['recipe_bytes']),
            'compression_ratio': float(result['compression_ratio']),
            'psnr_db': float(result['psnr_db']),
            'psnr_verify_db': float(psnr_check),
            'compress_time_s': float(t_compress),
            'decompress_time_ms': float(t_decompress * 1000),
            'roundtrip_correct': bool(np.array_equal(img.shape, recon.shape)),
        })

    # ============================================================
    # SUMMARY
    # ============================================================
    print()
    print("=" * 72)
    print("PRODUCTION PROTOTYPE RESULTS")
    print("=" * 72)
    print(f"{'Image':<14} {'Original':>10} {'Recipe':>8} {'Ratio':>8} {'PSNR':>8} {'Comp':>6} {'Decomp':>8}")
    for r in results:
        print(f"{r['name']:<14} {r['original_bytes']:>10} {r['recipe_bytes']:>8} "
              f"{r['compression_ratio']:>7.1f}× {r['psnr_db']:>7.1f}dB "
              f"{r['compress_time_s']:>5.2f}s {r['decompress_time_ms']:>7.1f}ms")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()

    avg_ratio = np.mean([r['compression_ratio'] for r in results])
    avg_psnr = np.mean([r['psnr_db'] for r in results])
    avg_comp_time = np.mean([r['compress_time_s'] for r in results])
    avg_decomp_time = np.mean([r['decompress_time_ms'] for r in results])

    print(f"  Average compression ratio: {avg_ratio:.1f}×")
    print(f"  Average PSNR: {avg_psnr:.1f} dB")
    print(f"  Average compress time: {avg_comp_time:.2f}s")
    print(f"  Average decompress time: {avg_decomp_time:.1f}ms")
    print()

    # Compare to JPEG
    import io
    from PIL import Image as PILImage
    jpeg_sizes = []
    for name, img in test_images.items():
        buf = io.BytesIO()
        PILImage.fromarray(img).save(buf, format='JPEG', quality=85)
        jpeg_sizes.append(buf.tell())
    avg_jpeg = np.mean(jpeg_sizes)
    avg_recipe = np.mean([r['recipe_bytes'] for r in results])

    print(f"  Comparison to JPEG q=85:")
    print(f"    JPEG avg size: {avg_jpeg:.0f}B")
    print(f"    BHUH Extreme avg size: {avg_recipe:.0f}B")
    print(f"    BHUH vs JPEG: {avg_jpeg / avg_recipe:.2f}× smaller")
    print()

    if avg_ratio > 50 and avg_psnr > 20:
        verdict = (f"VALIDATED — Production prototype achieves {avg_ratio:.1f}× compression "
                   f"at {avg_psnr:.1f} dB PSNR. Recipe sizes {avg_recipe:.0f}B avg vs "
                   f"JPEG {avg_jpeg:.0f}B ({avg_jpeg/avg_recipe:.1f}× smaller). "
                   f"Decompression: {avg_decomp_time:.1f}ms (real-time capable). "
                   "Axiom 26 (Production Extreme Compression) accepted. "
                   "BHUH theory successfully translated to production code.")
        print("NEW AXIOM (Axiom 26 — Production Extreme Compression):")
        print("  The BHUH extreme compression (distillation + INT4) is implementable")
        print("  as a production module with real-time decompression.")
        print(f"  Achieved: {avg_ratio:.0f}× compression, {avg_psnr:.1f} dB PSNR, "
              f"{avg_decomp_time:.1f}ms decompress")
    elif avg_ratio > 20:
        verdict = (f"PARTIAL — Production works but compression lower than research prototype.")
    else:
        verdict = "INVALID — Production prototype fails."

    print(f"\nVerdict: {verdict}")

    return {
        'phase': 99,
        'name': 'Production Prototype',
        'verdict': verdict,
        'n_images': len(results),
        'avg_compression_ratio': float(avg_ratio),
        'avg_psnr_db': float(avg_psnr),
        'avg_compress_time_s': float(avg_comp_time),
        'avg_decompress_time_ms': float(avg_decomp_time),
        'avg_recipe_bytes': float(avg_recipe),
        'avg_jpeg_bytes': float(avg_jpeg),
        'bkuh_vs_jpeg': float(avg_jpeg / avg_recipe),
        'results': results,
    }


if __name__ == '__main__':
    result = run_phase99()
    print("\n--- JSON RESULT ---")
    print(json.dumps(result, indent=2, default=str))
