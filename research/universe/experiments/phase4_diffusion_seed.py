# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 4 Experiment: Diffusion Seed (Latent Space Compression)
===============================================================
Tests Principle 1 (Singularity) with latent space approach.

CONCEPT:
  Instead of training a SIREN per image (Phase 1), we train a VARIATIONAL
  AUTOENCODER (VAE) on multiple images. Each image is encoded to a tiny
  latent vector (the "seed"). Decompression = decode the latent.

  This simulates what a diffusion model would do:
  - Diffusion: noise -> reverse process -> image
  - VAE: latent -> decoder -> image

  The latent vector IS the "seed" — the singular representation.

HYPOTHESIS:
  A VAE trained on 100 satellite images will compress each to ~64-128 bytes
  (latent vector), achieving 1000x+ compression on the image corpus.

METHOD:
  1. Train a small VAE on 100 satellite images
  2. Encode each image to 16-dim latent
  3. Measure: VAE size + 100 latents vs 100 separate images
  4. Compare with Phase 1 Multi-File SIREN

LIMITATION:
  VAE is lossy (like diffusion). Bit-perfect would need residual.

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys
import os
import time
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
from phase1_multi_file_siren import generate_satellite_images


# ============================================================
# VAE Architecture
# ============================================================

class VAEEncoder(nn.Module):
    """Encode image to latent space."""
    def __init__(self, in_channels=3, latent_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(64 * 16 * 16, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decode latent to image."""
    def __init__(self, latent_dim=16, out_channels=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, 3, stride=2, padding=1, output_padding=1),  # 64 -> 128
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 64, 16, 16)
        return self.deconv(h)


class VAE(nn.Module):
    """Variational Autoencoder for image compression."""
    def __init__(self, in_channels=3, latent_dim=16):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, in_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu  # Use mean (deterministic)

    def decode(self, z):
        return self.decoder(z)


def vae_loss(recon_x, x, mu, logvar, beta=0.001):
    """VAE loss = reconstruction + KL divergence."""
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD


# ============================================================
# Training
# ============================================================

def train_vae(images, epochs=200, lr=3e-3, latent_dim=16, device='cpu', verbose=False):
    """Train VAE on multiple images."""
    # Prepare data: (N, C, H, W) normalized to [0, 1]
    data = np.stack([img.astype(np.float32) / 255.0 for img in images])
    data = torch.from_numpy(data).permute(0, 3, 1, 2).to(device)  # (N, 3, H, W)

    model = VAE(in_channels=3, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = vae_loss(recon, data, mu, logvar)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 20 == 0:
            # Compute PSNR
            with torch.no_grad():
                mse = F.mse_loss(recon, data).item()
                psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
            print(f"  VAE Epoch {epoch}: loss={loss.item():.1f}, PSNR={psnr:.1f}dB")

    return model


# ============================================================
# Compression
# ============================================================

def compress_with_vae(model, images, device='cpu'):
    """Compress images using VAE latent space.

    Returns: (vae_size, latents_size, total_size, avg_psnr)
    """
    # 1. Compress VAE weights (INT8 + zlib)
    weights_buf = bytearray()
    for param in model.parameters():
        w = param.detach().cpu().numpy()
        # Quantize to int16 (VAE weights have larger range)
        max_abs = max(np.abs(w).max(), 1e-8)
        scale = max_abs / 32767.0
        q = np.round(w / scale).astype(np.int16)
        weights_buf.extend(q.tobytes())

    vae_size = len(zlib.compress(bytes(weights_buf), 9))

    # 2. Encode each image to latent
    latents = []
    psnrs = []
    with torch.no_grad():
        for img in images:
            x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            z = model.encode(x)  # (1, latent_dim)
            recon = model.decode(z)

            # Compute PSNR
            recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
            mse = np.mean((img / 255.0 - recon_np) ** 2)
            psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
            psnrs.append(psnr)

            # Quantize latent to int16
            z_np = z.squeeze(0).cpu().numpy()
            max_abs = max(np.abs(z_np).max(), 1e-8)
            z_scale = max_abs / 32767.0
            z_q = np.round(z_np / z_scale).astype(np.int16)
            latents.append((z_q.tobytes(), z_scale))

    # 3. Compress all latents together
    latents_buf = bytearray()
    for z_bytes, _ in latents:
        latents_buf.extend(z_bytes)
    latents_size = len(zlib.compress(bytes(latents_buf), 9))

    total = vae_size + latents_size
    avg_psnr = np.mean(psnrs)

    return vae_size, latents_size, total, avg_psnr


# ============================================================
# Main Experiment
# ============================================================

def run_phase4_experiment(verbose=True):
    """Run Phase 4 Diffusion Seed experiment."""
    print("=" * 80)
    print("🧪 Phase 4: Diffusion Seed (VAE Latent Compression)")
    print("=" * 80)

    device = 'cpu'

    # Generate images
    print("\n📸 Generating satellite images...")
    images = generate_satellite_images(n_images=50, size=128, seed=42)
    total_raw = sum(img.nbytes for img in images)
    total_zip = sum(len(zlib.compress(img.tobytes(), 9)) for img in images)
    print(f"  50 images @ 128x128, Raw: {total_raw:,}B, ZIP: {total_zip:,}B")

    # Test different latent dimensions
    print("\n🔧 Training VAEs with different latent dimensions...")
    results = []

    for latent_dim in [8, 16, 32, 64]:
        print(f"\n  Latent dim = {latent_dim}...")
        t0 = time.time()
        model = train_vae(images, epochs=100, lr=3e-3, latent_dim=latent_dim,
                          device=device, verbose=verbose)
        train_time = time.time() - t0

        vae_size, latents_size, total, avg_psnr = compress_with_vae(model, images, device)

        vs_zip = total_zip / max(total, 1)
        per_image = total / len(images)

        print(f"  VAE: {vae_size:,}B, Latents: {latents_size:,}B, Total: {total:,}B")
        print(f"  PSNR: {avg_psnr:.1f}dB, vs ZIP: {vs_zip:.2f}x, Per image: {per_image:.0f}B")

        results.append({
            'latent_dim': latent_dim,
            'vae_size': vae_size,
            'latents_size': latents_size,
            'total': total,
            'psnr': avg_psnr,
            'vs_zip': vs_zip,
            'per_image': per_image,
            'train_time': train_time,
        })

    # Summary
    print("\n" + "=" * 80)
    print("📊 PHASE 4 SUMMARY")
    print("=" * 80)
    print(f"\n{'Latent':>7} {'VAE':>8} {'Latents':>8} {'Total':>8} {'PSNR':>8} {'vs ZIP':>8} {'Per img':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['latent_dim']:>7} {r['vae_size']:>7,}B {r['latents_size']:>7,}B "
              f"{r['total']:>7,}B {r['psnr']:>6.1f}dB {r['vs_zip']:>7.2f}x {r['per_image']:>7.0f}B")

    # Compare with Phase 1
    print(f"\n📋 Phase 1 Multi-SIREN reference: ~24,000B (17.93x vs SIREN, 62.78x vs ZIP)")
    print(f"📋 Phase 4 VAE best: {min(r['total'] for r in results):,}B ({total_zip/min(r['total'] for r in results):.2f}x vs ZIP)")

    # Find best
    best = min(results, key=lambda x: x['total'])
    print(f"\n✅ Best: latent_dim={best['latent_dim']}, {best['total']:,}B, "
          f"PSNR={best['psnr']:.1f}dB, {best['vs_zip']:.2f}x vs ZIP")

    return results


if __name__ == '__main__':
    results = run_phase4_experiment(verbose=True)
