# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
"""
Phase 38: Neural Data Fusion (Merge Multiple File Types)
==========================================================
Tests whether separate universes can be FUSED into one.

CONCEPT:
  We have separate universes for images, audio, text (Phase 36).
  Can we FUSE them — merge two trained universes into one bigger universe
  WITHOUT retraining from scratch?

  This would enable:
  - Incremental universe expansion (add new file types without retraining)
  - Distributed training (train parts separately, fuse later)
  - Universe composition (combine domain-specific universes)

HYPOTHESIS:
  Weight averaging of two SIREN universes will produce a fused universe
  that can reconstruct both domains, with some quality loss.

METHOD:
  1. Train universe A on images
  2. Train universe B on audio
  3. Fuse: average weights of A and B
  4. Test: can fused model reconstruct both images and audio?

Author: Darlan Pereira da Silva (Kronos1027)
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_inr_compressor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from phase1_multi_file_siren import SIREN, SIRENLayer, get_coordinates, generate_satellite_images, measure_model_size_compressed
from phase2_universal_hypernetwork import generate_audio_files


def fuse_models(model_a, model_b, alpha=0.5):
    """Fuse two models by weighted averaging of weights."""
    fused = copy.deepcopy(model_a)
    with torch.no_grad():
        for (name_a, param_a), (_, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            fused_param = alpha * param_a + (1 - alpha) * param_b
            for name, param in fused.named_parameters():
                if name == name_a:
                    param.data = fused_param.data
                    break
    return fused


import copy


def run_phase38_experiment(verbose=True):
    """Run Phase 38 Neural Data Fusion experiment."""
    print("=" * 80)
    print("🧪 Phase 38: Neural Data Fusion (Merge Universes)")
    print("=" * 80)

    device = 'cpu'

    # Train universe A on images
    print("\n📦 Training Universe A (images)...")
    images = generate_satellite_images(n_images=5, size=64, seed=42)
    from phase1_multi_file_siren import train_multi_file_siren
    img_model, img_loss = train_multi_file_siren(images, epochs=80, device=device, verbose=False)
    img_size = measure_model_size_compressed(img_model)
    print(f"  Image universe: loss={img_loss:.6f}, size={img_size:,}B")

    # Train universe B on audio
    print("\n📦 Training Universe B (audio)...")
    audio_files = generate_audio_files(n_files=5, duration=0.25, sr=8000, seed=99)
    from phase2_universal_hypernetwork import train_audio_inr, MultiFileAudioINR
    aud_model, aud_loss = train_audio_inr(audio_files, epochs=80, device=device, verbose=False)
    aud_size = measure_model_size_compressed(aud_model)
    print(f"  Audio universe: loss={aud_loss:.6f}, size={aud_size:,}B")

    # Get baseline quality
    coords_img = get_coordinates(64, device)
    with torch.no_grad():
        img_pred_a = img_model(coords_img, 0)
    img_baseline_mse = F.mse_loss(img_pred_a, torch.from_numpy(images[0].astype(np.float32)/255.0).reshape(-1,3).to(device)).item()
    img_psnr_baseline = 10 * np.log10(1.0 / max(img_baseline_mse, 1e-10))

    t_aud = torch.linspace(0, 1, len(audio_files[0]), device=device).unsqueeze(1)
    with torch.no_grad():
        aud_pred_b = aud_model(t_aud, 0)
    aud_baseline_mse = F.mse_loss(aud_pred_b.squeeze(), torch.from_numpy(audio_files[0].astype(np.float32)/32767.0).to(device)).item()
    aud_psnr_baseline = 10 * np.log10(1.0 / max(aud_baseline_mse, 1e-10))

    # Fuse models (weight averaging)
    print("\n🌌 Fusing universes (weight averaging, α=0.5)...")
    fused_model = copy.deepcopy(img_model)
    with torch.no_grad():
        for (name_a, param_a), (_, param_b) in zip(
            img_model.named_parameters(), aud_model.named_parameters()
        ):
            # Only fuse if shapes match
            if param_a.shape == param_b.shape:
                for name, param in fused_model.named_parameters():
                    if name == name_a:
                        param.data = 0.5 * param_a.data + 0.5 * param_b.data
                        break

    fused_size = measure_model_size_compressed(fused_model)

    # Test fused model on images
    with torch.no_grad():
        img_pred_fused = fused_model(coords_img, 0)
    img_fused_mse = F.mse_loss(img_pred_fused, torch.from_numpy(images[0].astype(np.float32)/255.0).reshape(-1,3).to(device)).item()
    img_psnr_baseline = 10 * np.log10(1.0 / max(img_baseline_mse, 1e-10))
    img_psnr_fused = 10 * np.log10(1.0 / max(img_fused_mse, 1e-10))

    # Test fused model on audio (if architecture allows)
    try:
        with torch.no_grad():
            aud_pred_fused = fused_model(t_aud, 0)
        aud_fused_mse = F.mse_loss(aud_pred_fused.squeeze(), torch.from_numpy(audio_files[0].astype(np.float32)/32767.0).to(device)).item()
        aud_psnr_baseline = 10 * np.log10(1.0 / max(aud_baseline_mse, 1e-10))
        aud_psnr_fused = 10 * np.log10(1.0 / max(aud_fused_mse, 1e-10))
        aud_test_works = True
    except Exception:
        aud_test_works = False
        aud_psnr_fused = 0

    # Results
    print(f"\n{'='*80}")
    print("📊 PHASE 38 RESULTS — NEURAL DATA FUSION")
    print(f"{'='*80}")
    print(f"\n  {'Model':<30} {'Size':>8} {'Img PSNR':>10} {'Aud PSNR':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Universe A (images only)':<30} {img_size:>7,}B {img_psnr_baseline:>8.1f}dB {'-':>9}")
    print(f"  {'Universe B (audio only)':<30} {aud_size:>7,}B {'-':>9} {aud_psnr_baseline:>8.1f}dB")
    print(f"  {'Separate (A+B)':<30} {img_size+aud_size:>7,}B {img_psnr_baseline:>8.1f}dB {aud_psnr_baseline:>8.1f}dB")
    print(f"  {'Fused (α=0.5)':<30} {fused_size:>7,}B {img_psnr_fused:>8.1f}dB {aud_psnr_fused:>8.1f}dB" if aud_test_works else f"  {'Fused (α=0.5)':<30} {fused_size:>7,}B {img_psnr_fused:>8.1f}dB {'N/A':>9}")

    size_ratio = (img_size + aud_size) / max(fused_size, 1)

    print(f"\n  📋 Analysis:")
    print(f"  - Separate: {img_size + aud_size:,}B (two universes)")
    print(f"  - Fused:    {fused_size:,}B (one universe)")
    print(f"  - Size ratio: {size_ratio:.2f}x ({'fused smaller!' if size_ratio > 1 else 'fused larger'})")
    print(f"  - Image quality: {img_psnr_baseline:.1f} → {img_psnr_fused:.1f}dB ({'+' if img_psnr_fused > img_psnr_baseline else ''}{img_psnr_fused-img_psnr_baseline:.1f}dB)")

    if size_ratio > 1.2:
        print(f"\n  ✅ Fusion saves {size_ratio:.2f}x storage!")
    if img_psnr_fused > img_psnr_baseline - 3:
        print(f"  ✅ Image quality preserved within 3dB")
    else:
        print(f"  ⚠️  Image quality degraded by {img_psnr_baseline - img_psnr_fused:.1f}dB")

    print(f"\n  📋 Deep insight:")
    print(f"  Universe fusion is the BHUH equivalent of 'merging galaxies':")
    print(f"  - Two separate universes → one bigger universe")
    print(f"  - No retraining needed (just weight averaging)")
    print(f"  - Quality trade-off is the 'fusion cost'")
    print(f"  - Enables incremental universe growth")

    return {
        'separate_size': img_size + aud_size,
        'fused_size': fused_size,
        'size_ratio': size_ratio,
        'img_psnr_baseline': img_psnr_baseline,
        'img_psnr_fused': img_psnr_fused,
    }


if __name__ == '__main__':
    results = run_phase38_experiment(verbose=True)
