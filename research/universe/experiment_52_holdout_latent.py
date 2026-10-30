"""
Experiment 52: can a trained shared SIREN encode a NEW image by optimizing
only its small latent vector? This is encoder-side adaptation, not zero-shot
reconstruction: original pixels of the new image ARE used to optimize z.

Design:
  - First N-1 images train shared backbone + their own latent vectors.
  - Freeze backbone.
  - Adapt ONLY the held-out image's latent vector using held-out pixels.
  - Quantize to float16, serialize whole corpus, decode and evaluate.
  - Report actual marginal encoded cost of the additional image.

Try:
  python research/universe/experiment_52_holdout_latent.py --size 16 --epochs 10 --adapt-epochs 10 --seeds 42
  python research/universe/experiment_52_holdout_latent.py --size 64 --epochs 600 --adapt-epochs 600 --seeds 42,123
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from experiment_51_fair_sharing import (
    ConditionalSiren,
    collect_smooth_images,
    image_grid,
    metrics,
    pack_recipe,
    reconstruct,
)


def experiment_one(
    images: np.ndarray,
    seed: int,
    epochs: int,
    adapt_epochs: int,
    width: int,
    depth: int,
    omega: float,
    latent_dim: int,
    lr: float,
    heldout: int,
    output: Path,
) -> dict:
    count, size, width_px = images.shape
    if width_px != size or not 0 <= heldout < count or count < 3:
        raise ValueError("Need >=3 square images and a valid held-out image ID")
    coords = image_grid(size)
    train_ids = [i for i in range(count) if i != heldout]
    torch.manual_seed(seed + 5000)
    model = ConditionalSiren(count, width, depth, omega, latent_dim)
    target = torch.from_numpy(images[train_ids].reshape(count - 1, -1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.perf_counter()

    for _ in range(epochs):
        opt.zero_grad()
        pos = coords.unsqueeze(0).expand(len(train_ids), -1, -1)
        z = model.latents[train_ids].unsqueeze(1).expand(-1, len(coords), -1)
        prediction = model.net(torch.cat((pos, z), dim=-1))
        loss = (prediction - target).square().mean()
        loss.backward()
        opt.step()
    backbone_train_seconds = time.perf_counter() - start

    # Frozen backbone prevents new image from changing the representation
    # learned on the first N-1 images. Only z_new can change.
    for param in model.net.parameters():
        param.requires_grad_(False)
    z_new = nn.Parameter(model.latents[heldout].detach().clone())
    adapt_opt = torch.optim.Adam([z_new], lr=lr)
    heldout_target = torch.from_numpy(images[heldout].ravel())
    start = time.perf_counter()
    for _ in range(adapt_epochs):
        adapt_opt.zero_grad()
        x_with_z = torch.cat(
            (coords, z_new.unsqueeze(0).expand(len(coords), -1)), dim=-1
        )
        pred = model.net(x_with_z)
        loss = (pred - heldout_target).square().mean()
        loss.backward()
        adapt_opt.step()
    adaptation_seconds = time.perf_counter() - start

    with torch.no_grad():
        model.latents[heldout].copy_(z_new)

    groups = [list(range(count))]
    blob_all = pack_recipe(
        [model], "conditional", groups, count, size,
        width, depth, omega, latent_dim,
    )
    decoded = reconstruct(blob_all)
    quality = metrics(images, decoded, blob_all)
    if not quality["verified_post_decode"]:
        raise AssertionError("Decoded-image evaluation is required")

    # A *separately serialized* base model with only the training images.
    # The size difference includes new latent, mapping and header changes,
    # including effects of zlib (not just 2*latent_dim).
    base_model = ConditionalSiren(count - 1, width, depth, omega, latent_dim)
    base_model.net.load_state_dict(model.net.state_dict())
    with torch.no_grad():
        base_model.latents.copy_(model.latents[train_ids])
    base_blob = pack_recipe(
        [base_model], "conditional", [list(range(count - 1))],
        count - 1, size, width, depth, omega, latent_dim,
    )
    # Check that the base file can be decoded too.
    decoded_base = reconstruct(base_blob)
    if decoded_base.shape != images[train_ids].shape:
        raise AssertionError("Base model decode shape mismatch")

    full_filename = f"exp52_shared_n{count}_h{width}_seed{seed}.blkh51"
    base_filename = f"exp52_shared_n{count-1}_h{width}_seed{seed}.blkh51"
    (output / full_filename).write_bytes(blob_all)
    (output / base_filename).write_bytes(base_blob)

    return {
        "seed": seed, "size": size, "count": count,
        "trained_image_ids": train_ids, "heldout_image_id": heldout,
        "backbone_frozen_during_adaptation": True,
        "encoder_uses_heldout_pixels_to_optimize_latent": True,
        "epochs_backbone": epochs, "epochs_latent_only": adapt_epochs,
        "width": width, "latent_dim": latent_dim,
        "total_bytes_with_new_image": len(blob_all),
        "total_bytes_without_new_image": len(base_blob),
        "marginal_bytes_for_new_image": len(blob_all) - len(base_blob),
        "train_image_psnr_db": [quality["image_psnr_db"][i] for i in train_ids],
        "mean_train_image_psnr_db": float(np.mean([
            quality["image_psnr_db"][i] for i in train_ids
        ])),
        "heldout_psnr_db": quality["image_psnr_db"][heldout],
        "backbone_train_seconds": round(backbone_train_seconds, 3),
        "new_image_adaptation_seconds": round(adaptation_seconds, 3),
        "full_file": full_filename, "base_file": base_filename,
        "sha256": hashlib.sha256(blob_all).hexdigest(),
        "verified_post_decode": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--adapt-epochs", type=int, default=600)
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--omega", type=float, default=30.0)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--heldout", type=int, default=-1)
    parser.add_argument("--out", default="research/universe/_exp52_out")
    args = parser.parse_args()
    if args.count < 3 or args.size < 4 or args.epochs < 1 or args.adapt_epochs < 1:
        parser.error("Require >=3 images, size>=4 and epochs>=1")
    heldout = args.heldout if args.heldout >= 0 else args.count - 1
    images, names = collect_smooth_images(args.size, args.count)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    result = []
    for seed in [int(x) for x in args.seeds.split(",")]:
        record = experiment_one(
            images, seed, args.epochs, args.adapt_epochs, args.width,
            args.depth, args.omega, args.latent_dim, args.lr, heldout, output
        )
        result.append(record)
        print(
            f"seed={seed} new-image PSNR={record['heldout_psnr_db']:.2f} dB "
            f"shared file={record['total_bytes_with_new_image']} B "
            f"marginal={record['marginal_bytes_for_new_image']} B",
            flush=True,
        )
    payload = {
        "experiment": 52,
        "purpose": "latent-only encoding of a held-out correlated analytic surface",
        "data_type": "correlated synthetic, not general photographs",
        "image_names": names,
        "input_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
        "limitations": [
            "Held-out encoding uses original pixel data to optimize z.",
            "This is in-domain adaptation, not zero-shot and not independent-domain generalization.",
            "The fixed shared backbone is trained on five closely related textures.",
            "Compare total file costs and encoder compute before practical claims.",
        ],
        "results": result,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
