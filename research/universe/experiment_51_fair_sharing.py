"""
Experiment 51 — auditable post-decode comparison of individual vs conditional SIREN.

This experiment addresses two pitfalls in earlier BHUH comparisons:
1) K >= N creates effectively singleton clusters, so there is no weight sharing.
2) Comparing pre-quantization PSNR with quantized/entropy-coded byte counts
   is not a valid rate-distortion measurement.

Both arms use exactly the same float16 packing and zlib wrapper. The actual
on-disk encoded file is decoded before PSNR is measured. No claim of codec SOTA
is made. Compare rate-distortion curves, not just individual settings.

Examples:
  python research/universe/experiment_51_fair_sharing.py --self-test
  python research/universe/experiment_51_fair_sharing.py --size 32 --widths 8,16 --epochs 100 --seeds 42,123
  python research/universe/experiment_51_fair_sharing.py --images-dir ./images --count 30 --k 5,10 --size 32 --epochs 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import zlib

import numpy as np
import torch
from torch import nn

MAGIC = b"BLK51"
torch.set_num_threads(2)


def image_grid(size: int) -> torch.Tensor:
    points = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(points, points, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)


class Siren(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, omega: float):
        super().__init__()
        self.omega = omega
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, width)]
            + [nn.Linear(width, width) for _ in range(depth)]
            + [nn.Linear(width, 1)]
        )
        with torch.no_grad():
            for i, layer in enumerate(self.layers[:-1]):
                fan_in = layer.in_features
                bound = 1.0 / fan_in if i == 0 else math.sqrt(6 / fan_in) / omega
                layer.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega * layer(x))
        return self.layers[-1](x).squeeze(-1)


class ConditionalSiren(nn.Module):
    """Image identity enters through learned, explicitly serialized latent codes."""

    def __init__(self, group_size: int, width: int, depth: int, omega: float, latent_dim: int):
        super().__init__()
        self.net = Siren(2 + latent_dim, width, depth, omega)
        self.latents = nn.Parameter(torch.randn(group_size, latent_dim) * 0.03)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        n = self.latents.shape[0]
        p = coords.shape[0]
        positions = coords.unsqueeze(0).expand(n, -1, -1)
        z = self.latents.unsqueeze(1).expand(-1, p, -1)
        return self.net(torch.cat((positions, z), dim=-1))


def collect_images(size: int, images_dir: str | None, count: int) -> tuple[np.ndarray, list[str]]:
    if images_dir:
        from PIL import Image
        files = sorted(
            p for p in Path(images_dir).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp")
        )[:count]
        if not files:
            raise ValueError("No supported images in --images-dir")
        arrays = []
        for p in files:
            with Image.open(p) as im:
                arrays.append(np.asarray(
                    im.convert("L").resize((size, size), Image.Resampling.LANCZOS),
                    dtype=np.float32,
                ) / 255.0)
        return np.stack(arrays), [str(p) for p in files]

    # Ten distinct canonical scikit-image subjects. Never call augmented crops
    # "independent photographs"; use --images-dir for independent large corpora.
    from skimage import color, data, transform
    sources = [
        ("astronaut", data.astronaut), ("camera", data.camera),
        ("cell", data.cell), ("coins", data.coins),
        ("moon", data.moon), ("page", data.page),
        ("text", data.text), ("clock", data.clock),
        ("coffee", data.coffee), ("chelsea", data.chelsea),
    ]
    output, names = [], []
    for name, fn in sources[:count]:
        image = fn()
        if image.ndim == 3:
            image = color.rgb2gray(image)
        image = transform.resize(
            image, (size, size), preserve_range=False, anti_aliasing=True
        )
        output.append(np.clip(image.astype(np.float32), 0, 1))
        names.append(name)
    return np.stack(output), names



def collect_smooth_images(size: int, count: int) -> tuple[np.ndarray, list[str]]:
    """Deterministic, correlated analytic textures; NOT independent photographs."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    x = x / (size - 1) * 2 - 1
    y = y / (size - 1) * 2 - 1
    images, names = [], []
    for i in range(count):
        amplitude = 0.18 + 0.03 * (i % 4)
        phase = 0.14 * i
        surface = (
            0.5 + amplitude * np.sin(3.8 * x + 0.6 * y + phase)
            + 0.14 * np.cos(3.1 * y - 0.8 * x + 0.65 * phase)
            + 0.09 * np.sin(2.2 * x + 2.6 * y - 0.3 * phase)
        )
        images.append(np.clip(surface, 0, 1).astype(np.float32))
        names.append(f"synthetic_smooth_{i}")
    return np.stack(images), names


def codec_baselines(images: np.ndarray) -> list[dict]:
    """Ordinary independent PNG/JPEG/WebP files; all bytes and headers counted."""
    from PIL import Image
    import io
    result = []
    for codec, qualities in [
        ("PNG", [None]), ("JPEG", [10, 25, 40, 60, 80, 95]),
        ("WEBP", [10, 25, 40, 60, 80, 95]),
    ]:
        for quality in qualities:
            sizes, decoded = [], []
            for img in images:
                # Reference remains the original float32 tensor for both arms.
                pixels = np.rint(img * 255).clip(0, 255).astype(np.uint8)
                bio = io.BytesIO()
                kwargs = {} if quality is None else {"quality": quality}
                Image.fromarray(pixels).save(bio, format=codec, **kwargs)
                payload = bio.getvalue()
                sizes.append(len(payload))
                with Image.open(io.BytesIO(payload)) as restored:
                    decoded.append(
                        np.asarray(restored.convert("L"), dtype=np.float32) / 255.0
                    )
            encoded_bytes = sum(sizes)
            restored_images = np.stack(decoded)
            mse = np.square(images - restored_images).mean(axis=(1, 2))
            psnr = -10 * np.log10(np.maximum(mse, 1e-12))
            result.append({
                "codec": codec, "quality": quality,
                "bytes": encoded_bytes,
                "bpp": 8 * encoded_bytes / images.size,
                "mean_psnr_db": float(psnr.mean()),
                "min_psnr_db": float(psnr.min()),
                "note": "8-bit grayscale rounding; each image is a separate file",
            })
    return result


def cluster_images(images: np.ndarray, k: int, seed: int) -> list[list[int]]:
    from sklearn.cluster import KMeans
    n = len(images)
    if not (1 <= k < n):
        raise ValueError(f"For a real sharing test, require 1 <= K < N; got K={k}, N={n}")
    features = []
    for img in images:
        features.append([
            float(img.mean()), float(img.std()),
            float(np.quantile(img, 0.1)), float(np.quantile(img, 0.5)),
            float(np.quantile(img, 0.9)), float(np.abs(np.diff(img, axis=0)).mean()),
            float(np.abs(np.diff(img, axis=1)).mean()),
        ])
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(features)
    groups = [
        np.flatnonzero(labels == j).astype(int).tolist()
        for j in sorted(set(labels))
    ]
    assert sum(len(g) for g in groups) == n
    assert any(len(g) > 1 for g in groups)
    return groups


def train_individual(
    images: np.ndarray, width: int, depth: int, omega: float,
    epochs: int, lr: float, seed: int,
) -> list[nn.Module]:
    coords = image_grid(images.shape[1])
    models = []
    for i, image in enumerate(images):
        torch.manual_seed(seed + i)
        model = Siren(2, width, depth, omega)
        target = torch.from_numpy(image.flatten())
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = (model(coords) - target).square().mean()
            loss.backward()
            opt.step()
        models.append(model.eval())
    return models


def train_conditional(
    images: np.ndarray, groups: list[list[int]], width: int, depth: int,
    omega: float, latent_dim: int, epochs: int, lr: float, seed: int,
) -> list[nn.Module]:
    coords = image_grid(images.shape[1])
    models = []
    for j, indices in enumerate(groups):
        torch.manual_seed(seed + 5000 + j)
        model = ConditionalSiren(len(indices), width, depth, omega, latent_dim)
        target = torch.from_numpy(images[indices].reshape(len(indices), -1))
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = (model(coords) - target).square().mean()
            loss.backward()
            opt.step()
        models.append(model.eval())
    return models


def pack_recipe(
    models: list[nn.Module], mode: str, groups: list[list[int]],
    count: int, size: int, width: int, depth: int, omega: float, latent_dim: int,
) -> bytes:
    header = {
        "version": 1, "mode": mode, "groups": groups, "count": count,
        "size": size, "width": width, "depth": depth,
        "omega": omega, "latent_dim": latent_dim,
        "precision": "float16_le",
    }
    header_data = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    raw = bytearray(MAGIC + struct.pack("<I", len(header_data)) + header_data)
    for model in models:
        for param in model.parameters():
            raw.extend(param.detach().cpu().numpy().astype("<f2").tobytes())
    return zlib.compress(bytes(raw), level=9)


def unpack_recipe(blob: bytes) -> tuple[list[nn.Module], dict]:
    data = zlib.decompress(blob)
    if len(data) < 9 or data[:5] != MAGIC:
        raise ValueError("Invalid BLK51 file")
    header_len = struct.unpack("<I", data[5:9])[0]
    if not (0 < header_len < 1_000_000) or len(data) < 9 + header_len:
        raise ValueError("Invalid BLK51 header")
    header = json.loads(data[9:9 + header_len])
    if header.get("precision") != "float16_le":
        raise ValueError("Unsupported precision")
    mode = header["mode"]
    if mode not in ("individual", "conditional"):
        raise ValueError("Unsupported mode")
    groups = header["groups"]
    n = int(header["count"])
    if sorted(i for g in groups for i in g) != list(range(n)):
        raise ValueError("Missing or duplicated image IDs")
    if mode == "individual" and any(len(g) != 1 for g in groups):
        raise ValueError("Invalid individual model mapping")
    offset = 9 + header_len
    models = []
    for g in groups:
        if mode == "individual":
            model = Siren(2, header["width"], header["depth"], header["omega"])
        else:
            model = ConditionalSiren(
                len(g), header["width"], header["depth"],
                header["omega"], header["latent_dim"],
            )
        with torch.no_grad():
            for param in model.parameters():
                length = param.numel() * 2
                if offset + length > len(data):
                    raise ValueError("Truncated BLK51 parameters")
                values = np.frombuffer(
                    data, dtype="<f2", count=param.numel(), offset=offset
                ).astype(np.float32).copy()
                param.copy_(torch.from_numpy(values.reshape(tuple(param.shape))))
                offset += length
        models.append(model.eval())
    if offset != len(data):
        raise ValueError("Unexpected BLK51 trailing bytes")
    return models, header


def reconstruct(blob: bytes) -> np.ndarray:
    """Crucial: outputs come ONLY from the serialized, quantized payload."""
    models, header = unpack_recipe(blob)
    size, n = header["size"], header["count"]
    coords = image_grid(size)
    images = np.empty((n, size, size), dtype=np.float32)
    with torch.no_grad():
        for model, g in zip(models, header["groups"]):
            if header["mode"] == "individual":
                result = model(coords).reshape(1, size, size)
            else:
                result = model(coords).reshape(len(g), size, size)
            images[g] = result.clamp(0, 1).cpu().numpy()
    return images


def metrics(reference: np.ndarray, decoded: np.ndarray, blob: bytes) -> dict:
    if reference.shape != decoded.shape or not np.isfinite(decoded).all():
        raise ValueError("Decode failed shape or finite-value checks")
    mse_per_image = np.square(reference - decoded).mean(axis=(1, 2))
    psnr_per_image = -10 * np.log10(np.maximum(mse_per_image, 1e-12))
    return {
        "bytes": len(blob),
        "bpp": 8 * len(blob) / reference.size,
        "mean_psnr_db": float(np.mean(psnr_per_image)),
        "min_psnr_db": float(np.min(psnr_per_image)),
        "max_psnr_db": float(np.max(psnr_per_image)),
        "image_psnr_db": [float(v) for v in psnr_per_image],
        "sha256_encoded_file": hashlib.sha256(blob).hexdigest(),
        "verified_post_decode": True,
    }


def one_run(
    images: np.ndarray, mode: str, k: int, width: int, depth: int,
    omega: float, latent_dim: int, epochs: int, lr: float, seed: int,
    out_dir: Path | None,
) -> dict:
    n, h, w = images.shape
    if h != w:
        raise ValueError("Only square images in Exp 51")
    groups = [[i] for i in range(n)] if mode == "individual" else cluster_images(images, k, seed)
    start = time.perf_counter()
    if mode == "individual":
        models = train_individual(images, width, depth, omega, epochs, lr, seed)
    else:
        models = train_conditional(
            images, groups, width, depth, omega, latent_dim, epochs, lr, seed
        )
    train_seconds = time.perf_counter() - start
    blob = pack_recipe(models, mode, groups, n, h, width, depth, omega, latent_dim)
    decoded = reconstruct(blob)
    result = metrics(images, decoded, blob)
    result.update({
        "mode": mode, "k": (n if mode == "individual" else k),
        "n": n, "size": h, "width": width, "depth": depth,
        "omega": omega, "latent_dim": latent_dim,
        "epochs": epochs, "seed": seed,
        "train_seconds": round(train_seconds, 3),
        "actual_shared_images": sum(len(g) for g in groups if len(g) > 1),
        "group_sizes": [len(g) for g in groups],
    })
    if out_dir is not None:
        name = f"exp51_{mode}_n{n}_k{k}_h{width}_s{seed}.blkh51"
        (out_dir / name).write_bytes(blob)
        assert hashlib.sha256((out_dir / name).read_bytes()).hexdigest() == result["sha256_encoded_file"]
        result["encoded_file"] = name
    return result


def self_test() -> None:
    rng = np.random.default_rng(7)
    images = rng.random((3, 8, 8), dtype=np.float32)
    assert len(cluster_images(images, 1, seed=42)) == 1
    try:
        cluster_images(images, 3, seed=42)
        raise AssertionError("K=N must raise")
    except ValueError:
        pass
    for mode in ("individual", "conditional"):
        result = one_run(
            images, mode, 1, 8, 1, 30.0, 2, 2, 1e-3, 42, None,
        )
        assert result["verified_post_decode"] and result["bytes"] > 0
        assert result["bpp"] > 0 and np.isfinite(result["mean_psnr_db"])
        assert result["actual_shared_images"] == (0 if mode == "individual" else 3)
    print("Exp 51 self-test PASS: container roundtrip, strict K<N, post-decode PSNR")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--dataset", choices=["natural", "smooth"], default="natural")
    ap.add_argument("--compare-codecs", action="store_true")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--widths", default="8,16")
    ap.add_argument("--k", default="2,5")
    ap.add_argument("--seeds", default="42,123,2024")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--omega", type=float, default=30.0)
    ap.add_argument("--latent-dim", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="research/universe/_exp51_out")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.size < 4 or args.epochs < 1 or args.count < 2:
        ap.error("Require size>=4, epochs>=1, count>=2")
    if args.dataset == "smooth" and args.images_dir:
        ap.error("Choose either --dataset smooth or --images-dir")
    images, names = (
        collect_smooth_images(args.size, args.count)
        if args.dataset == "smooth"
        else collect_images(args.size, args.images_dir, args.count)
    )
    widths = [int(v) for v in args.widths.split(",")]
    k_values = [int(v) for v in args.k.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    if any(not 1 <= k < len(images) for k in k_values):
        ap.error(f"For genuine sharing every K must be less than N={len(images)}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for width in widths:
        for seed in seeds:
            for mode, kval in [("individual", len(images))] + [
                ("conditional", k) for k in k_values
            ]:
                result = one_run(
                    images, mode, kval, width, args.depth, args.omega,
                    args.latent_dim, args.epochs, args.lr, seed, out_dir,
                )
                all_results.append(result)
                print(
                    f"{mode:12} K={kval:<3} h={width:<3} seed={seed}: "
                    f"{result['bytes']:>7} B, {result['mean_psnr_db']:6.2f} dB "
                    f"POST DECODE, {result['train_seconds']}s", flush=True,
                )
    manifest = {
        "experiment": 51,
        "reproducibility": "float16 model bytes + zlib; decode and evaluate",
        "data_type": ("independent input files" if args.images_dir else
                       "correlated analytic textures" if args.dataset == "smooth" else
                       "10 distinct scikit-image subjects"),
        "image_names": names,
        "dataset": args.dataset,
        "codec_baselines": codec_baselines(images) if args.compare_codecs else [],
        "image_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
        "note": "No SOTA claim. Compare full RD curves and training cost. "
                "This is a model-sharing ablation, not a matched-quality codec leaderboard.",
        "results": all_results,
    }
    manifest_path = out_dir / "results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"RESULTS {manifest_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
