"""
Experiment 53 (exploratory): correlated REAL lunar image tiles, not SRTM.

Six adjacent nonoverlapping 64x64 regions from skimage.data.moon().
This tests whether Exp 51's smooth-synthetic sharing gain survives a genuine
photographic/astronomical texture without claiming georeferenced terrain data.

Usage:
    python research/universe/experiment_53_real_lunar_tiles.py --epochs 300 --seeds 42,123
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from skimage import data

from experiment_51_fair_sharing import codec_baselines, one_run


LOCATIONS = [(160, 128), (160, 192), (160, 256),
             (224, 128), (224, 192), (224, 256)]
SIZE = 64


def lunar_tiles() -> np.ndarray:
    moon = data.moon()
    images = np.stack([
        moon[y:y+SIZE, x:x+SIZE].astype(np.float32) / 255.0
        for y, x in LOCATIONS
    ])
    assert images.shape == (6, SIZE, SIZE)
    return images


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seeds", default="42,123")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--out", default="research/universe/_exp53_out")
    args = p.parse_args()
    if args.epochs < 1 or args.width < 1:
        p.error("epochs and width must be positive")

    images = lunar_tiles()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in [int(v) for v in args.seeds.split(",")]:
        for mode, k in [("individual", 6), ("conditional", 1)]:
            rec = one_run(
                images, mode, k, args.width, 1, 30.0, 4,
                args.epochs, 1e-3, seed, dest,
            )
            results.append(rec)
            print(
                f"{mode} seed={seed}: {rec['bytes']} B, "
                f"{rec['mean_psnr_db']:.2f} dB post-decode",
                flush=True,
            )
    baseline = codec_baselines(images)
    for rec in baseline:
        if rec["codec"] == "WEBP":
            print(
                f"WebP quality={rec['quality']}: {rec['bytes']} B, "
                f"{rec['mean_psnr_db']:.2f} dB post-decode",
                flush=True,
            )
    report = {
        "experiment": 53,
        "data": "6 nonoverlapping tiles of skimage.data.moon (photographic/astronomical grayscale, NOT SRTM)",
        "locations_y_x": LOCATIONS,
        "pixels_per_image": SIZE * SIZE,
        "sha256_input_float32": hashlib.sha256(images.tobytes()).hexdigest(),
        "epochs": args.epochs,
        "results": results,
        "codec_baselines": baseline,
        "note": "Synthetic and photographic families may have different rate-distortion behavior. No universal codec claim.",
    }
    (dest / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
