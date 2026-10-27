"""
Exp 56B: stringent LOSSLESS codec comparisons and timing.
Fairness: signed elevation meters are shifted to uint16 for PNG and JPEG2000;
the signed origin is included in 8-byte metadata. All files store per-tile
4-byte lengths, just as Exp54's PNG16 benchmark does. JPEG2000 is genuine
lossless (irreversible=False), decoded independently, byte-for-byte compared.

No novelty or general dominance claim. Core and two preregistered heldout
geographic regions are the same as Exp 56. This tests Exp56 vs optimized PNG,
lossless JPEG2000, and raw signed-int16 Brotli. Size and times are real.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import brotli
import numpy as np
from PIL import Image, features

from experiment_56_reversible_dem import (
    CORE, HELDOUT, decode, encode, source_hgt, tile_cases
)


def image_format_baseline(tiles: np.ndarray, fmt: str) -> tuple[int, str]:
    lo, hi = int(tiles.min()), int(tiles.max())
    if hi - lo > 65535:
        raise ValueError("Cannot express in unsigned 16 bits")
    total_bytes = 8  # stored lo/hi int32, as in previous experiment
    checksum = hashlib.sha256()
    for tile in tiles:
        src = (tile.astype(np.int32) - lo).astype(np.uint16)
        with io.BytesIO() as f:
            img = Image.fromarray(src)
            if fmt == "PNG":
                img.save(f, format="PNG", optimize=True, compress_level=9)
            elif fmt == "JPEG2000":
                img.save(f, format="JPEG2000", irreversible=False)
            else:
                raise ValueError("Unsupported comparison")
            payload = f.getvalue()
        total_bytes += 4 + len(payload)
        checksum.update(payload)
        with Image.open(io.BytesIO(payload)) as img:
            decoded = np.asarray(img).astype(np.int32) + lo
        if not np.array_equal(decoded, tile):
            raise AssertionError(f"{fmt} baseline is not lossless")
    return total_bytes, checksum.hexdigest()


def benchmark_one(tiles: np.ndarray, repeats: int) -> dict:
    if repeats < 2:
        raise ValueError("Need >=2 repeats")
    original = np.asarray(tiles, dtype=np.int16)
    encode_times = []
    decode_times = []
    coded = None
    for _ in range(repeats):
        t = time.perf_counter()
        blob = encode(original)
        encode_times.append(time.perf_counter() - t)
        t = time.perf_counter()
        restored = decode(blob)
        decode_times.append(time.perf_counter() - t)
        if not np.array_equal(original, restored):
            raise AssertionError("Not lossless")
        if coded is not None and blob != coded:
            raise AssertionError("Nondeterministic bitstream")
        coded = blob
    optimum, png_sha = image_format_baseline(original, "PNG")
    j2k, j2k_sha = image_format_baseline(original, "JPEG2000")
    raw_brotli = brotli.compress(original.astype("<i2").tobytes(), quality=8)
    return {
        "exp56_bytes": len(coded),
        "exp56_sha256": hashlib.sha256(coded).hexdigest(),
        "png16_optimized_bytes": optimum, "png_sha256": png_sha,
        "j2k_lossless_bytes": j2k, "j2k_sha256": j2k_sha,
        "raw_brotli_bytes": len(raw_brotli) + 6,
        "exp56_smaller_than_png_opt": len(coded) < optimum,
        "exp56_smaller_than_j2k": len(coded) < j2k,
        "exp56_saving_vs_png_opt_pct": 100*(1-len(coded)/optimum),
        "exp56_saving_vs_j2k_pct": 100*(1-len(coded)/j2k),
        "encode_seconds_median": float(np.median(encode_times)),
        "decode_seconds_median": float(np.median(decode_times)),
        "max_error_meters": 0,
        "lossless_verified": True,
        "source_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="/tmp/blkh_exp56_skadi_cache")
    p.add_argument("--out", default="/tmp/blkh_exp56_rigorous")
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    if not features.check("jpg_2000"):
        p.error("Lossless JPEG2000 support is mandatory, do not silently skip")
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for tile_name in CORE + HELDOUT:
        dem, provenance = source_hgt(tile_name, Path(a.cache))
        for case, tiles in tile_cases(dem):
            row = benchmark_one(tiles, a.repeats)
            row.update({
                "region": tile_name, "case": case,
                "heldout_region": tile_name in HELDOUT,
                "source": provenance,
                "input_shape": list(tiles.shape)
            })
            records.append(row)
            print(
                tile_name, case,
                "BLKH", row["exp56_bytes"],
                "PNG_optimized", row["png16_optimized_bytes"],
                "JPEG2000_lossless", row["j2k_lossless_bytes"],
                "win_J2K", row["exp56_smaller_than_j2k"],
                flush=True,
            )
        (root/"results.json").write_text(
            json.dumps({
                "experiment": "56B", "benchmark_type": "lossless real DEM",
                "baseline_warning": "Not a comprehensive JPEG-LS/JPEG-XL/FFV1 leaderboard",
                "regions_independent": len(CORE + HELDOUT),
                "cases_non_independent": len(records),
                "timing_note": "median of repeated in-process encodes/decodes; environment-specific",
                "records": records,
            }, indent=2), encoding="utf8",
        )
    if len(records) != 16:
        raise AssertionError("Expected 16 exact cases")
    print(
        "EXP56B: lossless 16 cases; vs optimized PNG",sum(x["exp56_smaller_than_png_opt"] for x in records),
        "vs JPEG2000",sum(x["exp56_smaller_than_j2k"] for x in records),
        flush=True,
    )


if __name__ == "__main__":
    main()
