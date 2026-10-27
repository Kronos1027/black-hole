"""
Experiment 56C — prevent misleading per-tile header overhead comparisons.

Six adjacent 64x64 patches can be reconstructed as one 128x192 mosaic.
Both BH56 and PNG/JPEG2000 lossless MUST get the same contiguous tensor.
This may REVERSE a favorable result from six independent J2K files.
Lossless integer elevations, all original bytes and packed codec metadata
counted. Genuine held-out regions N40W106, N46W122 included.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image, features

from experiment_56_reversible_dem import (
    CORE, HELDOUT, encode, decode, source_hgt, tile_cases
)


def six_to_mosaic(tiles: np.ndarray) -> np.ndarray:
    tiles = np.asarray(tiles, dtype=np.int16)
    if tiles.shape != (6, 64, 64):
        raise ValueError("Expected six 64x64 geospatially adjacent patches")
    first = np.concatenate([tiles[i] for i in range(3)], axis=1)
    second = np.concatenate([tiles[i] for i in range(3, 6)], axis=1)
    return np.concatenate([first, second], axis=0)[None, :, :]


def pack_image(original: np.ndarray, fmt: str) -> tuple[bytes, np.ndarray]:
    lo, hi = int(original.min()), int(original.max())
    arr = (original[0].astype(np.int32)-lo).astype(np.uint16)
    with io.BytesIO() as f:
        img = Image.fromarray(arr)
        if fmt == "PNG":
            img.save(f, format="PNG", optimize=True, compress_level=9)
        elif fmt == "JPEG2000":
            img.save(f, format="JPEG2000", irreversible=False)
        else:
            raise ValueError("Not a comparator")
        bytes_out = f.getvalue()
    container = struct.pack("<ii", lo, hi) + bytes_out
    with Image.open(io.BytesIO(container[8:])) as restored:
        result = np.asarray(restored).astype(np.int32) + struct.unpack("<ii", container[:8])[0]
    if not np.array_equal(result, original[0]):
        raise AssertionError("Codec was not mathematically lossless")
    return container, result[None].astype(np.int16)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/tmp/blkh_exp56_mosaic")
    p.add_argument("--cache", default="/tmp/blkh_exp56_skadi_cache")
    args = p.parse_args()
    if not features.check("jpg_2000"):
        p.error("OpenJPEG/JPEG2000 required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for region in CORE + HELDOUT:
        raw, source = source_hgt(region, Path(args.cache))
        tiles = dict(tile_cases(raw))["six_64x64"]
        mosaic = six_to_mosaic(tiles)
        roundtrip = decode(encode(mosaic))
        assert np.array_equal(roundtrip, mosaic)
        directory = out / region
        directory.mkdir(parents=True, exist_ok=True)
        variants = {
            "bh56_mosaic": (encode(mosaic), roundtrip),
        }
        for fmt in ("PNG", "JPEG2000"):
            variants[fmt.lower()+"_mosaic"] = pack_image(mosaic, fmt)
        sizes = {}
        digests = {}
        for method, (blob, rec) in variants.items():
            assert np.array_equal(rec, mosaic)
            ext = "bh56" if method == "bh56_mosaic" else "bin"
            name = f"mosaic_{method}.{ext}"
            (directory/name).write_bytes(blob)
            sizes[method] = len(blob)
            digests[method] = hashlib.sha256(blob).hexdigest()
        record = {
            "region": region, "heldout": region in HELDOUT,
            "mosaic_shape": list(mosaic.shape),
            "case": "six_64x64_tiled_as_one_128x192",
            "source": source,
            "input_sha256": hashlib.sha256(mosaic.tobytes()).hexdigest(),
            "all_byte_perfect": True,
            "sizes": sizes, "sha256": digests,
            "max_error_m": 0,
        }
        records.append(record)
        print(region, "MOSAIC", sizes, flush=True)
    (out/"results.json").write_text(json.dumps({
        "experiment": "56C",
        "purpose": "control for redundant per-tile PNG/JPEG2000 headers",
        "status": "fair same-topology lossless comparison",
        "records": records,
        "warning": "A per-tile codec advantage can vanish when the competing codec is allowed to encode a contiguous mosaic.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
