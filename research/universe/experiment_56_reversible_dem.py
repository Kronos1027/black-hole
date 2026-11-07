"""
Exp 56 — LOSSLESS real DEM codec, NOT a novel neural method.

Classic predictors (average, LOCO median, planar, Paeth) + ZigZag/LEB128
+ Brotli. This is an implementation and benchmark of established techniques.
All elevations are exact signed int16 meters. Decoder uses ONLY compressed
bytes; no hidden model. Predictor ID and all metadata needed for 3D shape
are serialized and counted. Geolocation / CRS is external, as for baselines.

Held-out HGT source tiles: N40W106, N46W122; chosen BEFORE downloading.
Core (previously examined): N37W119, N37W122.
Goal: compare lossless files vs true 16-bit PNG and raw Brotli, not lossy WebP.

  python research/universe/experiment_56_reversible_dem.py --self-test
  python research/universe/experiment_56_reversible_dem.py --out /tmp/exp56
"""
from __future__ import annotations

import argparse
import brotli
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import urllib.request

import numpy as np
from PIL import Image

MAGIC = b"BH56L"
HEADER = struct.Struct("<5sBHHHBI")
PREDICTORS = ("avg", "med", "planar", "paeth")
MAX_PIXELS = 2_000_000
CORE = ("N37W119", "N37W122")
HELDOUT = ("N40W106", "N46W122")
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat}/{tile}.hgt.gz"
HGT_DIM = 3601
ORIGINS = ((1200, 1300), (1200, 1364), (1200, 1428),
           (1264, 1300), (1264, 1364), (1264, 1428))


def predict(tile, y: int, x: int, mode: str) -> int:
    if y == 0 and x == 0:
        return 0
    if y == 0:
        return int(tile[y, x - 1])
    if x == 0:
        return int(tile[y - 1, x])
    a, b, c = int(tile[y, x-1]), int(tile[y-1, x]), int(tile[y-1, x-1])
    if mode == "avg":
        return (a + b) // 2
    if mode == "planar":
        return a + b - c
    if mode == "med":
        return min(max(a + b - c, min(a, b)), max(a, b))
    if mode == "paeth":
        p = a + b - c
        da, db, dc = abs(p - a), abs(p - b), abs(p - c)
        return a if da <= db and da <= dc else (b if db <= dc else c)
    raise ValueError("Unknown prediction mode")


def write_varint(buf: bytearray, unsigned: int) -> None:
    if unsigned < 0:
        raise ValueError("Must write unsigned integer")
    while unsigned >= 128:
        buf.append((unsigned & 127) | 128)
        unsigned >>= 7
    buf.append(unsigned)


def read_varint(buf: bytes, offset: int) -> tuple[int, int]:
    accum = 0
    for shift in (0, 7, 14, 21, 28):
        if offset >= len(buf):
            raise ValueError("Truncated variable-length integer")
        byte = buf[offset]
        offset += 1
        accum |= (byte & 127) << shift
        if byte < 128:
            return accum, offset
    raise ValueError("Invalid variable-length integer")


def residual_bytes(tiles: np.ndarray, mode: str) -> bytes:
    if mode not in PREDICTORS:
        raise ValueError("Unknown predictor")
    a = np.asarray(tiles)
    if a.ndim != 3 or a.size < 1 or a.size > MAX_PIXELS:
        raise ValueError("Bad dimensions")
    if not np.isfinite(a).all() or not np.all(a == np.rint(a)):
        raise ValueError("Lossless codec requires exact integral elevations")
    if a.min() < -32768 or a.max() > 32767:
        raise ValueError("Out of signed int16 range")
    arr = a.astype(np.int32, copy=False)
    out = bytearray()
    for tile in arr:
        for y in range(tile.shape[0]):
            for x in range(tile.shape[1]):
                err = int(tile[y, x]) - predict(tile, y, x, mode)
                write_varint(out, 2*err if err >= 0 else -2*err-1)
    return bytes(out)


def encode(tiles: np.ndarray, predictor: str = "auto", quality: int = 8) -> bytes:
    arr = np.asarray(tiles)
    if arr.ndim != 3 or any(not 1 <= x <= 4096 for x in arr.shape) or arr.size > MAX_PIXELS:
        raise ValueError("Invalid input array shape")
    if not 0 <= quality <= 11:
        raise ValueError("Invalid Brotli quality")
    modes = PREDICTORS if predictor == "auto" else (predictor,)
    if any(mode not in PREDICTORS for mode in modes):
        raise ValueError("Unknown predictor")
    best = None
    for mode in modes:
        stream = residual_bytes(arr, mode)
        compressed = brotli.compress(stream, quality=quality)
        header = HEADER.pack(MAGIC, 1, *arr.shape, PREDICTORS.index(mode), len(compressed))
        payload = header + compressed
        if best is None or len(payload) < len(best):
            best = payload
    return best


def decode(blob: bytes) -> np.ndarray:
    if len(blob) < HEADER.size:
        raise ValueError("Truncated header")
    magic, version, n, h, w, method, size = HEADER.unpack(blob[:HEADER.size])
    if (magic != MAGIC or version != 1 or method >= len(PREDICTORS)
            or any(not 1 <= v <= 4096 for v in (n, h, w))
            or n*h*w > MAX_PIXELS or size != len(blob)-HEADER.size):
        raise ValueError("Invalid BH56 metadata")
    decoder = brotli.Decompressor()
    buf = bytearray()
    compressed = blob[HEADER.size:]
    for i in range(0, len(compressed), 4096):
        buf.extend(decoder.process(compressed[i:i+4096]))
        if len(buf) > n*h*w*5:
            raise ValueError("Oversized residual stream")
    if not decoder.is_finished():
        raise ValueError("Incomplete Brotli bitstream")
    data = bytes(buf)
    decoded = np.zeros((n, h, w), dtype=np.int16)
    index = 0
    for tile in decoded:
        for y in range(h):
            for x in range(w):
                u, index = read_varint(data, index)
                signed = (u >> 1) if (u & 1) == 0 else -(u >> 1)-1
                elevation = predict(tile, y, x, PREDICTORS[method]) + signed
                if elevation < -32768 or elevation > 32767:
                    raise ValueError("Decoded elevation out of range")
                tile[y, x] = elevation
    if index != len(data):
        raise ValueError("Trailing residual data")
    return decoded


def self_test() -> None:
    rng = np.random.default_rng(7)
    cases = (
        np.zeros((2, 8, 12), dtype=np.int16),
        np.full((3, 4, 5), -240, dtype=np.int16),
        rng.integers(-1000, 1000, (3, 17, 11), dtype=np.int16),
        np.arange(500, dtype=np.int16).reshape(1, 20, 25),
        np.array([[[-32768, 32767], [-32767, 32766]]], dtype=np.int16),
    )
    for original in cases:
        for mode in (*PREDICTORS, "auto"):
            coded = encode(original, predictor=mode)
            assert np.array_equal(decode(coded), original)
            try:
                decode(coded[:-1])
            except (ValueError, brotli.error):
                pass
            else:
                raise AssertionError("Truncated stream should fail")
    print("EXP56 SELF-TEST PASS: lossless all modes, truncation rejected", flush=True)


def source_hgt(tile: str, cache: Path) -> tuple[np.ndarray, dict]:
    if tile not in CORE + HELDOUT:
        raise ValueError("Unregistered source tile")
    uri = TILE_URL.format(lat=tile[:3], tile=tile)
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / (tile + ".hgt.gz")
    if local.exists():
        compressed = local.read_bytes()
    else:
        request = urllib.request.Request(
            uri, headers={"User-Agent": "BLKH-Exp56-Research/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as connection:
            compressed = connection.read(32 * 1024 * 1024 + 1)
        if len(compressed) > 32 * 1024 * 1024:
            raise ValueError("Source HGT compressed file too large")
        local.write_bytes(compressed)
    raw = gzip.decompress(compressed)
    if len(raw) != HGT_DIM*HGT_DIM*2:
        raise ValueError("Not a Skadi 3601x3601 HGT tile")
    img = np.frombuffer(raw, dtype=">i2").reshape(HGT_DIM, HGT_DIM)
    return img, {
        "url": uri, "hgt_tile": tile,
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "source": "public Mapzen/AWS Terrain Tiles Skadi SRTM-derived",
    }


def tile_cases(hgt: np.ndarray) -> list[tuple[str, np.ndarray]]:
    result = []
    blocks = [
        np.asarray(hgt[y:y+64, x:x+64], dtype=np.int16)
        for y, x in ORIGINS
    ]
    if any(tile.shape != (64, 64) for tile in blocks):
        raise ValueError("Invalid tile origins")
    result.append(("six_64x64", np.stack(blocks)))
    patch = np.asarray(hgt[1200:1456, 1300:1556], dtype=np.int16)
    if patch.shape != (256, 256):
        raise ValueError("Invalid original 256-square area")
    for side in (64, 128, 256):
        stride = 256 // side
        averaged = patch.astype(np.float64).reshape(
            side, stride, side, stride
        ).mean(axis=(1, 3))
        result.append((f"one_{side}x{side}", np.rint(averaged).astype(np.int16)[None]))
    for name, arr in result:
        if (arr == -32768).any():
            raise ValueError(f"Void in preregistered area {name}")
    return result


def png16_bytes(tiles: np.ndarray) -> tuple[int, bool]:
    lo, hi = int(tiles.min()), int(tiles.max())
    count = 8  # signed origin metadata
    for tile in tiles:
        shifted = (tile.astype(np.int32)-lo).astype(np.uint16)
        with io.BytesIO() as fd:
            Image.fromarray(shifted).save(fd, format="PNG")
            encoded = fd.getvalue()
        count += 4 + len(encoded)
        with Image.open(io.BytesIO(encoded)) as decoded:
            reconstructed = np.asarray(decoded, dtype=np.int32)+lo
        if not np.array_equal(reconstructed, tile):
            raise AssertionError("PNG16 did not round-trip")
    return count, True


def run_regions(out: Path, cache: Path, cases_limit: int | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for tile_name in CORE + HELDOUT:
        hgt, source = source_hgt(tile_name, cache)
        for case_name, src in tile_cases(hgt):
            if cases_limit and case_name != "six_64x64":
                continue
            started = time.perf_counter()
            encoded = encode(src)
            decoded = decode(encoded)
            if not np.array_equal(src, decoded):
                raise AssertionError("LOSSLESS ROUNDTRIP FAILED")
            png, _ = png16_bytes(src)
            raw_brotli = len(brotli.compress(src.astype("<i2").tobytes(), quality=8))+6
            destination = out / tile_name
            destination.mkdir(parents=True, exist_ok=True)
            artifact = destination / (case_name+".bh56")
            artifact.write_bytes(encoded)
            mode = PREDICTORS[HEADER.unpack(encoded[:HEADER.size])[5]]
            row = {
                "region": tile_name, "heldout_geographic_region": tile_name in HELDOUT,
                "case": case_name, "input_shape": list(src.shape),
                "source": source, "source_sha256": hashlib.sha256(src.tobytes()).hexdigest(),
                "codec": "BH56 lossless predictive ZigZag/LEB128 Brotli",
                "predictor_chosen": mode, "bytes": len(encoded),
                "png16_bytes": png, "raw_brotli_bytes": raw_brotli,
                "saving_vs_png16_percent": 100*(1-len(encoded)/png),
                "decoded_equal_source": True, "max_abs_error_m": 0,
                "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
                "file": str(artifact.relative_to(out)),
                "encoding_seconds": time.perf_counter()-started,
            }
            records.append(row)
            print(tile_name, case_name, mode, "BH56", len(encoded),
                  "PNG16", png, "saving_pct", round(row["saving_vs_png16_percent"], 2),
                  flush=True)
        (out/"results.json").write_text(json.dumps(
            {"experiment": 56, "status": "reproduced only after CI pass",
             "data_warning": "4 HGT source regions, 16 correlated cases (not independent cases)",
             "non_neural": True, "known_methods": ["spatial prediction", "ZigZag", "LEB128", "Brotli"],
             "geospatial_metadata": "shared external for both codecs",
             "records": records}, indent=2), encoding="utf8")
    return {"records": records}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default="research/universe/_exp56_out")
    p.add_argument("--cache", default="/tmp/blkh_exp56_skadi_cache")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    result = run_regions(Path(args.out), Path(args.cache),
                         cases_limit=1 if args.smoke else None)
    if len(result["records"]) != (4 if args.smoke else 16):
        raise AssertionError("Incomplete preregistered corpus")
    print("EXP56 DONE: all real DEMs round-tripped byte-perfect", flush=True)


if __name__ == "__main__":
    main()
