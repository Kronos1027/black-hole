"""
BLKH / BHUH EXPERIMENT 54: Real DEM compression, measured in METERS.

Uses public Mapzen / AWS Terrain Tiles Skadi HGT source data. Elevations are
stored as signed 16-bit big-endian meter values, 3601x3601 per degree.
Samples are SIX predetermined adjacent, nonoverlapping 64x64 areas in each
region. No selective filtering by compression result is performed.

COMPARE:
 - Exp51 individual SIREN versus conditional shared SIREN (both float16/zlib)
 - WebP grayscale, evaluated against real elevation in meters
 - PNG 16-bit lossless
 - Predictive delta+zlib with 0.5,1,2,5,10,20,50m quantization

The lossy and lossless baselines MUST NOT be conflated. All quality scores
are calculated on the decoded bytes. Stored byte counts include the common
height range needed to reconstruct physical units; shared geographic
coordinates/CRS are treated as external dataset metadata for ALL codecs.

Commands:
  python research/universe/experiment_54_srtm_real_meters.py --self-test
  python research/universe/experiment_54_srtm_real_meters.py --epochs 200 --seeds 42,123
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import urllib.request
import zlib

import numpy as np
from PIL import Image
from experiment_51_fair_sharing import one_run, reconstruct


HGT_DIM = 3601
VOID = -32768
SRTM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat}/{tile}.hgt.gz"
ALLOWED_TILES = {"N37W122", "N37W119"}
ORIGINS = [(1200, 1300), (1200, 1364), (1200, 1428),
           (1264, 1300), (1264, 1364), (1264, 1428)]
SIDE = 64
META_BYTES = 8  # common min/max as two little-endian float32 values


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_hgt(tile: str, cache: Path) -> tuple[np.ndarray, dict]:
    if tile not in ALLOWED_TILES:
        raise ValueError(f"Tile {tile} not on fixed preregistered list")
    cache.mkdir(parents=True, exist_ok=True)
    uri = SRTM_URL.format(lat=tile[:3], tile=tile)
    path = cache / f"{tile}.hgt.gz"
    if not path.exists():
        request = urllib.request.Request(
            uri, headers={"User-Agent": "BLKH-DEM-Research/54 (reproducible)"}
        )
        with urllib.request.urlopen(request, timeout=120) as stream:
            compressed = stream.read(32 * 1024 * 1024 + 1)
        if len(compressed) > 32 * 1024 * 1024:
            raise ValueError("Unexpectedly large source HGT download")
        path.write_bytes(compressed)
    else:
        compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    if len(raw) != HGT_DIM * HGT_DIM * 2:
        raise ValueError(f"Unexpected DEM shape/bytes: {len(raw)}")
    data = np.frombuffer(raw, dtype=">i2").reshape(HGT_DIM, HGT_DIM)
    info = {
        "source_url": uri,
        "download_sha256": sha256(compressed),
        "decompressed_sha256": sha256(raw),
        "data_provenance": "Mapzen/AWS Terrain Tiles Skadi HGT (SRTM-derived)",
        "hgt_tile_id": tile,
        "tile_grid_shape": [HGT_DIM, HGT_DIM],
        "vertical_unit": "meter",
        "source_void_value": VOID,
    }
    return data, info


def extract_tiles(raw: np.ndarray, side: int = SIDE) -> np.ndarray:
    parts = []
    for y, x in ORIGINS:
        piece = np.asarray(raw[y:y+side, x:x+side], dtype=np.int16)
        if piece.shape != (side, side):
            raise ValueError("Tile origin outside source bounds")
        if (piece == VOID).any():
            raise ValueError("NoData in preregistered region; do not mask silently")
        parts.append(piece.astype(np.float32))
    return np.stack(parts)


def evaluate_meters(original: np.ndarray, reconstruction: np.ndarray) -> dict:
    if original.shape != reconstruction.shape:
        raise ValueError("Decoded shape mismatch")
    if not np.isfinite(reconstruction).all():
        raise ValueError("Nonfinite decoded height")
    err = reconstruction.astype(np.float64) - original.astype(np.float64)
    absolute = np.abs(err)
    range_m = float(original.max() - original.min())
    rmse = float(np.sqrt(np.mean(err * err)))
    return {
        "rmse_m": rmse,
        "mean_abs_error_m": float(absolute.mean()),
        "p95_abs_error_m": float(np.percentile(absolute, 95)),
        "max_abs_error_m": float(absolute.max()),
        "global_psnr_db": float(20 * np.log10(range_m / max(rmse, 1e-12))),
        "elevation_span_m": range_m,
    }


def baseline_webp(original: np.ndarray, quality: int) -> tuple[bytes, np.ndarray]:
    lo = float(original.min())
    hi = float(original.max())
    scale = hi - lo
    if scale == 0:
        raise ValueError("Flat DEM has no dynamic range")
    payload = bytearray()
    decoded = []
    for tile in original:
        u8 = np.rint((tile - lo) * (255 / scale)).clip(0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(u8).save(buffer, format="WEBP", quality=quality)
        block = buffer.getvalue()
        payload += struct.pack("<I", len(block)) + block
        with Image.open(io.BytesIO(block)) as restored:
            decoded.append(
                np.asarray(restored.convert("L"), dtype=np.float32) *
                (scale / 255) + lo
            )
    return struct.pack("<ff", lo, hi) + bytes(payload), np.stack(decoded)


def baseline_png16(original: np.ndarray) -> tuple[bytes, np.ndarray]:
    lo = int(original.min())
    hi = int(original.max())
    if hi - lo > 65535:
        raise ValueError("Elevation span exceeds unsigned 16-bit offset")
    payload = bytearray()
    decoded = []
    for tile in original:
        values = (tile.astype(np.int32) - lo).astype(np.uint16)
        buffer = io.BytesIO()
        Image.fromarray(values).save(buffer, format="PNG")
        block = buffer.getvalue()
        payload += struct.pack("<I", len(block)) + block
        with Image.open(io.BytesIO(block)) as restored:
            decoded.append(np.asarray(restored).astype(np.float32) + lo)
    return struct.pack("<ii", lo, hi) + bytes(payload), np.stack(decoded)


def baseline_predictive(original: np.ndarray, step: float) -> tuple[bytes, np.ndarray]:
    if step <= 0:
        raise ValueError("step must be > 0")
    lo = float(original.min())
    q = np.rint((original - lo) / step).astype(np.int32)
    # Horizontal delta predictor; first symbol of each row stored absolutely.
    residual = np.empty_like(q)
    residual[:, :, 0] = q[:, :, 0]
    residual[:, :, 1:] = q[:, :, 1:] - q[:, :, :-1]
    if residual.min() < -32768 or residual.max() > 32767:
        raise ValueError("Residual exceeds int16; change coding format")
    h, w = original.shape[1:]
    header = struct.pack("<5sffHHH", b"BHD54", lo, step, len(original), h, w)
    packed = header + zlib.compress(residual.astype("<i2").tobytes(), 9)
    # Real stream reconstruction, NEVER assess raw pre-quantization q.
    return packed, unpack_predictive(packed)


def unpack_predictive(data: bytes) -> np.ndarray:
    hsize = struct.calcsize("<5sffHHH")
    if len(data) < hsize:
        raise ValueError("Truncated predictor header")
    magic, lo, step, n, h, w = struct.unpack("<5sffHHH", data[:hsize])
    if magic != b"BHD54" or not (0 < n <= 200 and 0 < h <= 4096 and 0 < w <= 4096):
        raise ValueError("Corrupted predictive metadata")
    expected = n * h * w * 2
    if expected > 100 * 1024 * 1024:
        raise ValueError("Decompression bound exceeded")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(data[hsize:], expected + 1)
    if len(raw) != expected or decoder.unconsumed_tail or not decoder.eof or decoder.unused_data:
        raise ValueError("Corrupted/truncated predictor payload")
    residual = np.frombuffer(raw, dtype="<i2").reshape(n, h, w).astype(np.int32)
    q = np.cumsum(residual, axis=2, dtype=np.int32)
    return q.astype(np.float32) * step + lo


def baseline_raw_zlib(original: np.ndarray) -> tuple[bytes, np.ndarray]:
    # Lossless signed meter elevations. All tiles share one stream.
    integer = original.astype("<i2")
    payload = b"BH54Z" + struct.pack("<HHH", *integer.shape) + zlib.compress(integer.tobytes(), 9)
    restored = np.frombuffer(
        zlib.decompress(payload[11:]), dtype="<i2"
    ).reshape(integer.shape).astype(np.float32)
    return payload, restored


def attach_artifact(
    output: Path, name: str, original: np.ndarray, encoded: bytes,
    reconstructed: np.ndarray, method: str, **kwargs,
) -> dict:
    path = output / name
    path.write_bytes(encoded)
    assert path.read_bytes() == encoded
    return {
        "method": method, **kwargs,
        "bytes": len(encoded), "bits_per_elevation": 8 * len(encoded) / original.size,
        "sha256": sha256(encoded), "file": name,
        "post_decode_verified": True, **evaluate_meters(original, reconstructed),
    }


def benchmark_region(
    original: np.ndarray, source: dict, output: Path,
    epochs: int, seeds: list[int], width: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    lo, hi = float(original.min()), float(original.max())
    if hi <= lo:
        raise ValueError("No terrain variation")
    normalized = (original - lo) / (hi - lo)
    neural, classical = [], []
    n = len(original)
    for seed in seeds:
        for mode, k in (("individual", n), ("conditional", 1)):
            result = one_run(
                normalized, mode, k, width, 1, 30., 4,
                epochs, 1e-3, seed, output,
            )
            blob = (output / result["encoded_file"]).read_bytes()
            reconstructed = reconstruct(blob) * (hi - lo) + lo
            rec = attach_artifact(
                output, f"exp54_{mode}_s{seed}.blkh54", original,
                struct.pack("<ff", lo, hi) + blob, reconstructed,
                f"siren_{mode}", seed=seed, k=k, epochs=epochs, width=width,
                train_seconds=result["train_seconds"],
                neural_file=result["encoded_file"],
            )
            neural.append(rec)
            print(
                f"{source['hgt_tile_id']} {mode} s={seed}: "
                f"{rec['bytes']} B RMSE={rec['rmse_m']:.3f}m "
                f"max={rec['max_abs_error_m']:.2f}m",
                flush=True,
            )
    for quality in (1, 10, 25, 40, 60, 80, 95):
        buf, rec = baseline_webp(original, quality)
        classical.append(attach_artifact(
            output, f"exp54_webp_q{quality}.bin", original, buf, rec,
            "webp_8bit", quality=quality, bit_depth=8,
        ))
    for step in (.5, 1., 2., 5., 10., 20., 50.):
        buf, rec = baseline_predictive(original, step)
        classical.append(attach_artifact(
            output, f"exp54_delta_{str(step).replace('.', 'p')}m.bin",
            original, buf, rec, "delta_zlib", quantization_step_m=step,
        ))
    for label, implementation in [
        ("png16_lossless", baseline_png16),
        ("zlib_int16_lossless", baseline_raw_zlib),
    ]:
        buf, rec = implementation(original)
        classical.append(attach_artifact(
            output, f"exp54_{label}.bin", original, buf, rec, label,
        ))
    for point in neural:
        faster = [
            base for base in classical
            if base["bytes"] <= point["bytes"]
            and base["rmse_m"] <= point["rmse_m"]
            and (base["bytes"] < point["bytes"] or base["rmse_m"] < point["rmse_m"])
        ]
        point["dominated_by_classical"] = [
            {"method": p["method"], "bytes": p["bytes"],
             "rmse_m": p["rmse_m"], "quality": p.get("quality"),
             "quantization_step_m": p.get("quantization_step_m")}
            for p in faster
        ]
    return {
        "source": source,
        "tile_origins": ORIGINS,
        "tile_side": SIDE,
        "num_tiles": n,
        "height_unit": "meter",
        "meta_bytes_per_neural_or_webp_corpus": META_BYTES,
        "sha256_input_float32": sha256(original.tobytes()),
        "height_min_m": lo, "height_max_m": hi,
        "normalization": "same global min/max for entire 6-tile corpus",
        "limitations": [
            "SRTM-derived Skadi data; check provenance and vertical datum for geodetic use.",
            "All adjacent tiles are trained and evaluated in sample; not out-of-region generalization.",
            "Neural encoding requires gradient training. Runtime and decode cost matter.",
            "All methods assume location/CRS stored externally; no geographical metadata in encoded byte counts.",
            "WebP is 8-bit grayscale; predictive baseline remains in native elevation units.",
            "The experiment deliberately exposes negative results and classical dominance.",
        ],
        "neural": neural,
        "classical": classical,
    }


def self_test() -> None:
    raw = (np.arange(128 * 192, dtype=np.int32).reshape(128, 192) % 287).astype(np.int16)
    full = np.zeros((1800, 1800), dtype=np.int16)
    for y, x in ORIGINS:
        full[y:y+64, x:x+64] = raw[:64, :64]
    tiles = extract_tiles(full)
    assert tiles.shape == (6, 64, 64)
    for step in (.5, 1., 2., 5., 10.):
        blob, reconstruction = baseline_predictive(tiles, step)
        assert reconstruction.shape == tiles.shape
        assert np.max(np.abs(reconstruction - tiles)) <= step / 2 + 1e-5
        assert unpack_predictive(blob).shape == tiles.shape
    for impl in (baseline_png16, baseline_raw_zlib):
        blob, reconstruction = impl(tiles)
        assert np.array_equal(tiles, reconstruction)
    webp, rec = baseline_webp(tiles, 80)
    assert len(webp) > 0 and rec.shape == tiles.shape
    assert evaluate_meters(tiles, tiles)["rmse_m"] == 0
    print("EXP54 SELF-TEST PASS: true post-byte reconstruction, metric units, controls")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--regions", default="N37W122,N37W119")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seeds", default="42,123")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--out", default="research/universe/_exp54_out")
    p.add_argument("--cache", default="/tmp/blkh_exp54_srtm_cache")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.epochs < 1 or args.width < 1:
        p.error("epochs and width must be positive")
    tile_names = [x.strip() for x in args.regions.split(",") if x.strip()]
    if not tile_names or len(set(tile_names)) != len(tile_names):
        p.error("Regions must be distinct")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": 54, "scientific_status": "exploratory",
        "evaluation": "post-decode error in elevation meters plus actual file size",
        "regions": [],
    }
    for tile in tile_names:
        dem, source = read_hgt(tile, Path(args.cache))
        original = extract_tiles(dem)
        region = benchmark_region(
            original, source, out / tile,
            args.epochs, [int(v) for v in args.seeds.split(",")], args.width,
        )
        report["regions"].append(region)
        (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        dominated = sum(bool(n["dominated_by_classical"]) for n in region["neural"])
        print(f"{tile}: {dominated}/{len(region['neural'])} neural variants dominated by classical baselines")
    print(f"EXP54 DONE: {out/'results.json'}")


if __name__ == "__main__":
    main()
