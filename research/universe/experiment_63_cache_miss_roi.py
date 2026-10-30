"""
Experiment 63 — cache-miss stress test and real encoding throughput.

Falsifiable objective: Test whether BH60's measured warm-cache advantage in
Exp62 survives when a 256x256 terrain image has 64 independently decodable
32x32 tiles but its decoded-pixel LRU holds only 8 or 32 tiles. The comparison
includes a 64-tile BH60 LRU, and persistently opened GDAL GeoTIFF datasets
(DEFLATE, ZSTD, LZW) under identical ROI request sequences.

Three deterministic workloads, per source:
  cyclic64:    64 different ROI blocks, repeated 3x; LRU-32 must miss 100%
  uniform256:  256 IID positions across all 64 blocks (mixed hit/miss)
  hotspot256:  85% accesses to 8 blocks, 15% to the other 56 blocks.

Also measures median repeated whole-file encoding elapsed time, including
writing output and closing, 3 repeats per codec. All complete georeferenced
archive bytes count. 6 real Skadi HGT regions, one 256² crop per region.
Each pixel is an exact signed int16 meter; no lossy comparisons.

Important limitations: GDAL has its own tile and OS caches, not equivalently
bounded by the BH60 Python LRU limits. In-process timings on shared CI are
NOT cold-disk latencies, CPU-normalized benchmarks, or deployment SLAs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rasterio
from rasterio.windows import Window

from experiment_60_geotiff_roi import (
    REGIONS, MORE_REGIONS, CODECS, ROI_SIDE,
    read_holdout, source_hgt, source_cases,
    metadata, encode_bh60, encode_tiff, bh60_decode_full, verify_tiff,
)
from experiment_62_bounded_cache import CachedBH60

CAPACITIES = (8, 32, 64)
BH_MODES = tuple(f"bh60_lru{n}" for n in CAPACITIES)
ALL_MODES = BH_MODES + tuple(k for k in CODECS if k != "bh60")
ENCODE_REPEATS = 3
PIXEL_BYTES = ROI_SIDE * ROI_SIDE * 2


def checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sequences() -> dict[str, list[int]]:
    rng = np.random.default_rng(630063)
    cycles = list(range(64)) * 3
    random_ids = [int(x) for x in rng.integers(0, 64, size=256)]
    selector = rng.random(256) < 0.85
    hot = rng.integers(0, 8, size=256)
    cold = rng.integers(8, 64, size=256)
    locality = [int(v) for v in np.where(selector, hot, cold)]
    assert len(set(cycles)) == 64
    return {"cyclic64": cycles, "uniform256": random_ids, "hotspot256": locality}


def self_test(out: Path) -> None:
    from experiment_60_geotiff_roi import metadata
    arr = ((np.arange(256*256, dtype=np.int32) % 1000) - 100).astype(
        np.int16
    ).reshape(256, 256)
    path = out / "selftest.bh60"
    out.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_bh60(arr, metadata("N37W119", arr)))
    seqs = sequences()
    assert len(seqs["cyclic64"]) == 192
    assert len(seqs["uniform256"]) == 256
    assert len(seqs["hotspot256"]) == 256
    with CachedBH60(path, limit=32) as reader:
        for i in seqs["cyclic64"]:
            y, x = divmod(i, 8)
            tile = reader.read(y, x)
            assert np.array_equal(
                tile, arr[y*32:(y+1)*32, x*32:(x+1)*32]
            )
        assert reader.hits == 0, reader.hits
        assert reader.misses == 192, reader.misses
        assert reader.cache_bytes <= 32 * PIXEL_BYTES
    with CachedBH60(path, limit=64) as reader:
        for i in seqs["cyclic64"]:
            y, x = divmod(i, 8)
            reader.read(y, x)
        assert reader.misses == 64
        assert reader.hits == 128
    print("EXP63 SELF TEST PASS: 32/64 cache stress, exact tiles and deterministic workload")


def prepare_codecs(region: str, img: np.ndarray, out: Path) -> tuple[dict, dict]:
    out.mkdir(parents=True, exist_ok=True)
    info = metadata(region, img)
    files = {"bh60": out / "terrain.bh60"}
    for key in CODECS:
        if key != "bh60":
            files[key] = out / (key + ".tif")
    durations = {}
    for key in CODECS:
        elapsed = []
        for _ in range(ENCODE_REPEATS):
            start = time.perf_counter_ns()
            if key == "bh60":
                files[key].write_bytes(encode_bh60(img, info))
            else:
                encode_tiff(files[key], img, info, key.split("_", 1)[1])
            elapsed.append((time.perf_counter_ns() - start) / 1e9)
        durations[key] = elapsed
    restored, info_back = bh60_decode_full(files["bh60"])
    if info_back != info or not np.array_equal(restored, img):
        raise AssertionError("BH60 full georeferenced round-trip failed")
    for key in CODECS:
        if key != "bh60":
            verify_tiff(files[key], img, info)
    sizes = {key: files[key].stat().st_size for key in CODECS}
    hashes = {key: checksum(files[key].read_bytes()) for key in CODECS}
    return files, {
        "files": {key: files[key].name for key in CODECS},
        "sizes": sizes, "sha256": hashes,
        "encode_seconds_median": {key: float(np.median(v)) for key, v in durations.items()},
        "encode_seconds_samples": durations,
        "source_sha256": checksum(img.tobytes()),
        "metadata": info,
    }


def run_workload(paths: dict, original: np.ndarray,
                 name: str, requests: list[int]) -> dict:
    obs = {name: [] for name in ALL_MODES}
    kind = {name: {"hit": [], "miss": []} for name in BH_MODES}
    with ExitStack() as stack:
        readers = {
            f"bh60_lru{n}": stack.enter_context(
                CachedBH60(paths["bh60"], limit=n)
            ) for n in CAPACITIES
        }
        gdals = {mode: stack.enter_context(rasterio.open(paths[mode]))
                 for mode in CODECS if mode != "bh60"}
        widths = original.shape[1] // ROI_SIDE
        for pos, tile_id in enumerate(requests):
            y, x = divmod(tile_id, widths)
            expected = original[y*ROI_SIDE:(y+1)*ROI_SIDE,
                                x*ROI_SIDE:(x+1)*ROI_SIDE]
            rotated = ALL_MODES[pos % len(ALL_MODES):] + ALL_MODES[:pos % len(ALL_MODES)]
            for mode in rotated:
                hit = False
                if mode in readers:
                    reader = readers[mode]
                    hit = (y, x) in reader.cache
                start = time.perf_counter_ns()
                if mode in readers:
                    output = reader.read(y, x)
                else:
                    output = gdals[mode].read(
                        1, window=Window(x*ROI_SIDE, y*ROI_SIDE,
                                         ROI_SIDE, ROI_SIDE)
                    )
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                if not np.array_equal(output, expected):
                    raise AssertionError("ROI must be losslessly reconstructed")
                obs[mode].append(elapsed_ms)
                if mode in readers:
                    kind[mode]["hit" if hit else "miss"].append(elapsed_ms)
        counts = {}
        for mode in BH_MODES:
            reader = readers[mode]
            assert reader.cache_bytes <= reader.limit * PIXEL_BYTES
            assert reader.hits + reader.misses == len(requests)
            assert reader.hits == len(kind[mode]["hit"])
            assert reader.misses == len(kind[mode]["miss"])
            counts[mode] = {
                "hits": reader.hits,
                "misses": reader.misses,
                "hit_rate": reader.hits / len(requests),
                "decoded_cache_pixel_bytes": reader.cache_bytes,
                "hit_median_ms": (float(np.median(kind[mode]["hit"]))
                                  if kind[mode]["hit"] else None),
                "miss_median_ms": (float(np.median(kind[mode]["miss"]))
                                   if kind[mode]["miss"] else None),
            }
    if name == "cyclic64":
        assert counts["bh60_lru32"]["hits"] == 0
        assert counts["bh60_lru32"]["misses"] == 192
        assert counts["bh60_lru64"]["hits"] == 128
        assert counts["bh60_lru64"]["misses"] == 64
    return {
        "workload": name,
        "request_count": len(requests),
        "roi_ids_sha256": checksum(np.asarray(requests, dtype="<i4").tobytes()),
        "roi_ids": requests,
        "cache_stats": counts,
        "median_ms": {mode: float(np.median(obs[mode])) for mode in ALL_MODES},
        "p95_ms": {mode: float(np.percentile(obs[mode], 95)) for mode in ALL_MODES},
        "mean_ms": {mode: float(np.mean(obs[mode])) for mode in ALL_MODES},
        "verified_lossless": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default="/tmp/blkh_exp63")
    ap.add_argument("--cache", default="/tmp/blkh_exp63_hgt_cache")
    ap.add_argument("--regions", default=",".join(REGIONS))
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test(root)
        return
    chosen = [v.strip() for v in args.regions.split(",") if v.strip()]
    if len(chosen) != len(set(chosen)) or not chosen or any(x not in REGIONS for x in chosen):
        ap.error("Regions must be unique and on the predeclared six-region list")
    all_results = []
    workloads = sequences()
    for region in chosen:
        source, provenance = (read_holdout(region, Path(args.cache))
                              if region in MORE_REGIONS
                              else source_hgt(region, Path(args.cache)))
        cases = dict(source_cases(source))
        original = np.asarray(cases["single_256x256"], dtype=np.int16)
        if original.shape != (256, 256):
            raise AssertionError("All cases must have 64 independent 32px tiles")
        workdir = root / region
        files, prepared = prepare_codecs(region, original, workdir)
        runs = [run_workload(files, original, name, calls)
                for name, calls in workloads.items()]
        result = {
            "region": region,
            "source": provenance,
            "geometry": [256, 256],
            "tile_count": 64,
            "block_side": 32,
            "prepared_files": prepared,
            "workloads": runs,
        }
        all_results.append(result)
        for run in runs:
            print(region, run["workload"], "cache",
                  {k: round(v["hit_rate"], 3)
                   for k, v in run["cache_stats"].items()},
                  "med_ms",
                  {k: round(v, 4) for k, v in run["median_ms"].items()},
                  flush=True)
        (root / "results.json").write_text(json.dumps({
            "experiment": 63,
            "status": "provisional until workflow verifies all artifacts",
            "scientific_scope": "cache misses, real geospatial ROI, encoding time",
            "hardware_caveat": "shared hosted CPU; GDAL cache and OS cache unconstrained; not a cold-disk experiment",
            "source_count": len(all_results),
            "results": all_results,
        }, indent=2), encoding="utf-8")
    print("EXP63 DONE", len(all_results), "regions, each with 3 workloads", flush=True)


if __name__ == "__main__":
    main()
