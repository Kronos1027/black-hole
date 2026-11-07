"""
Experiment 64: faster *cache-miss* lossless DEM decoding with fixed-width deltas.

Exp63 established that BH60's ZipZag/LEB128 + Brotli reader is ~0.14ms
for an uncached 32x32 tile on the shared GitHub CI CPU, slower than GDAL.
Hypothesis: reversible modular uint16 spatial differences written directly
as little-endian u16 + Brotli avoid variable-length parsing; they may be
much quicker to decode at the expense of some file size.

Compare BH60 LRU-32 forced to miss every request, a new un-cached BH64 fixed
delta codec, and persistently opened ZSTD/LZW tiled GeoTIFF, all on the same
six 256x256 real Skadi HGT sources, with 64 blocks cycled three times.
Store full metadata and full file sizes, and time full output encoding x3.
All methods are strictly lossless. These are established techniques, not a
claimed new compression principle. GDAL has a different cache implementation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import struct
import time

import brotli
import numpy as np
import rasterio
from rasterio.windows import Window

from experiment_60_geotiff_roi import (
    REGIONS, MORE_REGIONS, ROI_SIDE, PRELUDE,
    read_holdout, source_hgt, source_cases, metadata,
    encode_bh60, encode_tiff, bh60_decode_full,
)
from experiment_62_bounded_cache import CachedBH60
from experiment_58_vectorized_decoder import INDEX_HEADER, ENTRY

OUTER_MAGIC = b"BHG64"
INDEX_MAGIC = b"BH64I"
PART_MAGIC = b"BHV64"
PART = struct.Struct("<5sBHHI")
CODEC_ID = 3
MAX_PIXELS = 2_000_000
ENCODE_TRIALS = 3
MODES = ("bh60_miss32", "bh64_fixed", "tiff_zstd", "tiff_lzw")


def block_encode(img: np.ndarray) -> bytes:
    if img.shape != (ROI_SIDE, ROI_SIDE) or img.dtype != np.int16:
        raise ValueError("BH64 requires an int16 32x32 tile")
    unsigned = img.view(np.uint16)
    candidates = []
    for mode in (0, 1):
        delta = np.empty_like(unsigned)
        if mode == 0:
            delta[:, 0] = unsigned[:, 0]
            delta[:, 1:] = unsigned[:, 1:] - unsigned[:, :-1]
        else:
            delta[0, :] = unsigned[0, :]
            delta[1:, :] = unsigned[1:, :] - unsigned[:-1, :]
        packed = delta.astype("<u2", copy=False).tobytes()
        compressed = brotli.compress(packed, quality=8)
        candidates.append(PART.pack(PART_MAGIC, mode, ROI_SIDE, ROI_SIDE,
                                    len(compressed)) + compressed)
    return min(candidates, key=len)


def block_decode(blob: bytes) -> np.ndarray:
    if len(blob) < PART.size:
        raise ValueError("Truncated BH64 tile")
    magic, mode, h, w, length = PART.unpack(blob[:PART.size])
    if (magic != PART_MAGIC or mode not in (0, 1) or
            (h, w) != (ROI_SIDE, ROI_SIDE) or length != len(blob) - PART.size):
        raise ValueError("Invalid BH64 block header")
    dec = brotli.Decompressor()
    raw = bytearray()
    for pos in range(PART.size, len(blob), 4096):
        raw.extend(dec.process(blob[pos:pos+4096]))
        if len(raw) > ROI_SIDE*ROI_SIDE*2:
            raise ValueError("Invalid or oversized BH64 decompression output")
    if not dec.is_finished() or len(raw) != ROI_SIDE*ROI_SIDE*2:
        raise ValueError("Incomplete BH64 tile stream")
    diff = np.frombuffer(raw, dtype="<u2").reshape((ROI_SIDE, ROI_SIDE))
    restored = np.cumsum(diff.astype(np.int32), axis=1 if mode == 0 else 0,
                         dtype=np.int32).astype(np.uint16).view(np.int16)
    return restored


def inner_index(blob: bytes) -> tuple[int, int, list[tuple[int, int]], int]:
    if len(blob) < INDEX_HEADER.size:
        raise ValueError("Truncated index header")
    magic, version, h, w, side, n, nbytes = INDEX_HEADER.unpack(
        blob[:INDEX_HEADER.size]
    )
    if (magic != INDEX_MAGIC or version != 1 or side != ROI_SIDE or
            h < side or w < side or h*w > MAX_PIXELS or
            h % side or w % side or n != (h//side)*(w//side) or
            nbytes != n*ENTRY.size):
        raise ValueError("Invalid BH64 geometry")
    start = INDEX_HEADER.size + nbytes
    if len(blob) < start:
        raise ValueError("Truncated BH64 index")
    entries = []
    offset = 0
    for i in range(n):
        codec, a, length = ENTRY.unpack_from(blob, INDEX_HEADER.size+i*ENTRY.size)
        if codec != CODEC_ID or a != offset or length < PART.size:
            raise ValueError("Invalid BH64 index item")
        offset += length
        entries.append((a, length))
    if start + offset != len(blob):
        raise ValueError("Invalid encoded BH64 archive size")
    return h, w, entries, start


def encode_image(img: np.ndarray, info: dict) -> bytes:
    if (img.dtype != np.int16 or img.ndim != 2 or
            any(v % ROI_SIDE for v in img.shape) or
            img.size > MAX_PIXELS):
        raise ValueError("Expected tile-aligned signed-int16 DEM")
    meta = json.dumps(info, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if not 0 < len(meta) < 16384:
        raise ValueError("Invalid geospatial metadata length")
    h, w = img.shape
    idx, data = bytearray(), bytearray()
    for y in range(0, h, ROI_SIDE):
        for x in range(0, w, ROI_SIDE):
            piece = block_encode(img[y:y+ROI_SIDE, x:x+ROI_SIDE])
            idx.extend(ENTRY.pack(CODEC_ID, len(data), len(piece)))
            data.extend(piece)
    total = len(idx)//ENTRY.size
    inner = INDEX_HEADER.pack(INDEX_MAGIC, 1, h, w, ROI_SIDE, total, len(idx))
    return PRELUDE.pack(OUTER_MAGIC, 1, len(meta)) + meta + inner + idx + data


def _read_meta(handle) -> tuple[dict, int]:
    header = handle.read(PRELUDE.size)
    if len(header) != PRELUDE.size:
        raise ValueError("Truncated BH64 geospatial header")
    magic, version, meta_length = PRELUDE.unpack(header)
    if magic != OUTER_MAGIC or version != 1 or not 0 < meta_length < 16384:
        raise ValueError("Invalid BH64 geospatial metadata")
    meta = handle.read(meta_length)
    if len(meta) != meta_length:
        raise ValueError("Truncated BH64 JSON")
    info = json.loads(meta)
    required = ("region", "crs", "transform", "nodata", "sample_unit",
                "dtype", "image_shape", "block_side")
    if not isinstance(info, dict) or any(k not in info for k in required):
        raise ValueError("Missing required geospatial fields")
    if (info["crs"] != "EPSG:4326" or info["sample_unit"] != "meter" or
            info["dtype"] != "int16" or info["block_side"] != ROI_SIDE):
        raise ValueError("Unsupported geospatial schema")
    return info, PRELUDE.size + meta_length


def decode_image(path: Path) -> tuple[np.ndarray, dict]:
    with path.open("rb") as f:
        info, payload_start = _read_meta(f)
        inner = f.read()
    h, w, entries, start = inner_index(inner)
    if info["image_shape"] != [h, w]:
        raise ValueError("Mismatched serialized raster dimensions")
    result = np.empty((h, w), dtype=np.int16)
    for i, (offset, length) in enumerate(entries):
        y, x = divmod(i, w//ROI_SIDE)
        piece = block_decode(inner[start+offset:start+offset+length])
        result[y*ROI_SIDE:(y+1)*ROI_SIDE, x*ROI_SIDE:(x+1)*ROI_SIDE] = piece
    return result, info


class OpenBH64:
    def __init__(self, path: Path):
        self.file = path.open("rb")
        try:
            self.info, self.start = _read_meta(self.file)
            # Fixed-size internal index, loaded once; compressed blocks remain on disk.
            head = self.file.read(INDEX_HEADER.size)
            if len(head) != INDEX_HEADER.size:
                raise ValueError("Invalid BH64 internal index")
            magic, ver, h, w, side, n, idx_bytes = INDEX_HEADER.unpack(head)
            if (magic != INDEX_MAGIC or ver != 1 or side != ROI_SIDE or
                    h < side or w < side or h*w > MAX_PIXELS or
                    h % side or w % side or
                    n != (h//side)*(w//side) or idx_bytes != n*ENTRY.size):
                raise ValueError("Invalid BH64 index geometry")
            raw_idx = self.file.read(idx_bytes)
            if len(raw_idx) != idx_bytes:
                raise ValueError("Incomplete BH64 index entries")
            self.entries = []
            offset = 0
            for i in range(n):
                codec, start, length = ENTRY.unpack_from(raw_idx, i*ENTRY.size)
                if codec != CODEC_ID or start != offset or length < PART.size:
                    raise ValueError("Corrupt BH64 index offset")
                offset += length
                self.entries.append((start, length))
            self.body_start = self.start + INDEX_HEADER.size + idx_bytes
            if self.body_start + offset != path.stat().st_size:
                raise ValueError("Invalid BH64 stream length")
            self.h, self.w = h, w
            if self.info["image_shape"] != [h, w]:
                raise ValueError("Georeferenced geometry mismatch")
        except BaseException:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.file.close()

    def read(self, row: int, col: int) -> np.ndarray:
        if not 0 <= row < self.h//ROI_SIDE or not 0 <= col < self.w//ROI_SIDE:
            raise IndexError("Outside BH64 tile grid")
        offset, length = self.entries[row*(self.w//ROI_SIDE)+col]
        self.file.seek(self.body_start+offset)
        raw = self.file.read(length)
        if len(raw) != length:
            raise ValueError("Truncated BH64 block payload")
        return block_decode(raw)


def self_test(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(64)
    images = (
        rng.integers(-32768, 32768, (64, 64), dtype=np.int16),
        ((np.arange(64*64).reshape(64,64)%65536)-32768).astype(np.int16),
        np.zeros((64,64), dtype=np.int16),
    )
    for i, img in enumerate(images):
        info = metadata("N37W119", img)
        path = root/f"sample_{i}.bh64"
        path.write_bytes(encode_image(img, info))
        result, metadata_back = decode_image(path)
        assert metadata_back == info and np.array_equal(result, img)
        with OpenBH64(path) as f:
            for y in range(2):
                for x in range(2):
                    assert np.array_equal(
                        f.read(y,x), img[y*32:(y+1)*32,x*32:(x+1)*32]
                    )
        try:
            path.write_bytes(path.read_bytes()[:-1])
            decode_image(path)
        except (ValueError, brotli.error):
            pass
        else:
            raise AssertionError("Truncated archive was accepted")
    print("EXP64 SELF-TEST PASS: exact modular deltas, geodata and truncated input rejection")


def benchmark_one(region: str, img: np.ndarray, root: Path) -> dict:
    assert img.dtype == np.int16 and img.shape == (256, 256)
    target = root/region
    target.mkdir(parents=True, exist_ok=True)
    info = metadata(region, img)
    output = {
        "bh60": target/"legacy.bh60",
        "bh64": target/"fixed16.bh64",
        "tiff_zstd": target/"tiff_zstd.tif",
        "tiff_lzw": target/"tiff_lzw.tif",
    }
    elapsed = {name: [] for name in output}
    for key, path in output.items():
        for _ in range(ENCODE_TRIALS):
            t = time.perf_counter_ns()
            if key == "bh60":
                path.write_bytes(encode_bh60(img, info))
            elif key == "bh64":
                path.write_bytes(encode_image(img, info))
            else:
                encode_tiff(path, img, info, key.split("_", 1)[1])
            elapsed[key].append((time.perf_counter_ns()-t)/1e9)
    original64, metadata64 = decode_image(output["bh64"])
    old, oldmeta = bh60_decode_full(output["bh60"])
    assert np.array_equal(original64, img) and np.array_equal(old, img)
    assert metadata64 == oldmeta == info
    with rasterio.open(output["tiff_zstd"]) as raster:
        assert np.array_equal(raster.read(1), img)
        assert raster.crs.to_string() == "EPSG:4326"
    with rasterio.open(output["tiff_lzw"]) as raster:
        assert np.array_equal(raster.read(1), img)
    samples = {name: [] for name in MODES}
    order_ids = list(range(64))*3  # deliberate 100% BH60 LRU32 misses
    with ExitStack() as st:
        oldreader = st.enter_context(CachedBH60(output["bh60"], limit=32))
        faster = st.enter_context(OpenBH64(output["bh64"]))
        tifz = st.enter_context(rasterio.open(output["tiff_zstd"]))
        tifl = st.enter_context(rasterio.open(output["tiff_lzw"]))
        for i, idx in enumerate(order_ids):
            y, x = divmod(idx, 8)
            expected = img[y*32:(y+1)*32, x*32:(x+1)*32]
            rotated = MODES[i%4:] + MODES[:i%4]
            for mode in rotated:
                t = time.perf_counter_ns()
                if mode == "bh60_miss32":
                    restored = oldreader.read(y,x)
                elif mode == "bh64_fixed":
                    restored = faster.read(y,x)
                elif mode == "tiff_zstd":
                    restored = tifz.read(1,window=Window(x*32,y*32,32,32))
                else:
                    restored = tifl.read(1,window=Window(x*32,y*32,32,32))
                dt_ms = (time.perf_counter_ns()-t)/1e6
                if not np.array_equal(restored, expected):
                    raise AssertionError("Lossless cache-miss ROI validation failed")
                samples[mode].append(dt_ms)
        if oldreader.hits != 0 or oldreader.misses != 192:
            raise AssertionError("Legacy 32-block LRU did not actually thrash")
    return {
        "region": region, "pixels": [256,256], "block_size": 32,
        "roi_count":192, "legacy_bh60_cache_misses":192,
        "lossless_verified":True, "max_height_error_m":0,
        "source_sha256":hashlib.sha256(img.tobytes()).hexdigest(),
        "metadata":info,
        "size_bytes":{key:path.stat().st_size for key,path in output.items()},
        "file_sha256":{key:hashlib.sha256(path.read_bytes()).hexdigest()
                       for key,path in output.items()},
        "file":{key:path.name for key,path in output.items()},
        "encode_median_seconds":{k:float(np.median(v)) for k,v in elapsed.items()},
        "roi_mean_ms":{k:float(np.mean(v)) for k,v in samples.items()},
        "roi_median_ms":{k:float(np.median(v)) for k,v in samples.items()},
        "roi_p95_ms":{k:float(np.percentile(v,95)) for k,v in samples.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", default="/tmp/blkh_exp64")
    parser.add_argument("--cache", default="/tmp/blkh_exp64_hgt_cache")
    parser.add_argument("--regions", default=",".join(REGIONS))
    args = parser.parse_args()
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    if args.self_test:
        self_test(root)
        return
    names=[x.strip() for x in args.regions.split(",") if x.strip()]
    if not names or len(set(names)) != len(names) or any(x not in REGIONS for x in names):
        parser.error("Use unique preregistered real HGT region IDs")
    results=[]
    for region in names:
        source,provenance=(read_holdout(region,Path(args.cache))
                           if region in MORE_REGIONS else source_hgt(region,Path(args.cache)))
        img=dict(source_cases(source))["single_256x256"]
        result=benchmark_one(region,img,root)
        result["provenance"]=provenance
        results.append(result)
        print(region, "bytes", result["size_bytes"],
              "mean_ms",{k:round(v,4) for k,v in result["roi_mean_ms"].items()},
              "encode_s",{k:round(v,4) for k,v in result["encode_median_seconds"].items()},
              flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":64,
            "status":"provisional until CI verifies decoded bytes",
            "objective":"faster uncached ROI by fixed-width reversible deltas",
            "limitations":"GDAL caching not controlled; CI hardware shared; compression vs decoding tradeoff",
            "sources":names,"results":results,
        },indent=2),encoding="utf8")
    print("EXP64 DONE, independent real HGT regions",len(results),flush=True)


if __name__ == "__main__":
    main()
