"""
Experiment 57 — FAIR RANDOM-ACCESS lossless DEM archives, real held-out regions.

Hypothesis: a block-indexed reversible stream can offer useful spatial ROI
access with a favorable storage/time trade-off. Compare identically indexed
BH56, reversible JPEG2000, and an adaptive per-block mixture. Unlike Exp56,
the archive truly supports seeking to and decoding ONE block without loading
or decoding the other blocks. All index bytes and per-block headers COUNT.

No novel compression algorithm is claimed. The hybrid selects the smaller of
two known existing codecs. Compare indexed formats against each other, and
report a full-mosaic JPEG2000 reference to expose indexing penalties.

Example:
  python research/universe/experiment_57_random_access_dem.py --self-test
  python research/universe/experiment_57_random_access_dem.py --out /tmp/blkh_exp57
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import time

import numpy as np
from PIL import Image, features

from experiment_56_reversible_dem import (
    CORE, HELDOUT, source_hgt, tile_cases, encode as bh56_encode,
    decode as bh56_decode,
)
from experiment_56c_contiguous_mosaic import six_to_mosaic, pack_image

MAGIC = b"BH57I"
VERSION = 1
HEADER = struct.Struct("<5sBHHHII")
ENTRY = struct.Struct("<BII")
CODEC_BH56 = 0
CODEC_J2K = 1
CODEC_NAMES = ("bh56", "j2k")
MAX_PIXELS = 2_000_000
BLOCK_SIDES = (32, 64, 128)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jpeg2000_block(block: np.ndarray) -> bytes:
    """Exact reversible JPEG2000; store per-block signed-int16 origin."""
    lo = int(block.min())
    offset = (block.astype(np.int32) - lo)
    if offset.max() > 65535:
        raise ValueError("Elevation range too large for 16-bit JPEG2000")
    bio = io.BytesIO()
    Image.fromarray(offset.astype(np.uint16)).save(
        bio, format="JPEG2000", irreversible=False
    )
    result = struct.pack("<i", lo) + bio.getvalue()
    if not np.array_equal(decode_block(CODEC_J2K, result, block.shape), block):
        raise AssertionError("Lossless JPEG2000 encoding failed")
    return result


def decode_block(codec: int, payload: bytes, expected: tuple[int, int]) -> np.ndarray:
    if codec == CODEC_BH56:
        values = bh56_decode(payload)
        if values.shape != (1, *expected):
            raise ValueError("Invalid BH56 block shape")
        return values[0]
    if codec == CODEC_J2K:
        if len(payload) < 5:
            raise ValueError("Truncated reversible JPEG2000")
        origin = struct.unpack("<i", payload[:4])[0]
        with Image.open(io.BytesIO(payload[4:])) as image:
            reconstructed = np.asarray(image).astype(np.int32) + origin
        if reconstructed.shape != expected:
            raise ValueError("Invalid J2K block shape")
        if reconstructed.min() < -32768 or reconstructed.max() > 32767:
            raise ValueError("Decoded elevations exceed signed int16")
        return reconstructed.astype(np.int16)
    raise ValueError("Unrecognized block codec")


def blocks_from_image(img: np.ndarray, side: int) -> list[np.ndarray]:
    if img.ndim != 2 or any(dim % side for dim in img.shape):
        raise ValueError("Image dimensions must be divisible by side")
    h, w = img.shape
    return [
        np.asarray(img[y:y+side, x:x+side], dtype=np.int16)
        for y in range(0, h, side) for x in range(0, w, side)
    ]


def candidate_payloads(img: np.ndarray, side: int) -> list[tuple[bytes, bytes]]:
    out = []
    for block in blocks_from_image(img, side):
        bh = bh56_encode(block[None])
        j2 = jpeg2000_block(block)
        if not np.array_equal(decode_block(CODEC_BH56, bh, block.shape), block):
            raise AssertionError("BH56 encoder failed")
        out.append((bh, j2))
    return out


def archive(img: np.ndarray, side: int, candidates: list[tuple[bytes, bytes]], mode: str) -> bytes:
    if mode not in ("bh56", "j2k", "hybrid"):
        raise ValueError("Unknown archival mode")
    h, w = img.shape
    if side not in BLOCK_SIDES or h % side or w % side:
        raise ValueError("Invalid block geometry")
    if len(candidates) != (h // side) * (w // side):
        raise ValueError("Missing block data")
    payload = bytearray()
    index = bytearray()
    for bh, j2 in candidates:
        if mode == "bh56":
            codec, encoded = CODEC_BH56, bh
        elif mode == "j2k":
            codec, encoded = CODEC_J2K, j2
        else:
            codec, encoded = (CODEC_BH56, bh) if len(bh) <= len(j2) else (CODEC_J2K, j2)
        index.extend(ENTRY.pack(codec, len(payload), len(encoded)))
        payload.extend(encoded)
    header = HEADER.pack(MAGIC, VERSION, h, w, side, len(candidates), len(index))
    return header + index + payload


def parse_header_and_index(header: bytes, idx: bytes, file_size: int) -> tuple[dict, list[tuple[int, int, int]]]:
    if len(header) != HEADER.size:
        raise ValueError("Incomplete archive header")
    magic, version, h, w, side, block_count, index_bytes = HEADER.unpack(header)
    if magic != MAGIC or version != VERSION:
        raise ValueError("Invalid archive magic/version")
    if h < 1 or w < 1 or h*w > MAX_PIXELS or side not in BLOCK_SIDES:
        raise ValueError("Invalid dimensions")
    if h % side or w % side or block_count != (h//side)*(w//side):
        raise ValueError("Invalid tile grid")
    if index_bytes != ENTRY.size * block_count or len(idx) != index_bytes:
        raise ValueError("Invalid index length")
    head_bytes = HEADER.size + index_bytes
    if file_size < head_bytes:
        raise ValueError("Truncated archive")
    entries = []
    end = 0
    for i in range(block_count):
        codec, offset, length = ENTRY.unpack_from(idx, i*ENTRY.size)
        if codec not in (CODEC_BH56, CODEC_J2K) or not length or offset != end:
            raise ValueError("Bad block index/offset")
        end = offset + length
        if end > file_size - head_bytes:
            raise ValueError("Block extends past EOF")
        entries.append((codec, offset, length))
    if end != file_size-head_bytes:
        raise ValueError("Unexpected bytes after archive")
    return {
        "height": h, "width": w, "block_side": side,
        "block_count": block_count,
        "index_bytes": index_bytes,
        "header_bytes": HEADER.size,
        "payload_start": head_bytes,
    }, entries


def decode_all(blob: bytes) -> np.ndarray:
    header = blob[:HEADER.size]
    if len(header) < HEADER.size:
        raise ValueError("Truncated archive")
    index_size = HEADER.unpack(header)[-1]
    if index_size > MAX_PIXELS:
        raise ValueError("Oversized index")
    idx = blob[HEADER.size:HEADER.size+index_size]
    meta, entries = parse_header_and_index(header, idx, len(blob))
    h, w, side = meta["height"], meta["width"], meta["block_side"]
    out = np.empty((h, w), dtype=np.int16)
    for i, (codec, start, length) in enumerate(entries):
        y, x = divmod(i, w//side)
        chunk = blob[meta["payload_start"]+start:meta["payload_start"]+start+length]
        out[y*side:(y+1)*side, x*side:(x+1)*side] = decode_block(
            codec, chunk, (side, side)
        )
    return out


def decode_roi_from_file(path: Path, block_row: int, block_col: int) -> tuple[np.ndarray, dict]:
    """Physical file seek: only header, index and ONE compressed block are read."""
    total_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("Truncated archive")
        index_size = HEADER.unpack(header)[-1]
        if index_size > MAX_PIXELS:
            raise ValueError("Index too large")
        idx = handle.read(index_size)
        meta, entries = parse_header_and_index(header, idx, total_size)
        cols = meta["width"]//meta["block_side"]
        rows = meta["height"]//meta["block_side"]
        if not (0 <= block_row < rows and 0 <= block_col < cols):
            raise IndexError("ROI block out of bounds")
        index = block_row*cols + block_col
        codec, offset, size = entries[index]
        handle.seek(meta["payload_start"] + offset)
        encoded = handle.read(size)
        if len(encoded) != size:
            raise ValueError("Incomplete ROI block")
        decoded = decode_block(
            codec, encoded, (meta["block_side"], meta["block_side"])
        )
        return decoded, {
            "physically_read_bytes": HEADER.size + index_size + size,
            "codec": CODEC_NAMES[codec],
            "block_row": block_row, "block_col": block_col,
        }


def validate_case(img: np.ndarray, side: int, output: Path, name: str, region: str,
                  heldout: bool, roi_repeats: int) -> list[dict]:
    if img.dtype != np.int16:
        raise ValueError("Expected int16 elevation meters")
    candidates = candidate_payloads(img, side)
    reference = img.copy()
    records = []
    for mode in ("bh56", "j2k", "hybrid"):
        encoded = archive(img, side, candidates, mode)
        restored = decode_all(encoded)
        if not np.array_equal(restored, reference):
            raise AssertionError("Full lossless archive roundtrip failed")
        saved = output/f"{region}_{name}_{side}_{mode}.bh57"
        saved.write_bytes(encoded)
        ncols = img.shape[1]//side
        rows = img.shape[0]//side
        coords = [(0,0),(rows//2,ncols//2),(rows-1,ncols-1)]
        bytes_read = []
        timings = []
        backend = []
        for y, x in coords:
            for _ in range(roi_repeats):
                start = time.perf_counter()
                crop, meta = decode_roi_from_file(saved, y, x)
                timings.append(time.perf_counter()-start)
                assert np.array_equal(crop, reference[y*side:(y+1)*side,x*side:(x+1)*side])
                bytes_read.append(meta["physically_read_bytes"])
                backend.append(meta["codec"])
        report = {
            "region": region, "heldout_region": heldout,
            "case": name, "mode": mode, "block_side": side,
            "shape": list(img.shape), "num_blocks": len(candidates),
            "size_bytes": len(encoded),
            "index_and_header_bytes": HEADER.size+len(candidates)*ENTRY.size,
            "roi_physical_read_bytes_mean": float(np.mean(bytes_read)),
            "roi_decode_s_median": float(np.median(timings)),
            "roi_fraction_file_read_mean": float(np.mean(bytes_read)/len(encoded)),
            "roi_codecs_encountered": sorted(set(backend)),
            "encoded_sha256": sha(encoded),
            "input_sha256": sha(reference.tobytes()),
            "lossless_verified": True,
            "zero_max_error_m": True,
            "file": saved.name,
        }
        records.append(report)
        print(
            f"{region} {name} b{side} {mode}: {len(encoded)} B, "
            f"ROI physical bytes {report['roi_physical_read_bytes_mean']:.0f}, "
            f"ROI median {report['roi_decode_s_median']*1000:.2f} ms",
            flush=True,
        )
    if records[2]["size_bytes"] > min(records[0]["size_bytes"], records[1]["size_bytes"]):
        raise AssertionError("Hybrid should never exceed same-index best pure codec")
    return records


def self_test() -> None:
    rng = np.random.default_rng(57)
    tests = (
        np.zeros((32,64),dtype=np.int16),
        rng.integers(-32768,32768,size=(64,64),dtype=np.int16),
        (np.indices((64,96)).sum(axis=0)*3-200).astype(np.int16),
    )
    for original in tests:
        for side in (32,):
            candidates = candidate_payloads(original,side)
            for mode in ("bh56","j2k","hybrid"):
                data = archive(original,side,candidates,mode)
                assert np.array_equal(decode_all(data),original)
                try:
                    decode_all(data[:-1])
                except (ValueError, OSError):
                    pass
                else:
                    raise AssertionError("Truncated archive accepted")
    print("EXP57 SELF-TEST PASS: indexed exact-decoding, 3 backends, truncation rejected")


def source_cases(hgt: np.ndarray) -> list[tuple[str, np.ndarray]]:
    originals = dict(tile_cases(hgt))
    return [
        ("mosaic_128x192",six_to_mosaic(originals["six_64x64"])[0]),
        ("single_256x256",originals["one_256x256"][0]),
    ]


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--out",default="/tmp/blkh_exp57")
    ap.add_argument("--cache",default="/tmp/blkh_exp56_skadi_cache")
    ap.add_argument("--roi-repeats",type=int,default=3)
    args=ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not features.check("jpg_2000"):
        ap.error("JPEG2000 support required to make comparisons meaningful")
    if not 1<=args.roi_repeats<=20:
        ap.error("Invalid ROI timing sample count")
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    records=[]
    full_baselines=[]
    for region in CORE+HELDOUT:
        source,provenance=source_hgt(region,Path(args.cache))
        for name,img in source_cases(source):
            target=root/region/name
            target.mkdir(parents=True,exist_ok=True)
            jpeg_full,decoded_full=pack_image(img[None],"JPEG2000")
            assert np.array_equal(decoded_full[0],img)
            full_baselines.append({
                "region":region,"case":name,"jpeg2000_full_image_bytes":len(jpeg_full),
                "input_sha256":sha(img.tobytes()),
                "source":provenance,
            })
            for side in BLOCK_SIDES:
                if any(axis%side for axis in img.shape):
                    continue
                records.extend(validate_case(
                    img,side,target,name,region,
                    region in HELDOUT,args.roi_repeats
                ))
        (root/"results.json").write_text(json.dumps({
            "experiment":57,
            "status":"completed only after verified CI pass",
            "scientific_claim":"indexed random-access overhead comparison, not a new codec",
            "physical_read_note":"one ROI request opens file and reads complete index plus only target compressed block; no full-map read",
            "timing_note":"in-process on shared CI runner, medians not production SLAs",
            "georeferencing":"external and excluded from all encoded variants",
            "records":records,"full_jpeg2000_baselines":full_baselines,
        },indent=2),encoding="utf8")
    assert len(records)==4*5*3, len(records)
    print("EXP57 DONE: exact indexed ROI archives: ",len(records),flush=True)


if __name__=="__main__":
    main()
