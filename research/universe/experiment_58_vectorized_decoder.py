"""
Experiment 58 — vectorized reversible elevation predictor with indexed ROI.

Research hypothesis: BH56 saves bytes against individually indexed J2K small
tiles, but its Python pixel loop is slower. A row-delta + vectorized ZigZag/
LEB128 encoder/decoder may preserve much of that rate advantage with far lower
ROI latency. Horizontal/vertical predictors chosen by actual full byte count.

Novelty: engineering trade-off only; all primitives are established.
Benchmark using SAME 20 terrain/geometry/block configurations and identical
archive index overhead against BH56 and J2K. Lossless physical meter units.
"""
from __future__ import annotations

import argparse
import brotli
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

from experiment_56_reversible_dem import CORE, HELDOUT, source_hgt
from experiment_57_random_access_dem import (
    HEADER as INDEX_HEADER, ENTRY, candidate_payloads,
    archive as old_archive, decode_all as old_decode_all,
    decode_roi_from_file as old_decode_roi, source_cases,
    BLOCK_SIDES, blocks_from_image,
)

MAGIC = b"BHV58"
HMAGIC = b"BH58I"
PART = struct.Struct("<5sBHHI")
MAX_PIXELS = 2_000_000
MODE_NAMES = ("horizontal", "vertical")


def vector_varint_pack(zig: np.ndarray) -> bytes:
    """LEB128 encode all ZigZag uint16 residuals using NumPy index scatter."""
    flat = np.asarray(zig, dtype=np.uint32).reshape(-1)
    sizes = np.where(flat < 128, 1, np.where(flat < 16384, 2, 3))
    positions = np.cumsum(sizes, dtype=np.int64) - sizes
    buf = np.empty(int(sizes.sum()), dtype=np.uint8)
    buf[positions] = np.where(sizes == 1, flat, (flat & 127) | 128).astype(np.uint8)
    two = sizes >= 2
    buf[positions[two] + 1] = np.where(
        sizes[two] == 2, flat[two] >> 7, ((flat[two] >> 7) & 127) | 128
    ).astype(np.uint8)
    three = sizes == 3
    buf[positions[three] + 2] = (flat[three] >> 14).astype(np.uint8)
    return buf.tobytes()


def vector_varint_unpack(payload: bytes, pixels: int) -> np.ndarray:
    raw = np.frombuffer(payload, dtype=np.uint8)
    if len(raw) == 0 or len(raw) > pixels * 3 or (raw[-1] & 128):
        raise ValueError("Invalid variable-length stream")
    starts = np.concatenate((
        np.array([0], dtype=np.int64),
        np.flatnonzero((raw[:-1] & 128) == 0).astype(np.int64) + 1,
    ))
    if len(starts) != pixels:
        raise ValueError("Wrong number of decoded residuals")
    widths = np.diff(np.append(starts, len(raw)))
    if np.any(widths < 1) or np.any(widths > 3):
        raise ValueError("Unrecognized varint width")
    # Verify each continuation bit, and ban non-canonical overlong forms.
    if np.any((raw[starts[widths > 1]] & 128) == 0):
        raise ValueError("Malformed multibyte integer")
    values = (raw[starts] & 127).astype(np.uint32)
    two = widths >= 2
    if np.any((raw[starts[two] + 1] & 128) != (widths[two] == 3).astype(np.uint8) * 128):
        raise ValueError("Malformed second varint byte")
    values[two] |= (raw[starts[two] + 1].astype(np.uint32) & 127) << 7
    three = widths == 3
    values[three] |= raw[starts[three] + 2].astype(np.uint32) << 14
    if np.any(values > 65535):
        raise ValueError("Invalid uint16 ZigZag residual")
    if np.any((widths == 2) & (values < 128)) or np.any((widths == 3) & (values < 16384)):
        raise ValueError("Noncanonical varint")
    return values


def delta_residual(heights: np.ndarray, mode: int) -> np.ndarray:
    raw = np.asarray(heights, dtype=np.int16)
    if raw.ndim != 2 or raw.size > MAX_PIXELS:
        raise ValueError("Expected 2D signed 16-bit elevations")
    u = raw.view(np.uint16)
    delta = np.empty_like(u)
    if mode == 0:
        delta[:, 0] = u[:, 0]
        delta[:, 1:] = u[:, 1:] - u[:, :-1]
    elif mode == 1:
        delta[0, :] = u[0, :]
        delta[1:, :] = u[1:, :] - u[:-1, :]
    else:
        raise ValueError("Invalid predictor mode")
    return delta.view(np.int16).astype(np.int32)


def block_encode(block: np.ndarray, quality: int = 8) -> bytes:
    data = np.asarray(block, dtype=np.int16)
    if data.ndim != 2 or any(v < 1 or v > 4096 for v in data.shape):
        raise ValueError("Invalid block dimensions")
    if not 0 <= quality <= 11:
        raise ValueError("Bad Brotli quality")
    options = []
    for mode in (0, 1):
        delta = delta_residual(data, mode)
        zig = np.where(delta >= 0, delta * 2, -delta * 2 - 1).astype(np.uint32)
        packed = vector_varint_pack(zig)
        payload = brotli.compress(packed, quality=quality)
        options.append(PART.pack(MAGIC, mode, *data.shape, len(payload)) + payload)
    return min(options, key=len)


def block_decode(data: bytes) -> np.ndarray:
    if len(data) < PART.size:
        raise ValueError("Truncated BH58 block header")
    magic, mode, h, w, compressed_size = PART.unpack(data[:PART.size])
    if (magic != MAGIC or mode not in (0,1) or
        any(v < 1 or v > 4096 for v in (h,w)) or
        h*w > MAX_PIXELS or compressed_size != len(data) - PART.size):
        raise ValueError("Invalid BH58 block metadata")
    decoder = brotli.Decompressor()
    buf = bytearray()
    for pos in range(PART.size, len(data), 4096):
        buf.extend(decoder.process(data[pos:pos+4096]))
        if len(buf) > h*w*3:
            raise ValueError("Decompressed block too large")
    if not decoder.is_finished():
        raise ValueError("Incomplete Brotli payload")
    residual_zig = vector_varint_unpack(bytes(buf), h*w)
    signed = ((residual_zig >> 1).astype(np.int32) ^
              -(residual_zig & 1).astype(np.int32)).reshape(h,w)
    if mode == 0:
        unsigned = np.cumsum(signed, axis=1, dtype=np.int32)
    else:
        unsigned = np.cumsum(signed, axis=0, dtype=np.int32)
    return unsigned.astype(np.uint16).view(np.int16)


def build_indexed(img: np.ndarray, side: int) -> bytes:
    blocks = blocks_from_image(img, side)
    data = bytearray()
    idx = bytearray()
    for block in blocks:
        coded = block_encode(block)
        idx.extend(ENTRY.pack(2, len(data), len(coded)))
        data.extend(coded)
    h,w=img.shape
    return INDEX_HEADER.pack(HMAGIC, 1, h, w, side, len(blocks), len(idx)) + idx + data


def read_index(head: bytes, rawindex: bytes, file_length: int):
    if len(head) != INDEX_HEADER.size:
        raise ValueError("Truncated archive")
    magic, version, h,w,side,num,size=INDEX_HEADER.unpack(head)
    if (magic!=HMAGIC or version!=1 or side not in BLOCK_SIDES or
        h<side or w<side or h*w>MAX_PIXELS or h%side or w%side or
        num!=(h//side)*(w//side) or size!=num*ENTRY.size or
        len(rawindex)!=size):
        raise ValueError("Invalid BH58 indexed header")
    entries=[]
    offset=0
    for i in range(num):
        codec, start, length=ENTRY.unpack_from(rawindex,i*ENTRY.size)
        if codec!=2 or length<PART.size or start!=offset:
            raise ValueError("Corrupt BH58 index")
        offset+=length
        entries.append((start,length))
    if INDEX_HEADER.size+size+offset != file_length:
        raise ValueError("Truncated or trailing archive data")
    return (h,w,side,entries)


def decode_indexed(blob: bytes) -> np.ndarray:
    head=blob[:INDEX_HEADER.size]
    if len(head)!=INDEX_HEADER.size:
        raise ValueError("Truncated header")
    count=INDEX_HEADER.unpack(head)[-1]
    if count>MAX_PIXELS:
        raise ValueError("Oversized index")
    idx=blob[INDEX_HEADER.size:INDEX_HEADER.size+count]
    h,w,side,entries=read_index(head,idx,len(blob))
    result=np.empty((h,w),dtype=np.int16)
    start=INDEX_HEADER.size+count
    for i,(offset,length) in enumerate(entries):
        y,x=divmod(i,w//side)
        decoded=block_decode(blob[start+offset:start+offset+length])
        if decoded.shape!=(side,side):
            raise ValueError("Mismatched block dimensions")
        result[y*side:(y+1)*side,x*side:(x+1)*side]=decoded
    return result


def roi_from_file(path: Path, row: int, col: int) -> tuple[np.ndarray, int]:
    size=path.stat().st_size
    with path.open("rb") as f:
        header=f.read(INDEX_HEADER.size)
        if len(header)!=INDEX_HEADER.size:
            raise ValueError("Truncated header")
        index_size=INDEX_HEADER.unpack(header)[-1]
        if index_size>MAX_PIXELS:
            raise ValueError("Oversized index")
        index=f.read(index_size)
        h,w,side,entries=read_index(header,index,size)
        if not (0<=row<h//side and 0<=col<w//side):
            raise IndexError("Out of bounds")
        offset,length=entries[row*(w//side)+col]
        f.seek(INDEX_HEADER.size+index_size+offset)
        blob=f.read(length)
    if len(blob)!=length:
        raise ValueError("Incomplete physical read")
    result=block_decode(blob)
    if result.shape!=(side,side):
        raise ValueError("Bad decoded geometry")
    return result, INDEX_HEADER.size+index_size+length


def self_test():
    rng=np.random.default_rng(58)
    cases=[
        np.zeros((32,32),np.int16),
        np.array([[-32768,32767],[32767,-32768]],np.int16),
        rng.integers(-32768,32768,(64,64),dtype=np.int16),
        (np.indices((64,96)).sum(axis=0)*3-100).astype(np.int16)
    ]
    for img in cases:
        for mode in (0,1):
            values=delta_residual(img,mode)
            zig=np.where(values>=0,values*2,-values*2-1).astype(np.uint32)
            encoded=vector_varint_pack(zig)
            assert np.array_equal(vector_varint_unpack(encoded,img.size),zig.ravel())
        blob=block_encode(img)
        assert np.array_equal(block_decode(blob),img)
        try:
            block_decode(blob[:-1])
        except (ValueError,brotli.error):
            pass
        else:
            raise AssertionError("Corrupt block accepted")
    sample=cases[2]
    blob=build_indexed(sample,32)
    assert np.array_equal(decode_indexed(blob),sample)
    print("EXP58 SELF TEST PASS: modular uint16 + vectorized variable-length decoding",flush=True)


def benchmark(root: Path, cache: Path):
    results=[]
    for region in CORE+HELDOUT:
        source,provenance=source_hgt(region,cache)
        for case,img in source_cases(source):
            h,w=img.shape
            for side in BLOCK_SIDES:
                if h%side or w%side: continue
                out=root/region/case
                out.mkdir(parents=True,exist_ok=True)
                candidates=candidate_payloads(img,side)
                old_bytes={}
                old_roi={}
                for method in ("bh56","j2k"):
                    blob=old_archive(img,side,candidates,method)
                    p=out/f"{case}_{side}_{method}.bh57"
                    p.write_bytes(blob)
                    assert np.array_equal(old_decode_all(blob),img)
                    old_bytes[method]=len(blob)
                    timed=[]
                    for y,x in ((0,0),(h//side-1,w//side-1)):
                        for _ in range(3):
                            start=time.perf_counter()
                            block,meta=old_decode_roi(p,y,x)
                            timed.append(time.perf_counter()-start)
                            assert np.array_equal(block,img[y*side:(y+1)*side,x*side:(x+1)*side])
                    old_roi[method]=float(np.median(timed))
                new=build_indexed(img,side)
                p=out/f"{case}_{side}_vectorized.bh58"
                p.write_bytes(new)
                assert np.array_equal(decode_indexed(new),img)
                timings=[]
                accessed=[]
                for y,x in ((0,0),(h//side-1,w//side-1)):
                    for _ in range(3):
                        start=time.perf_counter()
                        block,read=roi_from_file(p,y,x)
                        timings.append(time.perf_counter()-start)
                        accessed.append(read)
                        assert np.array_equal(block,img[y*side:(y+1)*side,x*side:(x+1)*side])
                sizes={**old_bytes,"vectorized":len(new)}
                latencies={**old_roi,"vectorized":float(np.median(timings))}
                row={
                    "region":region,"heldout_region":region in HELDOUT,
                    "case":case,"image_shape":list(img.shape),
                    "block_side":side,
                    "bytes":sizes,"roi_time_seconds_median":latencies,
                    "vectorized_roi_bytes_read_mean":float(np.mean(accessed)),
                    "vectorized_roi_fraction_read":float(np.mean(accessed)/len(new)),
                    "vectorized_sha256":hashlib.sha256(new).hexdigest(),
                    "image_sha256":hashlib.sha256(img.tobytes()).hexdigest(),
                    "lossless_all_modes":True,
                    "source":provenance,
                    "file":str(p.relative_to(root)),
                }
                results.append(row)
                print(region,case,"block",side,"bytes",sizes,
                      "ROI_ms",{k:round(v*1000,3) for k,v in latencies.items()},flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":58,
            "notes":"same indexed geometry, physically seekable, two selected corner ROIs and 3 repeats; shared runner not stable real-world latency",
            "lossless":True,
            "methods":"reversible int16 modular delta, vectorized ZigZag/LEB128, Brotli quality 8",
            "records":results,
        },indent=2),encoding="utf8")
    if len(results)!=20:
        raise AssertionError(f"Expected 20 geometry settings, got {len(results)}")
    print("EXP58 DONE: 20 geometry settings x 3 exactly lossless indexed codecs",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",default="/tmp/blkh_exp58")
    p.add_argument("--cache",default="/tmp/blkh_exp56_skadi_cache")
    a=p.parse_args()
    if a.self_test: self_test()
    else: benchmark(Path(a.out),Path(a.cache))


if __name__=="__main__":
    main()
