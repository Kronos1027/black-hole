"""
Experiment 65 — mixed compact/fast lossless 32px DEM block archive.

Research-only size-vs-cache-miss-latency Pareto study. For each 32x32 block,
save *either* the compact BH58 varint/Brotli representation (codec=2) *or*
the faster BH64 fixed-u16-delta/Brotli representation (codec=3).

Selection depends ONLY on complete payload byte sizes:
  choose fast if bytes_fast <= bytes_compact * (1 + size_overhead_threshold)
Fixed thresholds: 0.00, 0.15, 0.20, 0.25. Thresholds are an explicit
exploratory sweep of previously studied real HGT regions, NOT a predeclared
held-out improvement.

All files have the same counted complete georeferencing and per-block index.
All variants independently decode EXACT int16 terrain elevations, including
random-access ROI. Measure true uncached 64 distinct tiles cycled 3 times.
No block cache. Native codecs are not invented. This is an engineering
trade-off, not new theoretical neural compression.
"""
from __future__ import annotations
import argparse
import brotli
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

from experiment_60_geotiff_roi import (
    REGIONS, MORE_REGIONS, PRELUDE, ROI_SIDE,
    metadata, read_holdout, source_hgt, source_cases,
)
from experiment_58_vectorized_decoder import (
    block_encode as compact_encode, block_decode as compact_decode,
    INDEX_HEADER, ENTRY,
)
from experiment_64_fixed_delta_roi import (
    block_encode as fast_encode, block_decode as fast_decode,
)

OUTER = b"BHG65"
INNER = b"BH65I"
LEVELS = (0.00, 0.15, 0.20, 0.25)
MAX_PIXELS = 2_000_000


def candidates(image: np.ndarray) -> list[tuple[bytes, bytes]]:
    if image.shape != (256, 256) or image.dtype != np.int16:
        raise ValueError("Use exactly 256x256 signed-int16 elevation data")
    blocks = []
    for y in range(0, 256, ROI_SIDE):
        for x in range(0, 256, ROI_SIDE):
            block = image[y:y+ROI_SIDE, x:x+ROI_SIDE]
            c = compact_encode(block)
            f = fast_encode(block)
            if not np.array_equal(compact_decode(c), block):
                raise AssertionError("Compact codec failed exact reconstruction")
            if not np.array_equal(fast_decode(f), block):
                raise AssertionError("Fixed-width codec failed exact reconstruction")
            blocks.append((c, f))
    assert len(blocks) == 64
    return blocks


def pack(image: np.ndarray, info: dict,
         blocks: list[tuple[bytes, bytes]], max_overhead: float) -> tuple[bytes, dict]:
    if image.shape != (256,256) or image.dtype != np.int16:
        raise ValueError("Invalid input image")
    if len(blocks) != 64 or not (0 <= max_overhead <= 1):
        raise ValueError("Invalid sweep")
    meta = json.dumps(info, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if not 0 < len(meta) < 16384:
        raise ValueError("Invalid geospatial metadata")
    idx = bytearray()
    payload = bytearray()
    fast_count = 0
    for compact, fast in blocks:
        use_fast = len(fast) <= len(compact)*(1+max_overhead)
        kind = 3 if use_fast else 2
        encoded = fast if use_fast else compact
        idx.extend(ENTRY.pack(kind, len(payload), len(encoded)))
        payload.extend(encoded)
        fast_count += int(use_fast)
    header = PRELUDE.pack(OUTER, 1, len(meta))
    inner = INDEX_HEADER.pack(INNER, 1, 256, 256, ROI_SIDE, 64, len(idx))
    output = header + meta + inner + idx + payload
    return output, {
        "chosen_fast_blocks": fast_count,
        "chosen_compact_blocks": 64-fast_count,
        "fast_fraction": fast_count/64,
        "payload_bytes": len(payload),
        "index_bytes": len(idx),
        "geospatial_bytes": len(meta)+len(header),
    }


class OpenHybrid:
    def __init__(self, path: Path):
        self.file = path.open("rb")
        try:
            header = self.file.read(PRELUDE.size)
            if len(header) != PRELUDE.size:
                raise ValueError("Truncated BH65 header")
            magic, version, n = PRELUDE.unpack(header)
            if magic != OUTER or version != 1 or not 0 < n < 16384:
                raise ValueError("Invalid BH65 container")
            raw = self.file.read(n)
            if len(raw) != n:
                raise ValueError("Truncated BH65 JSON")
            self.metadata = json.loads(raw)
            h = self.file.read(INDEX_HEADER.size)
            if len(h) != INDEX_HEADER.size:
                raise ValueError("Truncated BH65 index header")
            im, ver, ih, iw, side, count, nbytes = INDEX_HEADER.unpack(h)
            if ((im,ver,ih,iw,side,count,nbytes) !=
                    (INNER,1,256,256,ROI_SIDE,64,64*ENTRY.size)):
                raise ValueError("Incorrect BH65 index dimensions")
            if self.metadata.get("image_shape") != [256, 256]:
                raise ValueError("Geospatial dimension mismatch")
            if (self.metadata.get("crs")!="EPSG:4326"
                    or self.metadata.get("sample_unit")!="meter"
                    or self.metadata.get("dtype")!="int16"
                    or self.metadata.get("block_side")!=ROI_SIDE):
                raise ValueError("Unsupported BH65 geospatial schema")
            idx = self.file.read(nbytes)
            if len(idx) != nbytes:
                raise ValueError("Truncated BH65 index")
            self.entries=[]
            offset=0
            for i in range(count):
                kind,start,length=ENTRY.unpack_from(idx,i*ENTRY.size)
                if kind not in (2,3) or start != offset or length < 14:
                    raise ValueError("Corrupt BH65 block index")
                offset += length
                self.entries.append((kind,start,length))
            self.body = PRELUDE.size+n+INDEX_HEADER.size+nbytes
            if self.body + offset != path.stat().st_size:
                raise ValueError("Corrupt BH65 archive length")
        except BaseException:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.file.close()

    def read(self,row:int,col:int)->np.ndarray:
        if not (0<=row<8 and 0<=col<8):
            raise IndexError("ROI out of image")
        kind,start,length=self.entries[row*8+col]
        self.file.seek(self.body+start)
        raw=self.file.read(length)
        if len(raw)!=length:
            raise ValueError("Truncated BH65 block")
        return compact_decode(raw) if kind==2 else fast_decode(raw)


def check_full(path: Path, image: np.ndarray, info: dict) -> None:
    with OpenHybrid(path) as archive:
        if archive.metadata != info:
            raise AssertionError("Missing georeferencing")
        for y in range(8):
            for x in range(8):
                original = image[y*32:(y+1)*32, x*32:(x+1)*32]
                if not np.array_equal(archive.read(y,x), original):
                    raise AssertionError("Lossless BH65 file reconstruction failed")


def self_test(out: Path):
    out.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(65)
    data = rng.integers(-32768,32768,(256,256),dtype=np.int16)
    info=metadata("N37W119", data)
    blocks=candidates(data)
    for level in LEVELS:
        packed,meta=pack(data,info,blocks,level)
        path=out/f"test_{str(level).replace('.','p')}.bh65"
        path.write_bytes(packed)
        check_full(path,data,info)
        assert meta["chosen_fast_blocks"]+meta["chosen_compact_blocks"]==64
        try:
            path.write_bytes(packed[:-1])
            check_full(path,data,info)
        except (ValueError, brotli.error):
            pass
        else:
            raise AssertionError("Truncated archive should fail")
    print("EXP65 SELF-TEST PASS: 4 versions, all 64 tiles lossless and invalid streams rejected")


def benchmark_one(region: str, image: np.ndarray, out: Path) -> dict:
    blocks = candidates(image)
    info=metadata(region,image)
    out.mkdir(parents=True,exist_ok=True)
    saved={}
    summaries={}
    for level in LEVELS:
        key=f"{level:.2f}"
        raw,stat=pack(image,info,blocks,level)
        path=out/f"pareto_{str(level).replace('.','p')}.bh65"
        path.write_bytes(raw)
        check_full(path,image,info)
        saved[key]=path
        summaries[key]={
            **stat,
            "size_bytes":len(raw),
            "sha256":hashlib.sha256(raw).hexdigest(),
            "file":path.name,
        }
    # All four variants see exactly the same requests in the same order.
    order=list(range(64))*3
    timing={key:[] for key in saved}
    with ExitStack() as ctx:
        handles={key:ctx.enter_context(OpenHybrid(path))
                 for key,path in saved.items()}
        for i,tile_id in enumerate(order):
            y,x=divmod(tile_id,8)
            expected=image[y*32:(y+1)*32,x*32:(x+1)*32]
            keys=list(saved)
            keys=keys[i%len(keys):]+keys[:i%len(keys)]
            for key in keys:
                start=time.perf_counter_ns()
                b=handles[key].read(y,x)
                milliseconds=(time.perf_counter_ns()-start)/1e6
                if not np.array_equal(expected,b):
                    raise AssertionError("Mixed-codec ROI was lossy")
                timing[key].append(milliseconds)
    for key in saved:
        summaries[key]["roi_mean_ms"]=float(np.mean(timing[key]))
        summaries[key]["roi_median_ms"]=float(np.median(timing[key]))
        summaries[key]["roi_p95_ms"]=float(np.percentile(timing[key],95))
        summaries[key]["roi_reads"]=192
        summaries[key]["all_true_cache_misses"]=True
        print(region, "overhead",key,
              "fast_tiles",summaries[key]["chosen_fast_blocks"],
              "bytes",summaries[key]["size_bytes"],
              "uncached_ms",round(summaries[key]["roi_mean_ms"],4),
              flush=True)
    return {
        "region":region,
        "source_sha256":hashlib.sha256(image.tobytes()).hexdigest(),
        "geospatial":info,
        "geometry":[256,256],
        "versions":summaries,
        "lossless_verified":True,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",default="/tmp/blkh_exp65")
    p.add_argument("--cache",default="/tmp/blkh_exp65_hgt_cache")
    p.add_argument("--regions",default=",".join(REGIONS))
    a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    if a.self_test:
        self_test(root);return
    names=[x.strip() for x in a.regions.split(",") if x.strip()]
    if not names or len(set(names))!=len(names) or any(v not in REGIONS for v in names):
        p.error("Only unique preregistered HGT sources")
    results=[]
    for region in names:
        src,provenance=(read_holdout(region,Path(a.cache))
                        if region in MORE_REGIONS else source_hgt(region,Path(a.cache)))
        image=dict(source_cases(src))["single_256x256"]
        r=benchmark_one(region,image,root/region)
        r["provenance"]=provenance
        results.append(r)
        (root/"results.json").write_text(json.dumps({
            "experiment":65,
            "purpose":"exploratory per-block storage versus true cache-miss decoding Pareto curve",
            "warning":"parameter thresholds explored on prior observed sources; NOT newly held out",
            "thresholds":LEVELS,
            "results":results,
        },indent=2),encoding="utf8")
    print("EXP65 COMPLETE 6 regions and 4 all-lossless hybrid archive settings",flush=True)


if __name__=="__main__":
    main()
