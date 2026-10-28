"""
Experiment 61 — WARM, persistently opened DEM ROI benchmarks.

Exp60's BH60 win in open+read+close latency may reflect GDAL dataset-open
overhead. Re-run exactly the same 6 real HGT regions, 2 geometries and
32-pixel tiles but open all 4 files ONCE, warm every ROI position, then
measure seek/read/decode vs GDAL rasterio.Window while each file remains open.

No physical TIFF bytes-read inference; internal GDAL block cache may favor
GeoTIFF, and physical BH60 bytes read include only requested block.
This is a warm-cache microbenchmark, NOT a production latency guarantee.
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
    source_hgt, read_holdout, source_cases, metadata, encode_bh60,
    encode_tiff, read_bh60_meta, read_index, INDEX_HEADER, block_decode
)

N_ROIS = 24


def setup_files(region: str, case: str, img: np.ndarray, root: Path) -> dict:
    out = root/region/case
    out.mkdir(parents=True, exist_ok=True)
    info=metadata(region,img)
    paths = {"bh60":out/"data.bh60"}
    paths["bh60"].write_bytes(encode_bh60(img,info))
    for mode in CODECS:
        if mode == "bh60":
            continue
        codec=mode.split("_",1)[1]
        path=out/(mode+".tif")
        encode_tiff(path,img,info,codec)
        paths[mode]=path
    return paths


def bench_case(region: str, case: str, img: np.ndarray,
               paths: dict, n_repeats: int) -> dict:
    h,w=img.shape
    side=ROI_SIDE
    rng=np.random.default_rng(610061)
    ids=sorted(int(v) for v in rng.choice(
        (h//side)*(w//side),size=N_ROIS,replace=False
    ))
    coords=[divmod(i,w//side) for i in ids]
    obs={key:[] for key in CODECS}
    with ExitStack() as stack:
        bh=stack.enter_context(paths["bh60"].open("rb"))
        info,offset=read_bh60_meta(bh)
        head=bh.read(INDEX_HEADER.size)
        if len(head)!=INDEX_HEADER.size:
            raise ValueError("BH60 truncated inner index header")
        idx_bytes=INDEX_HEADER.unpack(head)[-1]
        if idx_bytes>2_000_000:
            raise ValueError("Untrusted inner index size")
        index=bh.read(idx_bytes)
        hi,wi,si,entries=read_index(
            head,index,paths["bh60"].stat().st_size-offset
        )
        if (hi,wi,si)!=(h,w,side):
            raise ValueError("Index geometry mismatch")
        if info["image_shape"]!=[h,w]:
            raise ValueError("BH60 metadata geometry mismatch")
        base=offset+INDEX_HEADER.size+idx_bytes
        ds={}
        for mode in CODECS:
            if mode!="bh60":
                ds[mode]=stack.enter_context(rasterio.open(paths[mode]))
                if ds[mode].block_shapes[0]!=(side,side):
                    raise ValueError("Incorrect GeoTIFF tile geometry")

        def request(mode: str, y: int, x: int) -> np.ndarray:
            if mode=="bh60":
                start,length=entries[y*(w//side)+x]
                bh.seek(base+start)
                data=bh.read(length)
                if len(data)!=length:
                    raise ValueError("Incomplete BH60 ROI read")
                return block_decode(data)
            return ds[mode].read(1,window=Window(x*side,y*side,side,side))

        # Prime all queried ROI blocks, including GDAL's internal TIFF cache.
        for y,x in coords:
            expected=img[y*side:(y+1)*side,x*side:(x+1)*side]
            for mode in CODECS:
                if not np.array_equal(request(mode,y,x),expected):
                    raise AssertionError("Warmup ROI decode mismatch")
        for rep in range(n_repeats):
            order=CODECS[rep%len(CODECS):]+CODECS[:rep%len(CODECS)]
            for y,x in coords:
                expected=img[y*side:(y+1)*side,x*side:(x+1)*side]
                for mode in order:
                    t=time.perf_counter_ns()
                    actual=request(mode,y,x)
                    dt=(time.perf_counter_ns()-t)/1_000_000
                    if not np.array_equal(actual,expected):
                        raise AssertionError("Warm ROI decode is not exact")
                    obs[mode].append(dt)
    row={
        "region":region,"case":case,
        "geometry":list(img.shape),
        "block_size":side,
        "roi_positions":ids,
        "roi_repeats":n_repeats,
        "samples_per_mode":len(ids)*n_repeats,
        "codec_sizes":{k:paths[k].stat().st_size for k in CODECS},
        "codec_sha256":{k:hashlib.sha256(paths[k].read_bytes()).hexdigest()
                        for k in CODECS},
        "codec_file":{k:paths[k].name for k in CODECS},
        "median_ms":{k:float(np.median(obs[k])) for k in CODECS},
        "p95_ms":{k:float(np.percentile(obs[k],95)) for k in CODECS},
        "mean_ms":{k:float(np.mean(obs[k])) for k in CODECS},
        "lossless_verified":True,
        "max_error_m":0,
        "input_sha256":hashlib.sha256(img.tobytes()).hexdigest(),
    }
    print(region,case,"warm_ms",
          {k:round(v,3) for k,v in row["median_ms"].items()},flush=True)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="/tmp/blkh_exp61")
    p.add_argument("--cache",default="/tmp/blkh_exp61_cache")
    p.add_argument("--repeats",type=int,default=5)
    a=p.parse_args()
    if not 2<=a.repeats<=12:
        p.error("Repetitions must be 2..12")
    root=Path(a.out)
    root.mkdir(parents=True,exist_ok=True)
    records=[]
    for region in REGIONS:
        raw,_ = (read_holdout(region,Path(a.cache))
                 if region in MORE_REGIONS else source_hgt(region,Path(a.cache)))
        for case,img in source_cases(raw):
            paths=setup_files(region,case,img,root)
            records.append(bench_case(region,case,img,paths,a.repeats))
        (root/"results.json").write_text(json.dumps({
            "experiment":61,
            "status":"complete only after CI validation",
            "purpose":"fix GeoTIFF repeated-open bias of Exp60",
            "regions":list(REGIONS),
            "methodology":"persistently opened BH60 file and rasterio GeoTIFF datasets; all selected ROI blocks warmed before timing",
            "caveats":"GDAL may cache decompressed blocks, BH60 reader does not; shared runner is not dedicated CPU benchmark",
            "records":records,
        },indent=2),encoding="utf8")
    assert len(records)==12
    print("EXP61: 12 WARM-HANDLE GEOGRAPHIC ROI COMPARISONS COMPLETED",flush=True)


if __name__=="__main__":
    main()
