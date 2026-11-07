"""
Experiment 59 — preregistered geographic holdout for the Exp58 32px hypothesis.

BEFORE examining source pixels, fix independent HGT tile IDs:
  N39W105  Colorado, USA
  S20W044  Minas Gerais, Brazil

Lossless 32x32 random-access benchmark vs IDENTICALLY INDEXED reversible J2K
and old BH56, for the same mosaic_128x192 and single_256x256 source areas.
Queries: 24 preselected distinct ROI blocks per image, five repetitions,
rotated codec order. Measure file-seek + block decode medians and p95s.
No tuning of codecs on holdout data, all actual file bytes counted.

First verify the remote HGT is valid, NoData sentinel absent in the fixed crop,
then generate actual source SHA256, encoded SHA256 and detailed JSON evidence.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from experiment_57_random_access_dem import (
    archive as old_archive, candidate_payloads, decode_all as old_decode_all,
    decode_roi_from_file as old_roi, source_cases,
)
from experiment_58_vectorized_decoder import (
    build_indexed, decode_indexed, roi_from_file
)

REGIONS = ("N39W105", "S20W044")  # fixed before HGT source inspection
BLOCK_SIDE = 32
DIM = 3601
URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat}/{tile}.hgt.gz"


def read_holdout(region: str, cache: Path) -> tuple[np.ndarray, dict]:
    if region not in REGIONS:
        raise ValueError("Only preregistered real HGT regions")
    cache.mkdir(parents=True, exist_ok=True)
    url = URL.format(lat=region[:3], tile=region)
    dst = cache/f"{region}.hgt.gz"
    if dst.is_file():
        compressed = dst.read_bytes()
    else:
        req = urllib.request.Request(url, headers={"User-Agent":"BLKH-Exp59-Research/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            compressed = response.read(32*1024*1024+1)
        if len(compressed)>32*1024*1024:
            raise ValueError("Unreasonably large HGT download")
        dst.write_bytes(compressed)
    raw = gzip.decompress(compressed)
    if len(raw)!=DIM*DIM*2:
        raise ValueError(f"Invalid HGT dimensions: {len(raw)}")
    arr = np.frombuffer(raw, dtype=">i2").reshape(DIM,DIM)
    return arr,{
        "url":url,"compressed_sha256":hashlib.sha256(compressed).hexdigest(),
        "uncompressed_sha256":hashlib.sha256(raw).hexdigest(),
        "data_type":"SRTM-derived Skadi HGT int16 meters",
    }


def benchmark_case(
    region: str, case: str, image: np.ndarray, target: Path,
    positions: int, repeats: int,
) -> dict:
    if image.dtype!=np.int16 or image.ndim!=2:
        raise ValueError("Expected native signed-int16 geographic surface")
    h,w=image.shape
    if h%BLOCK_SIDE or w%BLOCK_SIDE:
        raise ValueError("Invalid 32px grid")
    blocks=candidate_payloads(image,BLOCK_SIDE)
    payloads={
        "bh56":old_archive(image,BLOCK_SIDE,blocks,"bh56"),
        "j2k":old_archive(image,BLOCK_SIDE,blocks,"j2k"),
        "vectorized":build_indexed(image,BLOCK_SIDE),
    }
    paths={}
    for mode,blob in payloads.items():
        path=target/f"{region}_{case}_{mode}.bin"
        path.write_bytes(blob)
        reconstructed=decode_indexed(blob) if mode=="vectorized" else old_decode_all(blob)
        if not np.array_equal(reconstructed,image):
            raise AssertionError("Full lossless decode failed")
        paths[mode]=path
    block_count=(h//BLOCK_SIDE)*(w//BLOCK_SIDE)
    rng=np.random.default_rng(590059)
    npositions=min(positions,block_count)
    ids=sorted(int(v) for v in rng.choice(block_count,size=npositions,replace=False))
    cols=w//BLOCK_SIDE
    times={m:[] for m in payloads}
    bytes_accessed={m:[] for m in payloads}
    modes=["bh56","j2k","vectorized"]
    for block_id in ids:
        y,x=divmod(block_id,cols)
        reference=image[y*BLOCK_SIDE:(y+1)*BLOCK_SIDE,
                        x*BLOCK_SIDE:(x+1)*BLOCK_SIDE]
        for attempt in range(repeats):
            for mode in modes[attempt%3:]+modes[:attempt%3]:
                start=time.perf_counter()
                if mode=="vectorized":
                    cropped,read=roi_from_file(paths[mode],y,x)
                else:
                    cropped,meta=old_roi(paths[mode],y,x)
                    read=meta["physically_read_bytes"]
                elapsed=time.perf_counter()-start
                if not np.array_equal(cropped,reference):
                    raise AssertionError("ROI differs in one or more meters")
                times[mode].append(elapsed)
                bytes_accessed[mode].append(read)
    return {
        "region":region,"case":case,"image_shape":list(image.shape),
        "block_side":BLOCK_SIDE,"roi_positions_tested":len(ids),
        "roi_repeats_per_position":repeats,
        "roi_block_ids":ids,
        "lossless_verified":True,"max_error_m":0,
        "input_sha256":hashlib.sha256(image.tobytes()).hexdigest(),
        "archive_bytes":{m:len(blob) for m,blob in payloads.items()},
        "archive_sha256":{m:hashlib.sha256(blob).hexdigest() for m,blob in payloads.items()},
        "archive_files":{m:paths[m].name for m in paths},
        "roi_median_ms":{m:float(1000*np.median(t)) for m,t in times.items()},
        "roi_p95_ms":{m:float(1000*np.percentile(t,95)) for m,t in times.items()},
        "physical_read_bytes_mean":{m:float(np.mean(v)) for m,v in bytes_accessed.items()},
        "vectorized_vs_j2k_byte_saving_pct":100*(
            1-len(payloads["vectorized"])/len(payloads["j2k"])
        ),
        "vectorized_vs_j2k_median_faster":float(np.median(times["vectorized"])) <
            float(np.median(times["j2k"])),
        "vectorized_vs_j2k_p95_faster":float(np.percentile(times["vectorized"],95)) <
            float(np.percentile(times["j2k"],95)),
        "vectorized_vs_j2k_smaller":len(payloads["vectorized"])<len(payloads["j2k"]),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",default="/tmp/blkh_exp59_holdout")
    ap.add_argument("--cache",default="/tmp/blkh_exp59_hgt_cache")
    ap.add_argument("--roi-positions",type=int,default=24)
    ap.add_argument("--repeats",type=int,default=5)
    args=ap.parse_args()
    if not (4<=args.roi_positions<=64 and 2<=args.repeats<=15):
        ap.error("Invalid sampling size")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    sources={}
    for region in REGIONS:
        raw,meta=read_holdout(region,Path(args.cache))
        sources[region]=meta
        for case,img in source_cases(raw):
            target=out/region/case
            target.mkdir(parents=True,exist_ok=True)
            r=benchmark_case(region,case,img,target,args.roi_positions,args.repeats)
            rows.append(r)
            print(region,case,
                  "size_B",r["archive_bytes"],
                  "median_ms",{k:round(v,3) for k,v in r["roi_median_ms"].items()},
                  "p95_ms",{k:round(v,3) for k,v in r["roi_p95_ms"].items()},
                  flush=True)
        (out/"results.json").write_text(json.dumps({
            "experiment":59,"status":"holdout confirmation only after CI completion",
            "sources_preregistered":list(REGIONS),
            "source_metadata":sources,
            "protocol":"identically indexed 32x32 int16 DEM, exact decode, physical seek",
            "measurement_warning":"GitHub shared CPU, 2 correlated shape variants per region; not a field benchmark",
            "notes":"32px selected from earlier 4-region Exp58; no hyperparameter changes on new regions",
            "records":rows,
        },indent=2),encoding="utf8")
    assert len(rows)==4
    print("EXP59 DONE: 2 NEW GEOGRAPHIC REGIONS, 4 CASES, EXACT DECODE",flush=True)


if __name__=="__main__":
    main()
