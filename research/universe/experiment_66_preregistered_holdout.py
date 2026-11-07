"""
Experiment 66 — preregistered, frozen BH65 hybrid versus reversible J2K/GeoTIFF.

Protocol was COMMITTED before source data inspection:
    research/universe/BHUH_EXP66_PREREGISTERED_PROTOCOL.md

Four NEW HGT grid IDs, 256x256 crop at rows 1200:1456, columns 1300:1556.
Fixed 32x32 blocks, BH65 tolerance=0.20, as selected in the PRIOR experiment.
Reference arms: BH65 compact (0.00), BH64 fixed width, equally indexed
reversible JPEG2000 (same geospatial JSON wrapper, index entry width/block
geometry), GDAL tiled DEFLATE and ZSTD (full CRS/nodata metadata).
Include standalone full-image reversible JPEG2000 size as a non-ROI control.
All elevations are signed int16 physical meters, exactly decoded.

No tuning to outcomes, no substituting source regions, no novel codec claims.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request

import numpy as np
import rasterio
from rasterio.windows import Window

from experiment_60_geotiff_roi import (
    metadata, encode_tiff, verify_tiff, PRELUDE, ROI_SIDE,
)
from experiment_65_hybrid_pareto import (
    candidates as bh_candidates, pack as bh_pack,
    OpenHybrid, check_full,
)
from experiment_64_fixed_delta_roi import (
    encode_image as fixed_encode, decode_image as fixed_decode,
    OpenBH64,
)
from experiment_57_random_access_dem import (
    jpeg2000_block, archive as j2k_archive,
    decode_block as j2k_decode_block, decode_all as j2k_decode_all,
    parse_header_and_index, HEADER as J2K_HEADER,
)
from experiment_56c_contiguous_mosaic import pack_image

REGIONS = ("N39W107", "N46E011", "N35E138", "S22W044")
FIXED_TOLERANCE = 0.20
HGT_DIM = 3601
PIXELS = 256
ORIGIN = (1200, 1300)
MAGIC_J2KGEO = b"J2G66"
CODECS = ("bh65_fixed20", "bh65_compact", "bh64_fixed", "j2k_indexed",
          "tiff_zstd", "tiff_deflate")
ROI_READS = list(range(64))*3
TRIALS = 3


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_hgt(region: str, cache: Path) -> tuple[np.ndarray, dict]:
    if region not in REGIONS:
        raise ValueError("Unregistered HGT region")
    cache.mkdir(parents=True, exist_ok=True)
    url = ("https://s3.amazonaws.com/elevation-tiles-prod/skadi/"
           f"{region[:3]}/{region}.hgt.gz")
    path = cache/f"{region}.hgt.gz"
    if not path.is_file():
        request = urllib.request.Request(url, headers={
            "User-Agent": "BLKH-exp66-public-science-1.0"
        })
        with urllib.request.urlopen(request, timeout=90) as response:
            compressed = response.read(32*1024*1024+1)
        if len(compressed)>32*1024*1024:
            raise ValueError("Oversize source HGT")
        path.write_bytes(compressed)
    else:
        compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    if len(raw)!=HGT_DIM*HGT_DIM*2:
        raise ValueError(f"Invalid HGT dimensions for {region}")
    hgt=np.frombuffer(raw,dtype=">i2").reshape(HGT_DIM,HGT_DIM)
    r,c=ORIGIN
    piece=np.asarray(hgt[r:r+PIXELS,c:c+PIXELS],dtype=np.int16)
    if piece.shape!=(PIXELS,PIXELS):
        raise ValueError("Unregistered region crop out of bounds")
    if np.any(piece==-32768):
        raise ValueError(f"Preregistered source {region} contains NoData; flag exclusion and do NOT replace region")
    return piece,{
        "source_url":url, "hgt_region":region,
        "source_file_sha256":digest(compressed),
        "raw_hgt_sha256":digest(raw),
        "crop_sha256":digest(piece.tobytes()),
        "origin_row_col":list(ORIGIN),
        "crop_pixels":[PIXELS,PIXELS],
        "height_unit":"meter",
        "source":"Skadi (SRTM-derived), int16 signed elevation",
    }


def j2k_geodata_archive(image:np.ndarray, info:dict)->bytes:
    blocks=[]
    for y in range(0,256,32):
        for x in range(0,256,32):
            p=jpeg2000_block(image[y:y+32,x:x+32])
            blocks.append((b"",p))
    inner=j2k_archive(image,32,blocks,"j2k")
    raw_meta=json.dumps(info,sort_keys=True,separators=(",",":")).encode()
    if not 0<len(raw_meta)<16384:
        raise ValueError("Too much J2K metadata")
    return PRELUDE.pack(MAGIC_J2KGEO,1,len(raw_meta))+raw_meta+inner


class OpenJ2KGeo:
    def __init__(self,path:Path):
        self.handle=path.open("rb")
        try:
            head=self.handle.read(PRELUDE.size)
            if len(head)!=PRELUDE.size:
                raise ValueError("Truncated geodata wrapper")
            magic,version,length=PRELUDE.unpack(head)
            if magic!=MAGIC_J2KGEO or version!=1 or not 0<length<16384:
                raise ValueError("Invalid geodata wrapper")
            self.metadata=json.loads(self.handle.read(length))
            if self.metadata.get("image_shape")!=[256,256]:
                raise ValueError("Mismatched J2K metadata")
            head=self.handle.read(J2K_HEADER.size)
            if len(head)!=J2K_HEADER.size:
                raise ValueError("Truncated J2K archive index")
            index_size=J2K_HEADER.unpack(head)[-1]
            if index_size>2_000_000:
                raise ValueError("Oversized J2K index")
            idx=self.handle.read(index_size)
            remaining=path.stat().st_size-PRELUDE.size-length
            self.index,self.blocks=parse_header_and_index(head,idx,remaining)
            self.payload_base=PRELUDE.size+length+self.index["payload_start"]
            if self.index["height"]!=256 or self.index["width"]!=256 or self.index["block_side"]!=32:
                raise ValueError("Invalid J2K geometry")
        except BaseException:
            self.handle.close()
            raise

    def __enter__(self):return self
    def __exit__(self,*args):self.handle.close()

    def read(self,row:int,col:int)->np.ndarray:
        if not 0<=row<8 or not 0<=col<8:
            raise IndexError("ROI outside source")
        code,offset,size=self.blocks[row*8+col]
        if code!=1:
            raise ValueError("Non-J2K block")
        self.handle.seek(self.payload_base+offset)
        coded=self.handle.read(size)
        if len(coded)!=size:
            raise ValueError("Truncated J2K ROI")
        return j2k_decode_block(code,coded,(32,32))


def decode_j2k_geo_file(path:Path)->tuple[np.ndarray,dict]:
    with OpenJ2KGeo(path) as reader:
        result=np.empty((256,256),dtype=np.int16)
        for y in range(8):
            for x in range(8):
                result[y*32:(y+1)*32,x*32:(x+1)*32]=reader.read(y,x)
        return result,reader.metadata


def encode_once(name:str,original:np.ndarray,info:dict,path:Path)->None:
    if name=="bh65_fixed20":
        blocks=bh_candidates(original)
        coded,_=bh_pack(original,info,blocks,FIXED_TOLERANCE)
        path.write_bytes(coded)
    elif name=="bh65_compact":
        blocks=bh_candidates(original)
        coded,_=bh_pack(original,info,blocks,0.)
        path.write_bytes(coded)
    elif name=="bh64_fixed":
        path.write_bytes(fixed_encode(original,info))
    elif name=="j2k_indexed":
        path.write_bytes(j2k_geodata_archive(original,info))
    elif name.startswith("tiff_"):
        encode_tiff(path,original,info,name.split("_",1)[1])
    else:
        raise ValueError("Unsupported trial")


def verify_file(name:str,path:Path,original:np.ndarray,info:dict)->None:
    if name.startswith("bh65"):
        check_full(path,original,info)
    elif name=="bh64_fixed":
        a,meta=fixed_decode(path)
        assert meta==info and np.array_equal(a,original)
    elif name=="j2k_indexed":
        a,meta=decode_j2k_geo_file(path)
        assert meta==info and np.array_equal(a,original)
    else:
        verify_tiff(path,original,info)


def open_decoder(name:str,path:Path,stack:ExitStack):
    if name.startswith("bh65"):
        return stack.enter_context(OpenHybrid(path))
    if name=="bh64_fixed":
        return stack.enter_context(OpenBH64(path))
    if name=="j2k_indexed":
        return stack.enter_context(OpenJ2KGeo(path))
    return stack.enter_context(rasterio.open(path))


def roi_read(name:str,reader,row:int,col:int)->np.ndarray:
    if name.startswith("tiff_"):
        return reader.read(1,window=Window(col*32,row*32,32,32))
    return reader.read(row,col)


def compare_one(region:str,image:np.ndarray,provenance:dict,root:Path)->dict:
    directory=root/region
    directory.mkdir(parents=True,exist_ok=True)
    info=metadata(region,image)
    files={}
    times={}
    for name in CODECS:
        suffix=".tif" if name.startswith("tiff") else ".bin"
        path=directory/(name+suffix)
        durations=[]
        for trial in range(TRIALS):
            start=time.perf_counter_ns()
            encode_once(name,image,info,path)
            durations.append((time.perf_counter_ns()-start)/1e9)
        verify_file(name,path,image,info)
        files[name]=path
        times[name]=durations
    full_j2k,decoded=pack_image(image[None],"JPEG2000")
    if not np.array_equal(decoded[0],image):
        raise AssertionError("Whole-JPEG2000 was lossy")
    full_meta=json.dumps(info,sort_keys=True,separators=(",",":")).encode()
    full_stored=PRELUDE.pack(b"J2F66",1,len(full_meta))+full_meta+full_j2k
    full_path=directory/"j2k_full_not_indexed.bin"
    full_path.write_bytes(full_stored)
    samples={name:[] for name in CODECS}
    with ExitStack() as ctx:
        handles={name:open_decoder(name,files[name],ctx) for name in CODECS}
        for i,block_id in enumerate(ROI_READS):
            y,x=divmod(block_id,8)
            actual=image[y*32:(y+1)*32,x*32:(x+1)*32]
            rotated=CODECS[i%len(CODECS):]+CODECS[:i%len(CODECS)]
            for name in rotated:
                started=time.perf_counter_ns()
                block=roi_read(name,handles[name],y,x)
                elapsed=(time.perf_counter_ns()-started)/1e6
                if not np.array_equal(block,actual):
                    raise AssertionError(f"Not exactly lossless: {name}")
                samples[name].append(elapsed)
    output={
        "region":region,"new_region":True,
        "source":provenance,
        "input_sha256":digest(image.tobytes()),
        "geodata":info,
        "roi_reads_per_method":len(ROI_READS),
        "roi_sequence_sha256":digest(np.asarray(ROI_READS,dtype="<i4").tobytes()),
        "roi_mode":"persistent-open, no decoded-block cache in BH65/J2K/BH64; GDAL cache allowed",
        "full_j2k_nonindexed_bytes":len(full_stored),
        "full_j2k_nonindexed_file":full_path.name,
        "full_j2k_nonindexed_sha256":digest(full_stored),
        "sizes_bytes":{name:files[name].stat().st_size for name in CODECS},
        "file_sha256":{name:digest(files[name].read_bytes()) for name in CODECS},
        "file":{name:files[name].name for name in CODECS},
        "encode_times_s":times,
        "encode_median_s":{name:float(np.median(times[name])) for name in CODECS},
        "roi_mean_ms":{name:float(np.mean(samples[name])) for name in CODECS},
        "roi_p95_ms":{name:float(np.percentile(samples[name],95)) for name in CODECS},
        "roi_median_ms":{name:float(np.median(samples[name])) for name in CODECS},
        "zero_elevation_error_m":0,
        "verified_lossless":True,
    }
    output["frozen_hybrid_success_vs_indexed_j2k"]=(
        output["sizes_bytes"]["bh65_fixed20"]<output["sizes_bytes"]["j2k_indexed"]
        and output["roi_mean_ms"]["bh65_fixed20"]<=output["roi_mean_ms"]["j2k_indexed"]
    )
    return output


def self_test(root:Path)->None:
    from PIL import features
    if not features.check("jpg_2000"):
        raise RuntimeError("JPEG2000 is mandatory")
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(66)
    test=rng.integers(-4000,4000,size=(256,256),dtype=np.int16)
    meta=metadata("N39W107",test)
    for name in CODECS:
        path=root/(name+(".tif" if name.startswith("tiff") else ".bin"))
        encode_once(name,test,meta,path)
        verify_file(name,path,test,meta)
    with ExitStack() as ctx:
        j2k=ctx.enter_context(OpenJ2KGeo(root/"j2k_indexed.bin"))
        assert j2k.metadata==meta
        for j in (0,13,63):
            y,x=divmod(j,8)
            assert np.array_equal(j2k.read(y,x),test[y*32:(y+1)*32,x*32:(x+1)*32])
    print("EXP66 SELFTEST PASS: six real codecs, exact signed int16 reconstruction")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--out",default="/tmp/blkh_exp66")
    ap.add_argument("--cache",default="/tmp/blkh_exp66_hgt_cache")
    a=ap.parse_args()
    root=Path(a.out)
    root.mkdir(parents=True,exist_ok=True)
    if a.self_test:
        self_test(root);return
    all_rows=[]
    excluded=[]
    for region in REGIONS:
        try:
            image,source=source_hgt(region,Path(a.cache))
        except Exception as e:
            excluded.append({"region":region,"error":str(e)})
            print("PREDECLARED REGION EXCLUDED, NO REPLACEMENT",region,str(e),flush=True)
            continue
        row=compare_one(region,image,source,root)
        all_rows.append(row)
        print(
            region,"bytes",row["sizes_bytes"],
            "roi_mean_ms",{k:round(v,4) for k,v in row["roi_mean_ms"].items()},
            "success_vs_j2k",row["frozen_hybrid_success_vs_indexed_j2k"],
            flush=True,
        )
        (root/"results.json").write_text(json.dumps({
            "experiment":66,
            "protocol":"BHUH_EXP66_PREREGISTERED_PROTOCOL.md",
            "threshold_frozen":FIXED_TOLERANCE,
            "predeclared_regions":list(REGIONS),
            "excluded_regions":excluded,
            "valid_regions":len(all_rows),
            "comparison_note":"J2K identical per-block index and counted geospatial JSON; TIFF independent 32px blocks",
            "limits":"GDAL block cache differs from uncached Python readers; shared GH runner, not a production CPU benchmark",
            "records":all_rows,
        },indent=2),encoding="utf-8")
    (root/"results.json").write_text(json.dumps({
        "experiment":66,
        "protocol":"BHUH_EXP66_PREREGISTERED_PROTOCOL.md",
        "threshold_frozen":FIXED_TOLERANCE,
        "predeclared_regions":list(REGIONS),
        "excluded_regions":excluded,
        "valid_regions":len(all_rows),
        "comparison_note":"J2K identical per-block index and counted geospatial JSON; TIFF independent 32px blocks",
        "limits":"GDAL block cache differs from uncached Python readers; shared GH runner, not a production CPU benchmark",
        "records":all_rows,
    },indent=2),encoding="utf-8")
    if excluded:
        raise RuntimeError(f"Exp66 had {len(excluded)} preregistered source exclusions, recorded in results.json")
    if len(all_rows)!=len(REGIONS):
        raise AssertionError("Expected all four new real HGT sources")
    wins=sum(x["frozen_hybrid_success_vs_indexed_j2k"] for x in all_rows)
    print(f"EXP66 DONE: {wins}/{len(REGIONS)} registered new regions meet both rate and latency criteria",flush=True)


if __name__=="__main__":
    main()
