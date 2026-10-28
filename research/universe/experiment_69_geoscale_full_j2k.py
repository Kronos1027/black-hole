"""
BLKH Experiment 69 — large-area scaling of truly seekable lossless terrain.

Controlled 256² / 512² / 1024² SAME-SOURCE real Skadi HGT crops against one
whole-image JPEG2000 with real GDAL Window reads. Fixed 32x32 independent
BH69 blocks choose known BH58 variable-vs-BH64 fixed delta at frozen 20%
per-block overhead tolerance. Count complete geospatial JSON, block index,
all codecs' metadata, and signed-height origin for both methods.

Two previously known Exp66 regions (NOT independent holdout): N39W107 and
S22W044; expand same fixed crop origin (1200,1300) at sizes 256/512/1024.
Void pixels (-32768) cause explicit failure/exclusion, not silent filling.
ROI samples: 64 distinct randomly selected block coordinates (uniform) then
repeat those coordinates 3 times on the SAME open data handles. GDAL may cache
internal JPEG2000 image portions while BH69 reader does not cache blocks.
All pixel data and actual regional ROIs must decode to exact int16 meters.

NO new theoretical compressor or guaranteed winner is claimed.
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

import numpy as np
import rasterio
from rasterio.windows import Window

from experiment_66_preregistered_holdout import (
    source_hgt as earlier_source, HGT_DIM, ORIGIN, digest,
)
from experiment_60_geotiff_roi import metadata, PRELUDE
from experiment_58_vectorized_decoder import (
    block_encode as compact_encode, block_decode as compact_decode,
)
from experiment_64_fixed_delta_roi import (
    block_encode as fast_encode, block_decode as fast_decode,
)
from experiment_56c_contiguous_mosaic import pack_image

REGIONS=("N39W107","S22W044")
SIZES=(256,512,1024)
BLOCK=32
THRESHOLD=0.20
MAGIC=b"BHG69"
INNER=struct.Struct("<5sBHHHI")
ENTRY=struct.Struct("<BII")
MAX_BLOCKS=16384
MAX_META=16384
MAX_PAYLOAD=32_000_000


def full_source(region:str,pixels:int,cache:Path)->tuple[np.ndarray,dict]:
    if region not in REGIONS or pixels not in SIZES:
        raise ValueError("Region/size not in locked exploration")
    _,prov=earlier_source(region,cache)
    compressed=(cache/(region+".hgt.gz")).read_bytes()
    raw=gzip.decompress(compressed)
    image=np.frombuffer(raw,dtype=">i2").reshape((HGT_DIM,HGT_DIM))
    row,col=ORIGIN
    original=np.asarray(image[row:row+pixels,col:col+pixels],dtype=np.int16)
    if original.shape!=(pixels,pixels):
        raise ValueError("Incomplete source footprint")
    if (original==-32768).any():
        raise ValueError(f"NoData void in preregistered source {region} size {pixels}")
    return original,{
        **prov, "source_original_int16_sha256":digest(original.tobytes()),
        "source_full_hgt_sha256":digest(raw),
        "crop_row_col":list(ORIGIN),"pixels":pixels,
    }


def encode_bh69(img:np.ndarray,info:dict)->tuple[bytes,dict]:
    h,w=img.shape
    if (img.dtype!=np.int16 or h!=w or h not in SIZES or
            h%BLOCK or w%BLOCK):
        raise ValueError("Invalid BH69 input")
    meta=json.dumps(info,sort_keys=True,separators=(",",":")).encode()
    if not 0<len(meta)<MAX_META:
        raise ValueError("Bad geographic metadata")
    index=bytearray()
    body=bytearray()
    fast_count=0
    for row in range(0,h,BLOCK):
        for col in range(0,w,BLOCK):
            tile=img[row:row+BLOCK,col:col+BLOCK]
            small=compact_encode(tile)
            quick=fast_encode(tile)
            if len(quick)<=len(small)*(1+THRESHOLD):
                codec,data=3,quick
                fast_count+=1
            else:
                codec,data=2,small
            index.extend(ENTRY.pack(codec,len(body),len(data)))
            body.extend(data)
    if len(body)>MAX_PAYLOAD:
        raise ValueError("Compressed source too large")
    header=INNER.pack(MAGIC,1,h,w,BLOCK,len(meta))
    return header+meta+index+body,{
        "fast_blocks":fast_count,
        "compact_blocks":(h//BLOCK)*(w//BLOCK)-fast_count,
        "complete_index_bytes":len(index),
        "metadata_and_header_bytes":len(header)+len(meta),
        "compressed_payload_bytes":len(body),
    }


class OpenBH69:
    def __init__(self,path:Path):
        self.handle=path.open("rb")
        try:
            header=self.handle.read(INNER.size)
            if len(header)!=INNER.size:
                raise ValueError("Truncated BH69 header")
            magic,ver,h,w,side,meta_len=INNER.unpack(header)
            if (magic!=MAGIC or ver!=1 or h not in SIZES or w!=h
                    or side!=BLOCK or not 0<meta_len<MAX_META):
                raise ValueError("Invalid BH69 archive geometry")
            raw=self.handle.read(meta_len)
            if len(raw)!=meta_len:
                raise ValueError("Truncated metadata")
            self.info=json.loads(raw)
            if (self.info.get("image_shape")!=[h,w] or
                self.info.get("crs")!="EPSG:4326" or
                self.info.get("sample_unit")!="meter" or
                self.info.get("dtype")!="int16"):
                raise ValueError("Incorrect geospatial header")
            count=(h//BLOCK)*(w//BLOCK)
            if count>MAX_BLOCKS:
                raise ValueError("Too many blocks")
            raw_index=self.handle.read(count*ENTRY.size)
            if len(raw_index)!=count*ENTRY.size:
                raise ValueError("Truncated BH69 index")
            entries=[]
            offset=0
            for i in range(count):
                codec,a,length=ENTRY.unpack_from(raw_index,i*ENTRY.size)
                if codec not in (2,3) or a!=offset or length<14:
                    raise ValueError("Invalid or non-contiguous BH69 index")
                offset+=length
                if offset>MAX_PAYLOAD:
                    raise ValueError("Invalid payload length")
                entries.append((codec,a,length))
            self.start=INNER.size+meta_len+len(raw_index)
            if self.start+offset!=path.stat().st_size:
                raise ValueError("Invalid BH69 final stream size")
            self.h=h
            self.entries=entries
        except BaseException:
            self.handle.close()
            raise

    def __enter__(self):return self
    def __exit__(self,*args):self.handle.close()

    def read(self,y:int,x:int)->np.ndarray:
        bound=self.h//BLOCK
        if not (0<=y<bound and 0<=x<bound):
            raise IndexError("BH69 ROI out of range")
        codec,start,length=self.entries[y*bound+x]
        self.handle.seek(self.start+start)
        raw=self.handle.read(length)
        if len(raw)!=length:
            raise ValueError("Truncated tile")
        return compact_decode(raw) if codec==2 else fast_decode(raw)


def encode_j2k_full(img:np.ndarray,info:dict)->tuple[bytes,bytes,int]:
    raw,decoded=pack_image(img[None],"JPEG2000")
    if not np.array_equal(decoded[0],img):
        raise AssertionError("J2K whole source mismatch")
    meta=json.dumps(info,sort_keys=True,separators=(",",":")).encode()
    if not 0<len(meta)<MAX_META:
        raise ValueError("Invalid metadata")
    origin=struct.unpack("<i",raw[:4])[0]
    full=PRELUDE.pack(b"J2G69",1,len(meta))+meta+raw
    return full,raw[8:],origin


def benchmark(region:str,n:int,src:np.ndarray,prov:dict,out:Path)->dict:
    out.mkdir(parents=True,exist_ok=True)
    info=metadata(region,src)
    info["geographic_crop_row_col"]=list(ORIGIN)
    files={}
    start=time.perf_counter()
    bh,detail=encode_bh69(src,info)
    path=out/"terrain_bh69.bin"
    path.write_bytes(bh)
    enc_bh=time.perf_counter()-start
    files["bh69"]=path

    start=time.perf_counter()
    j2kg,raw_jp2,origin=encode_j2k_full(src,info)
    jp2=out/"whole_image.jp2"
    jp2.write_bytes(raw_jp2)
    wrapped=out/"full_jp2_with_metadata.bin"
    wrapped.write_bytes(j2kg)
    enc_j2k=time.perf_counter()-start
    files["j2k_full"]=wrapped

    with ExitStack() as stack:
        b=stack.enter_context(OpenBH69(path))
        ds=stack.enter_context(rasterio.open(jp2))
        if ds.width!=n or ds.height!=n:
            raise AssertionError("J2K geometry mismatch")
        if b.info!=info:
            raise AssertionError("BH69 geospatial metadata corrupted")
        full_reconstructed=np.empty((n,n),dtype=np.int16)
        for y in range(n//BLOCK):
            for x in range(n//BLOCK):
                tile=b.read(y,x)
                full_reconstructed[y*BLOCK:(y+1)*BLOCK,x*BLOCK:(x+1)*BLOCK]=tile
        if not np.array_equal(full_reconstructed,src):
            raise AssertionError("Full BH69 reconstruction is not lossless")
        with_jp2=ds.read(1).astype(np.int32)+origin
        if not np.array_equal(with_jp2,src):
            raise AssertionError("Full JP2 losslessness failed")
        rng=np.random.default_rng(690069)
        cells=(n//BLOCK)**2
        selected=[int(i) for i in rng.choice(cells,size=64,replace=False)]
        orders={
            "distinct64":selected,
            "repeat64x3":selected*3
        }
        timing={}
        for workload,ids in orders.items():
            t={"bh69":[],"j2k_full":[]}
            for count,idx in enumerate(ids):
                y,x=divmod(idx,n//BLOCK)
                truth=src[y*BLOCK:(y+1)*BLOCK,x*BLOCK:(x+1)*BLOCK]
                methods=("bh69","j2k_full") if count%2==0 else ("j2k_full","bh69")
                for mode in methods:
                    begin=time.perf_counter_ns()
                    if mode=="bh69":
                        block=b.read(y,x)
                    else:
                        block=ds.read(1,window=Window(x*BLOCK,y*BLOCK,BLOCK,BLOCK)).astype(np.int32)+origin
                    elapsed=(time.perf_counter_ns()-begin)/1e6
                    if not np.array_equal(block,truth):
                        raise AssertionError("Incorrect ROI decoded")
                    t[mode].append(elapsed)
            timing[workload]={
                "positions":len(ids),
                "mean_ms":{k:float(np.mean(v)) for k,v in t.items()},
                "median_ms":{k:float(np.median(v)) for k,v in t.items()},
                "p95_ms":{k:float(np.percentile(v,95)) for k,v in t.items()},
            }
    return {
        "region":region,"pixels":n,
        "source":prov,"source_sha256":digest(src.tobytes()),
        "geospatial_metadata":info,
        "size_bytes":{k:p.stat().st_size for k,p in files.items()},
        "sha256":{k:digest(p.read_bytes()) for k,p in files.items()},
        "file":{k:p.name for k,p in files.items()},
        "jp2_gdal_file":jp2.name,"jp2_gdal_sha256":digest(raw_jp2),
        "frozen_hybrid_selector":THRESHOLD,
        "bh69_detail":detail,
        "encode_seconds":{"bh69":enc_bh,"j2k_full":enc_j2k},
        "roi":timing,
        "zero_elevation_error_m":0,
        "all_lossless":True,
    }


def self_test(root:Path):
    rng=np.random.default_rng(69)
    source=rng.integers(-32768,32768,(256,256),dtype=np.int16)
    root.mkdir(parents=True,exist_ok=True)
    region=REGIONS[0]
    m=metadata(region,source)
    blob,_=encode_bh69(source,m)
    path=root/"test.bh69"
    path.write_bytes(blob)
    with OpenBH69(path) as f:
        assert f.info==m
        for y in range(8):
            for x in range(8):
                assert np.array_equal(f.read(y,x),
                    source[y*32:(y+1)*32,x*32:(x+1)*32])
    try:
        path.write_bytes(blob[:-1])
        with OpenBH69(path):pass
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupted BH69 accepted")
    print("EXP69 SELF-TEST PASS: exact uint16 deltas + variable grid metadata")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",default="/tmp/blkh_exp69")
    p.add_argument("--cache",default="/tmp/blkh_exp69_skadi_cache")
    a=p.parse_args()
    root=Path(a.out)
    root.mkdir(parents=True,exist_ok=True)
    if a.self_test:
        self_test(root)
        return
    records=[]
    excluded=[]
    for region in REGIONS:
        for n in SIZES:
            try:
                src,prov=full_source(region,n,Path(a.cache))
            except ValueError as error:
                excluded.append({"region":region,"size":n,"reason":str(error)})
                print("EXCLUDED: no data; no alternate source",region,n,str(error),flush=True)
                continue
            result=benchmark(region,n,src,prov,root/region/str(n))
            records.append(result)
            print(region,n,"bytes",result["size_bytes"],
                  "fresh_64_mean_ms",result["roi"]["distinct64"]["mean_ms"],
                  "repeat_mean_ms",result["roi"]["repeat64x3"]["mean_ms"],
                  flush=True)
            (root/"results.json").write_text(json.dumps({
                "experiment":69,
                "status":"provisional until independent GitHub action validation",
                "sources_predefined":list(REGIONS),
                "resolutions":list(SIZES),
                "excluded":excluded,
                "source_count":len(records),
                "limitations":"Only 2 previously used HGT sources; GDAL cache not matched to BH69; one runner",
                "records":records,
            },indent=2),encoding="utf-8")
    if excluded:
        print("SOME GEOGRAPHIES EXCLUDED: NOT A COMPLETE SIX-CONDITION TEST",flush=True)
    print("EXP69 DONE: ",len(records),"fully decoded real raster size configurations",flush=True)


if __name__=="__main__":
    main()
