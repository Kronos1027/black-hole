"""
Exp62 — bounded LRU for persistently opened BH60 georeferenced DEM tiles.

Addresses Exp61's finding that GDAL caches warm tiles while BH60's previous
reader recomputed the exact same 32px block on every query. Equal workload:
24 selected distinct blocks warmed before timing, 5 repeated passes, same
persistent open GeoTIFF datasets and BH60 handle. Cache holds at most
32 decoded blocks = 65,536 bytes of int16 pixel arrays; record byte budget,
cache hit rate, and exact lossless output. This measures a *warm-cache system*
feature, not better compression or a new decompression algorithm.

Warning: BH60 cached hits may avoid disk I/O and decoding entirely; GIS/GDAL
does the same. To measure miss latency, conduct an independent >cache-size
random-access study; never generalize these warm times to cold performance.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
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
    source_hgt, read_holdout, source_cases,
    read_bh60_meta, INDEX_HEADER, read_index, block_decode,
)
from experiment_61_warm_geotiff import setup_files


class CachedBH60:
    def __init__(self,path:Path,limit:int=32):
        if not 1<=limit<=1000:
            raise ValueError("Cache entries must be positive and bounded")
        self.file=path.open("rb")
        self.cache=OrderedDict()
        self.limit=limit
        self.hits=0
        self.misses=0
        try:
            self.info,self.start=read_bh60_meta(self.file)
            head=self.file.read(INDEX_HEADER.size)
            if len(head)!=INDEX_HEADER.size:
                raise ValueError("Incomplete inner header")
            size=INDEX_HEADER.unpack(head)[-1]
            if size>2_000_000:
                raise ValueError("Invalid index length")
            idx=self.file.read(size)
            self.height,self.width,self.side,self.entries=read_index(
                head,idx,path.stat().st_size-self.start
            )
            if self.info["image_shape"]!=[self.height,self.width]:
                raise ValueError("Metadata and index disagree")
            self.payload_start=self.start+INDEX_HEADER.size+size
        except BaseException:
            self.file.close()
            raise

    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.file.close()
        self.cache.clear()

    @property
    def cache_bytes(self):
        return sum(block.nbytes for block in self.cache.values())

    def read(self,row:int,col:int)->np.ndarray:
        if not (0<=row<self.height//self.side and 0<=col<self.width//self.side):
            raise IndexError("Tile outside image")
        key=(row,col)
        if key in self.cache:
            self.hits+=1
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        self.misses+=1
        i=row*(self.width//self.side)+col
        offset,length=self.entries[i]
        self.file.seek(self.payload_start+offset)
        payload=self.file.read(length)
        if len(payload)!=length:
            raise ValueError("Truncated BH60 compressed tile")
        block=block_decode(payload)
        if block.shape!=(self.side,self.side):
            raise ValueError("Corrupted ROI shape")
        self.cache[key]=block
        if len(self.cache)>self.limit:
            self.cache.popitem(last=False)
        return block.copy()


def test_cache(tmp:Path):
    from experiment_60_geotiff_roi import encode_bh60,metadata
    img=np.arange(64*96,dtype=np.int16).reshape(64,96)
    path=tmp/"test.bh60"
    path.write_bytes(encode_bh60(img,metadata("N37W119",img)))
    with CachedBH60(path,limit=2) as reader:
        a=reader.read(0,0)
        a[0,0]=777
        assert reader.read(0,0)[0,0] == img[0,0] # copy-on-return
        reader.read(0,1)
        reader.read(0,2)
        assert len(reader.cache)==2
        assert reader.read(0,0)[0,0]==img[0,0] # re-decode after eviction
        assert reader.cache_bytes<=2*32*32*2
        assert reader.hits>=1 and reader.misses>=4
    print("EXP62 CACHE SELF-TEST PASS: exact, bounded, evicts, returns copy")


def benchmark_case(region:str,case:str,img:np.ndarray,root:Path,repeats:int)->dict:
    p=setup_files(region,case,img,root)
    h,w=img.shape
    rng=np.random.default_rng(620062)
    count=24
    ids=sorted(int(x) for x in rng.choice(
        (h//ROI_SIDE)*(w//ROI_SIDE),size=count,replace=False
    ))
    coords=[divmod(i,w//ROI_SIDE) for i in ids]
    samples={key:[] for key in CODECS}
    with ExitStack() as stack:
        cached=stack.enter_context(CachedBH60(p["bh60"],32))
        tiff={k:stack.enter_context(rasterio.open(p[k]))
              for k in CODECS if k!="bh60"}
        def read(mode,row,col):
            return cached.read(row,col) if mode=="bh60" else tiff[mode].read(
                1,window=Window(col*ROI_SIDE,row*ROI_SIDE,ROI_SIDE,ROI_SIDE)
            )
        for row,col in coords:
            expected=img[row*ROI_SIDE:(row+1)*ROI_SIDE,
                         col*ROI_SIDE:(col+1)*ROI_SIDE]
            for key in CODECS:
                if not np.array_equal(read(key,row,col),expected):
                    raise AssertionError("Wrong warmup tile")
        initial_hits=cached.hits
        initial_misses=cached.misses
        for repetition in range(repeats):
            order=CODECS[repetition%len(CODECS):]+CODECS[:repetition%len(CODECS)]
            for row,col in coords:
                expected=img[row*ROI_SIDE:(row+1)*ROI_SIDE,
                             col*ROI_SIDE:(col+1)*ROI_SIDE]
                for key in order:
                    start=time.perf_counter_ns()
                    result=read(key,row,col)
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    if not np.array_equal(result,expected):
                        raise AssertionError("Lossless ROI mismatch")
                    samples[key].append(elapsed)
        reads_h=cached.hits-initial_hits
        reads_m=cached.misses-initial_misses
        cache_bytes=cached.cache_bytes
    expected_bytes=32*ROI_SIDE*ROI_SIDE*2
    if cache_bytes>expected_bytes or reads_m!=0 or reads_h!=count*repeats:
        raise AssertionError("Warm cache unexpectedly missed or exceeded byte bound")
    row={
        "region":region,"case":case,"shape":list(img.shape),
        "full_files_bytes":{key:p[key].stat().st_size for key in CODECS},
        "complete_file_sha256":{key:hashlib.sha256(p[key].read_bytes()).hexdigest()
                                for key in CODECS},
        "files":{key:p[key].name for key in CODECS},
        "input_sha256":hashlib.sha256(img.tobytes()).hexdigest(),
        "sampled_tile_ids":ids,"repetitions":repeats,
        "cache_limit_blocks":32,"cache_max_decoded_pixel_bytes":expected_bytes,
        "actual_decoded_pixel_cache_bytes":cache_bytes,
        "bh60_warm_cache_hits":reads_h,"bh60_warm_cache_misses":reads_m,
        "median_ms":{key:float(np.median(samples[key])) for key in CODECS},
        "p95_ms":{key:float(np.percentile(samples[key],95)) for key in CODECS},
        "lossless_verified":True,"max_error_m":0,
    }
    print(region,case,"warm_cache_ms",
          {k:round(v,4) for k,v in row["median_ms"].items()},
          "BH60 cache hits",reads_h,flush=True)
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",default="/tmp/blkh_exp62")
    ap.add_argument("--cache",default="/tmp/blkh_exp62_cache")
    ap.add_argument("--repeats",type=int,default=5)
    a=ap.parse_args()
    if not 2<=a.repeats<=10:
        ap.error("Invalid repeat count")
    root=Path(a.out)
    root.mkdir(parents=True,exist_ok=True)
    test=root/"_selftest"
    test.mkdir(parents=True,exist_ok=True)
    test_cache(test)
    rows=[]
    for region in REGIONS:
        src,_=(read_holdout(region,Path(a.cache))
               if region in MORE_REGIONS else source_hgt(region,Path(a.cache)))
        for case,img in source_cases(src):
            rows.append(benchmark_case(region,case,img,root,a.repeats))
        (root/"results.json").write_text(json.dumps({
            "experiment":62,
            "purpose":"bounded cached BH60 read vs naturally GDAL-cached persistent GeoTIFF",
            "limitations":"24 distinct warmed blocks fit in 32-block LRU; all warmed queries are cache hits, not decoding; shared GH runner",
            "records":rows,
        },indent=2),encoding="utf8")
    assert len(rows)==12
    print("EXP62 DONE: 12 GEOREFERENCED WARM-CACHE TESTS, 32-TILE CAP",flush=True)


if __name__=="__main__":
    main()
