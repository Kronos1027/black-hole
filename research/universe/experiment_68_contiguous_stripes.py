"""
Experiment 68: contiguous horizontal STRIPES versus full-JPEG2000 ROI.

Exp67 revealed whole-JP2 is much smaller than a 64-block independent codec
and can still expose 32x32 windows via GDAL. Test a continuum: encode 32,
64 or 128-high full-width stripes independently. Per-stripe choose
  (1) reversible modular-u16 delta + Brotli
  (2) reversible JPEG2000 with signed-elevation origin
  (3) smaller byte payload of the two.
Compare total serialised sizes, true physical seek to ONE stripe for ROI,
and persistent-handle read+decode. All stored archive bytes include matching
geospatial JSON, index, height origin and headers. Four previously used
Exp66 real HGT grids are EXPLORATORY, not independent held-out validation.
This is established codec engineering, NOT a novel compression theorem.

  python research/universe/experiment_68_contiguous_stripes.py --self-test
  python research/universe/experiment_68_contiguous_stripes.py --out /tmp/blkh_exp68
"""
from __future__ import annotations

import argparse
import brotli
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import struct
import time

import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window

from experiment_66_preregistered_holdout import REGIONS, source_hgt, digest
from experiment_60_geotiff_roi import metadata, encode_tiff, PRELUDE
from experiment_65_hybrid_pareto import (
    candidates as bh65_candidates, pack as bh65_pack, OpenHybrid, check_full
)
from experiment_56c_contiguous_mosaic import pack_image

HEADER = struct.Struct("<5sBHHHHI")
ENTRY = struct.Struct("<BII")
PART = struct.Struct("<5sBHHI")
MAGIC = b"BH68S"
PART_MAGIC = b"BH68B"
VERS = 1
WIDTH = 256
HEIGHT = 256
ROI = 32
STRIPES = (32, 64, 128)
STRATEGIES = ("brotli_delta", "jpeg2000", "adaptive")
MODES = tuple(f"stripe_{height}_{method}" for height in STRIPES
              for method in STRATEGIES)
BASE = ("bh65_individual", "jpeg2000_full", "tiff_zstd")
ALL = MODES+BASE
MAX_META = 16_384
MAX_PAYLOAD = 2_000_000


def bytes_sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pack_fixed_delta(a: np.ndarray) -> bytes:
    a=np.asarray(a,dtype=np.int16)
    h,w=a.shape
    if w != WIDTH or h not in STRIPES:
        raise ValueError("Unexpected stripe geometry")
    u=a.view(np.uint16)
    forms=[]
    for mode in (0,1):
        delta=np.empty_like(u)
        if mode==0:
            delta[:,0]=u[:,0]
            delta[:,1:]=u[:,1:]-u[:,:-1]
        else:
            delta[0,:]=u[0,:]
            delta[1:,:]=u[1:,:]-u[:-1,:]
        packed=delta.astype("<u2",copy=False).tobytes()
        c=brotli.compress(packed,quality=8)
        forms.append(PART.pack(PART_MAGIC,mode,h,w,len(c))+c)
    return min(forms,key=len)


def decode_fixed_delta(blob: bytes) -> np.ndarray:
    if len(blob)<PART.size:
        raise ValueError("Truncated delta")
    m,mode,h,w,n=PART.unpack(blob[:PART.size])
    if (m!=PART_MAGIC or mode not in (0,1) or w!=WIDTH or h not in STRIPES
            or n != len(blob)-PART.size or n>MAX_PAYLOAD):
        raise ValueError("Invalid stripe block")
    dec=brotli.Decompressor()
    buf=bytearray()
    for pos in range(PART.size,len(blob),4096):
        buf.extend(dec.process(blob[pos:pos+4096]))
        if len(buf)>h*w*2:
            raise ValueError("Oversized delta stream")
    if not dec.is_finished() or len(buf)!=h*w*2:
        raise ValueError("Incomplete delta stream")
    arr=np.frombuffer(buf,dtype="<u2").reshape(h,w)
    return np.cumsum(arr.astype(np.int32),axis=1 if mode==0 else 0,
                     dtype=np.int32).astype(np.uint16).view(np.int16)


def pack_jpeg2000(a: np.ndarray)->bytes:
    lo=int(a.min())
    shifted=(a.astype(np.int32)-lo).astype(np.uint16)
    with io.BytesIO() as f:
        Image.fromarray(shifted).save(f,format="JPEG2000",irreversible=False)
        raw=f.getvalue()
    out=struct.pack("<i",lo)+raw
    if not np.array_equal(decode_jpeg2000(out,a.shape),a):
        raise AssertionError("JPEG2000 stripe not exactly lossless")
    return out


def decode_jpeg2000(blob: bytes, shape: tuple[int,int])->np.ndarray:
    if len(blob)<16:
        raise ValueError("Truncated JP2 stripe")
    lo=struct.unpack("<i",blob[:4])[0]
    with Image.open(io.BytesIO(blob[4:])) as f:
        restored=np.asarray(f).astype(np.int32)+lo
    if (restored.shape!=shape or restored.min()<-32768
            or restored.max()>32767):
        raise ValueError("Invalid signed JP2 reconstruction")
    return restored.astype(np.int16)


def pack_stripes(image: np.ndarray, info:dict, height:int,
                 strategy:str)->tuple[bytes,dict]:
    if image.shape!=(HEIGHT,WIDTH) or image.dtype!=np.int16:
        raise ValueError("Incorrect source image")
    if height not in STRIPES or strategy not in STRATEGIES:
        raise ValueError("Unexpected stripe configuration")
    meta=json.dumps(info,sort_keys=True,separators=(",",":")).encode("utf-8")
    if not 0<len(meta)<MAX_META:
        raise ValueError("Too much geospatial metadata")
    payload=bytearray()
    index=bytearray()
    histogram={"brotli_delta":0,"jpeg2000":0}
    for y in range(0,HEIGHT,height):
        band=np.asarray(image[y:y+height],dtype=np.int16)
        native=pack_fixed_delta(band)
        j2k=pack_jpeg2000(band)
        selected=(1,native) if strategy=="brotli_delta" else (
            (2,j2k) if strategy=="jpeg2000" else
            ((1,native) if len(native)<=len(j2k) else (2,j2k))
        )
        kind,coded=selected
        histogram["brotli_delta" if kind==1 else "jpeg2000"]+=1
        index.extend(ENTRY.pack(kind,len(payload),len(coded)))
        payload.extend(coded)
    n=HEIGHT//height
    header=HEADER.pack(MAGIC,VERS,HEIGHT,WIDTH,height,n,len(meta))
    return header+meta+index+payload,{
        "stripes":n,"stripe_height":height,
        "band_codec_counts":histogram,
        "metadata_bytes":HEADER.size+len(meta),
        "index_bytes":len(index),
        "payload_bytes":len(payload),
    }


class OpenStripes:
    def __init__(self,path:Path):
        self.fd=path.open("rb")
        try:
            hdr=self.fd.read(HEADER.size)
            if len(hdr)!=HEADER.size:
                raise ValueError("Missing header")
            magic,vers,h,w,band,count,mlen=HEADER.unpack(hdr)
            if (magic!=MAGIC or vers!=VERS or h!=HEIGHT or w!=WIDTH
                    or band not in STRIPES or count!=HEIGHT//band
                    or not 0<mlen<MAX_META):
                raise ValueError("Invalid BH68 header")
            meta=self.fd.read(mlen)
            if len(meta)!=mlen:
                raise ValueError("Truncated georeferencing")
            self.info=json.loads(meta)
            if (self.info.get("image_shape")!=[HEIGHT,WIDTH]
                    or self.info.get("crs")!="EPSG:4326"
                    or self.info.get("sample_unit")!="meter"
                    or self.info.get("dtype")!="int16"):
                raise ValueError("Unsupported geographic metadata")
            idx=self.fd.read(count*ENTRY.size)
            if len(idx)!=count*ENTRY.size:
                raise ValueError("Truncated band index")
            self.entries=[]
            end=0
            for i in range(count):
                codec,offset,length=ENTRY.unpack_from(idx,i*ENTRY.size)
                if codec not in (1,2) or offset!=end or length<16:
                    raise ValueError("Invalid band index")
                end+=length
                if end>MAX_PAYLOAD:
                    raise ValueError("Excessive compressed payload")
                self.entries.append((codec,offset,length))
            self.base=HEADER.size+mlen+len(idx)
            if self.base+end!=path.stat().st_size:
                raise ValueError("Truncated/trailing coded bands")
            self.band=band
        except BaseException:
            self.fd.close()
            raise

    def __enter__(self):return self
    def __exit__(self,*args):self.fd.close()

    def read_band(self,index:int)->np.ndarray:
        if not 0<=index<len(self.entries):
            raise IndexError("Outside band index")
        codec,off,n=self.entries[index]
        self.fd.seek(self.base+off)
        raw=self.fd.read(n)
        if len(raw)!=n:
            raise ValueError("Truncated compressed band")
        arr=(decode_fixed_delta(raw) if codec==1
             else decode_jpeg2000(raw,(self.band,WIDTH)))
        if arr.shape!=(self.band,WIDTH):
            raise ValueError("Wrong decoded band geometry")
        return arr

    def read(self,row:int,col:int)->np.ndarray:
        if not 0<=row<8 or not 0<=col<8:
            raise IndexError("Outside 32x32 ROI")
        y=row*ROI
        band_index=y//self.band
        strip=self.read_band(band_index)
        relative=y-band_index*self.band
        return np.ascontiguousarray(strip[relative:relative+ROI,col*ROI:(col+1)*ROI])


def verify_file(path: Path,original:np.ndarray,info:dict)->None:
    with OpenStripes(path) as r:
        assert r.info==info
        for y in range(8):
            for x in range(8):
                if not np.array_equal(r.read(y,x),
                         original[y*32:(y+1)*32,x*32:(x+1)*32]):
                    raise AssertionError("BH68 lossless ROI failed")


def self_test(root:Path)->None:
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(68)
    tests=[
        rng.integers(-32768,32768,(256,256),dtype=np.int16),
        (np.indices((256,256)).sum(axis=0)*2-300).astype(np.int16),
    ]
    for j,image in enumerate(tests):
        info=metadata("N39W107",image)
        for h in STRIPES:
            for strategy in STRATEGIES:
                b,stat=pack_stripes(image,info,h,strategy)
                f=root/f"selftest_{j}_{h}_{strategy}.bh68"
                f.write_bytes(b)
                verify_file(f,image,info)
                assert stat["stripes"]==256//h
                try:
                    f.write_bytes(b[:-1])
                    verify_file(f,image,info)
                except (ValueError,brotli.error,OSError):
                    pass
                else:
                    raise AssertionError("Truncated archive accepted")
    print("EXP68 SELF-TEST PASS: 2 synthetic rasters, 3 stripe heights x 3 coding modes")


def benchmark_one(region:str,image:np.ndarray,provenance:dict,
                  root:Path)->dict:
    target=root/region
    target.mkdir(parents=True,exist_ok=True)
    info=metadata(region,image)
    stored={}
    encode_s={}
    stats={}
    for height in STRIPES:
        for strategy in STRATEGIES:
            key=f"stripe_{height}_{strategy}"
            start=time.perf_counter()
            blob,counts=pack_stripes(image,info,height,strategy)
            file=target/(key+".bh68")
            file.write_bytes(blob)
            encode_s[key]=time.perf_counter()-start
            verify_file(file,image,info)
            stored[key]=file
            stats[key]=counts

    bh65=target/"bh65_fixed20.bh65"
    start=time.perf_counter()
    blocks=bh65_candidates(image)
    raw,_=bh65_pack(image,info,blocks,0.20)
    bh65.write_bytes(raw)
    encode_s["bh65_individual"]=time.perf_counter()-start
    check_full(bh65,image,info)
    stored["bh65_individual"]=bh65

    # The global JP2 file must be accounted for with its geodata JSON and
    # signed int16 origin, as it was in Exp67.
    start=time.perf_counter()
    full_data,rec=pack_image(image[None],"JPEG2000")
    if not np.array_equal(rec[0],image):
        raise AssertionError("Full-JP2 is not lossless")
    rawmeta=json.dumps(info,sort_keys=True,separators=(",",":")).encode()
    fullwrap=PRELUDE.pack(b"J2F68",1,len(rawmeta))+rawmeta+full_data
    fullfile=target/"jpeg2000_full_georeferenced_wrapper.bin"
    fullfile.write_bytes(fullwrap)
    jp2=target/"jpeg2000_full_access.jp2"
    jp2.write_bytes(full_data[8:])
    encode_s["jpeg2000_full"]=time.perf_counter()-start
    stored["jpeg2000_full"]=fullfile

    tif=target/"tiff_zstd.tif"
    start=time.perf_counter()
    encode_tiff(tif,image,info,"zstd")
    encode_s["tiff_zstd"]=time.perf_counter()-start
    with rasterio.open(tif) as ds:
        if not np.array_equal(ds.read(1),image):
            raise AssertionError("Lossless GeoTIFF failed")
    stored["tiff_zstd"]=tif

    times={key:[] for key in ALL}
    accesses={}
    with ExitStack() as stack:
        handles={key:stack.enter_context(OpenStripes(stored[key]))
                 for key in MODES}
        handles["bh65_individual"]=stack.enter_context(OpenHybrid(bh65))
        handles["jpeg2000_full"]=stack.enter_context(rasterio.open(jp2))
        handles["tiff_zstd"]=stack.enter_context(rasterio.open(tif))
        origin=struct.unpack("<ii",full_data[:8])[0]
        full=np.asarray(handles["jpeg2000_full"].read(1),dtype=np.int32)+origin
        assert np.array_equal(full,image)
        for i in list(range(64))*2:
            y,x=divmod(i,8)
            expected=image[y*32:(y+1)*32,x*32:(x+1)*32]
            order=ALL[i%len(ALL):]+ALL[:i%len(ALL)]
            for key in order:
                t=time.perf_counter_ns()
                h=handles[key]
                if key=="jpeg2000_full":
                    rec=h.read(1,window=Window(x*32,y*32,32,32)).astype(np.int32)+origin
                    rec=rec.astype(np.int16)
                elif key=="tiff_zstd":
                    rec=h.read(1,window=Window(x*32,y*32,32,32))
                else:
                    rec=h.read(y,x)
                elapsed=(time.perf_counter_ns()-t)/1e6
                if not np.array_equal(rec,expected):
                    raise AssertionError(f"Incorrect ROI decode for {key}")
                times[key].append(elapsed)
    return {
        "region":region,
        "source":provenance,
        "geographic_metadata":info,
        "original_sha256":bytes_sha(image.tobytes()),
        "requested_roi_count":128,
        "counted_file_bytes":{key:f.stat().st_size for key,f in stored.items()},
        "file_sha256":{key:bytes_sha(f.read_bytes()) for key,f in stored.items()},
        "file":{key:f.name for key,f in stored.items()},
        "full_jp2_raw_for_gdal_only":jp2.name,
        "full_jp2_raw_sha256":bytes_sha(jp2.read_bytes()),
        "stripe_config":stats,
        "encode_seconds":encode_s,
        "roi_mean_ms":{key:float(np.mean(v)) for key,v in times.items()},
        "roi_median_ms":{key:float(np.median(v)) for key,v in times.items()},
        "roi_p95_ms":{key:float(np.percentile(v,95)) for key,v in times.items()},
        "lossless":True,
        "max_height_error_m":0,
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="/tmp/blkh_exp68")
    p.add_argument("--cache",default="/tmp/blkh_exp68_hgt_cache")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    root=Path(a.out)
    root.mkdir(parents=True,exist_ok=True)
    if a.self_test:
        self_test(root)
        return
    results=[]
    for region in REGIONS:
        image,provenance=source_hgt(region,Path(a.cache))
        record=benchmark_one(region,image,provenance,root)
        results.append(record)
        print(region,
              "bytes",{k:v for k,v in record["counted_file_bytes"].items()},
              "roi_ms",{k:round(v,3) for k,v in record["roi_mean_ms"].items()},
              flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":68,
            "status":"reproduced only if CI checks saved bytes and sources",
            "scope":"exploratory 4 already seen HGT sources; not holdout",
            "note":"stripes have physical file-seek; each ROI decompresses full 256-wide strip; JPEG2000/GDAL can internally cache",
            "records":results,
        },indent=2),encoding="utf-8")
    assert len(results)==4
    print("EXP68 DONE: real stripes vs full JPEG2000 on 4 known regions",flush=True)


if __name__=="__main__":
    main()
