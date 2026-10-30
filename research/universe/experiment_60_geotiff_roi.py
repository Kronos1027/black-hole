"""
Exp 60 — DEM ROI comparison vs REAL tiled GeoTIFF (DEFLATE, ZSTD, LZW).

Same original int16 terrain, 32×32 independently decoded blocks.
BH58 gets geospatial metadata inside a counted BH60 container; GeoTIFF files
carry EPSG:4326, affine transform, nodata, source ID and vertical unit.

Scientific boundary:
- Compare complete on-disk byte size and end-to-end open+window-read time.
- GDAL internally manages TIFF I/O and cache: DO NOT imply physical TIFF bytes
  read are exactly one compressed tile or claim cold-storage latency.
- GeoTIFF raster window is 32×32; BH60 seeks physically to one block.
- Query 24 fixed positions × 3 repeats, rotating codec order.
- Two source arrangements per 6 HGT source tiles (4 previous + 2 held out).
- True standalone metadata; only lossless signed int16 meter elevations.

Usage:
  python research/universe/experiment_60_geotiff_roi.py --self-test
  python research/universe/experiment_60_geotiff_roi.py --out /tmp/blkh_exp60
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import time

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from experiment_56_reversible_dem import CORE, HELDOUT, source_hgt
from experiment_59_independent_roi_regions import REGIONS as MORE_REGIONS, read_holdout
from experiment_57_random_access_dem import source_cases
from experiment_58_vectorized_decoder import (
    build_indexed, block_decode, decode_indexed, read_index,
    INDEX_HEADER, ENTRY
)

REGIONS = (*CORE, *HELDOUT, *MORE_REGIONS)
BH_MAGIC = b"BHG60"
PRELUDE = struct.Struct("<5sBI")
ROI_SIDE = 32
HGT_SAMPLE_COUNT = 3600
MAX_META = 16384
CODECS = ("bh60", "tiff_deflate", "tiff_zstd", "tiff_lzw")


def hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def georeference(region: str) -> Affine:
    m = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})", region)
    if not m:
        raise ValueError("Expected SRTM region ID, e.g. N37W119")
    lat = int(m.group(2))*(1 if m.group(1)=="N" else -1)
    lon = int(m.group(4))*(1 if m.group(3)=="E" else -1)
    delta = 1/HGT_SAMPLE_COUNT
    # HGT samples are pixel centers, not pixel corners.
    # All source_cases use crop row=1200 col=1300 of the same HGT tile.
    x = lon + 1300*delta - delta/2
    y = lat + 1 - 1200*delta + delta/2
    return Affine(delta, 0, x, 0, -delta, y)


def metadata(region: str, img: np.ndarray) -> dict:
    transform = georeference(region)
    return {
        "crs": "EPSG:4326",
        "transform": [float(x) for x in transform[:6]],
        "nodata": -32768,
        "sample_unit": "meter",
        "dtype": "int16",
        "region": region,
        "image_shape": list(img.shape),
        "block_side": ROI_SIDE,
        "source_crop_row_col": [1200,1300],
        "HGT_pixel_center": True,
    }


def encode_bh60(image: np.ndarray, info: dict) -> bytes:
    if image.dtype != np.int16 or image.ndim != 2 or any(
        n % ROI_SIDE for n in image.shape
    ):
        raise ValueError("Only 2D int16, tile-aligned crops")
    obj=json.dumps(info,sort_keys=True,separators=(",",":")).encode()
    if not (0<len(obj)<MAX_META):
        raise ValueError("Oversized metadata")
    body=build_indexed(image, ROI_SIDE)
    return PRELUDE.pack(BH_MAGIC,1,len(obj))+obj+body


def read_bh60_meta(handle) -> tuple[dict,int]:
    head=handle.read(PRELUDE.size)
    if len(head)!=PRELUDE.size:
        raise ValueError("Truncated BH60 metadata")
    magic,version,n=PRELUDE.unpack(head)
    if magic!=BH_MAGIC or version!=1 or not 0<n<MAX_META:
        raise ValueError("Unsupported BH60 container")
    blob=handle.read(n)
    if len(blob)!=n:
        raise ValueError("Truncated BH60 JSON")
    info=json.loads(blob)
    assert isinstance(info,dict)
    keys=("crs","transform","nodata","sample_unit","dtype","region","image_shape","block_side")
    if any(k not in info for k in keys):
        raise ValueError("Missing geospatial metadata")
    if info["crs"]!="EPSG:4326" or info["dtype"]!="int16" or info["block_side"]!=ROI_SIDE:
        raise ValueError("Unexpected geospatial schema")
    if len(info["transform"])!=6:
        raise ValueError("Corrupt affine metadata")
    return info,PRELUDE.size+n


def bh60_decode_full(path:Path) -> tuple[np.ndarray,dict]:
    with path.open("rb") as f:
        info, offset=read_bh60_meta(f)
        arr=decode_indexed(f.read())
    if list(arr.shape)!=info["image_shape"]:
        raise ValueError("Geospatial shape mismatch")
    return arr,info


def bh60_window(path:Path, y:int, x:int) -> tuple[np.ndarray,dict]:
    size=path.stat().st_size
    with path.open("rb") as f:
        info,offset=read_bh60_meta(f)
        head=f.read(INDEX_HEADER.size)
        if len(head)!=INDEX_HEADER.size:
            raise ValueError("Truncated inner header")
        idx_size=INDEX_HEADER.unpack(head)[-1]
        if idx_size>2_000_000:
            raise ValueError("Invalid index bytes")
        idx=f.read(idx_size)
        (h,w,side,entries)=read_index(head,idx,size-offset)
        if not (0<=y<h//side and 0<=x<w//side):
            raise IndexError("Requested tile outside HGT")
        j=y*(w//side)+x
        start,length=entries[j]
        f.seek(offset+INDEX_HEADER.size+idx_size+start)
        chunk=f.read(length)
    if len(chunk)!=length:
        raise ValueError("Short ROI read")
    result=block_decode(chunk)
    if result.shape!=(ROI_SIDE,ROI_SIDE):
        raise ValueError("Bad ROI shape")
    return result,info


def encode_tiff(path:Path, image:np.ndarray, info:dict, codec:str) -> None:
    options = {
        "driver":"GTiff","width":image.shape[1],"height":image.shape[0],
        "count":1,"dtype":"int16","tiled":True,"blockxsize":ROI_SIDE,
        "blockysize":ROI_SIDE,"compress":codec,"predictor":2,
        "crs":info["crs"],"transform":Affine(*info["transform"]),
        "nodata":info["nodata"],"BIGTIFF":"IF_SAFER",
        "NUM_THREADS":"1",
    }
    if codec=="deflate":
        options["zlevel"]=9
    if codec=="zstd":
        options["zstd_level"]=9
    with rasterio.open(path,"w",**options) as dst:
        dst.write(image,1)
        dst.update_tags(
            region=info["region"],vertical_unit=info["sample_unit"],
            HGT_pixel_center="True",
        )


def verify_tiff(path:Path, image:np.ndarray, info:dict) -> None:
    with rasterio.open(path) as ds:
        recovered=ds.read(1)
        if ds.crs.to_string()!=info["crs"] or ds.nodata!=info["nodata"]:
            raise AssertionError("GeoTIFF CRS/nodata not retained")
        if any(abs(a-b)>1e-10 for a,b in zip(ds.transform[:6],info["transform"])):
            raise AssertionError("GeoTIFF affine not retained")
        if ds.tags().get("region")!=info["region"]:
            raise AssertionError("GeoTIFF source tile tag missing")
        if ds.tags().get("vertical_unit")!=info["sample_unit"]:
            raise AssertionError("GeoTIFF meter unit tag missing")
        if ds.block_shapes[0]!=(ROI_SIDE,ROI_SIDE):
            raise AssertionError("GeoTIFF not tiled correctly")
        if not np.array_equal(recovered,image):
            raise AssertionError("GeoTIFF lossless reconstruction failed")


def roi_tiff(path:Path, y:int, x:int) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1, window=Window(x*ROI_SIDE,y*ROI_SIDE,ROI_SIDE,ROI_SIDE))


def benchmark_one(region:str, case:str, image:np.ndarray,
                  target:Path, positions:int, repeats:int) -> dict:
    info=metadata(region,image)
    target.mkdir(parents=True,exist_ok=True)
    paths={}
    encoding={}
    start=time.perf_counter()
    raw=encode_bh60(image,info)
    p=target/"geographic_lossless.bh60"
    p.write_bytes(raw)
    encoding["bh60"]=time.perf_counter()-start
    recovered,decoded_info=bh60_decode_full(p)
    if not np.array_equal(image,recovered) or info!=decoded_info:
        raise AssertionError("BH60 georeferenced round-trip failed")
    paths["bh60"]=p
    for key,codec in [
        ("tiff_deflate","deflate"),("tiff_zstd","zstd"),("tiff_lzw","lzw"),
    ]:
        p=target/(key+".tif")
        start=time.perf_counter()
        encode_tiff(p,image,info,codec)
        encoding[key]=time.perf_counter()-start
        verify_tiff(p,image,info)
        paths[key]=p

    h,w=image.shape
    n=(h//ROI_SIDE)*(w//ROI_SIDE)
    rng=np.random.default_rng(600060)
    ids=sorted(int(i) for i in rng.choice(n,size=min(n,positions),replace=False))
    observations={key:[] for key in CODECS}
    reads_meta=[]
    cols=w//ROI_SIDE
    for idx in ids:
        y,x=divmod(idx,cols)
        expected=image[y*ROI_SIDE:(y+1)*ROI_SIDE,
                       x*ROI_SIDE:(x+1)*ROI_SIDE]
        for repetition in range(repeats):
            order=CODECS[repetition%len(CODECS):]+CODECS[:repetition%len(CODECS)]
            for key in order:
                start=time.perf_counter()
                if key=="bh60":
                    output,bmeta=bh60_window(paths[key],y,x)
                else:
                    output=roi_tiff(paths[key],y,x)
                elapsed=time.perf_counter()-start
                if not np.array_equal(expected,output):
                    raise AssertionError("ROI differs from original integer meters")
                observations[key].append(elapsed)
    sizes={key:paths[key].stat().st_size for key in CODECS}
    median={key:float(np.median(observations[key])*1000) for key in CODECS}
    p95={key:float(np.percentile(observations[key],95)*1000) for key in CODECS}
    sha={key:hexdigest(paths[key].read_bytes()) for key in CODECS}
    row={
        "region":region,"case":case,"shape":list(image.shape),
        "block_side":ROI_SIDE,"roi_block_positions":ids,
        "roi_repeats":repeats,"num_roi_samples_per_codec":len(ids)*repeats,
        "metadata":info,"all_lossless_verified":True,"max_error_m":0,
        "complete_file_bytes":sizes,
        "file_sha256":sha,
        "encoded_file":{key:paths[key].name for key in CODECS},
        "input_sha256":hexdigest(image.tobytes()),
        "encode_seconds":encoding,
        "roi_open_read_decode_median_ms":median,
        "roi_open_read_decode_p95_ms":p95,
    }
    print(region,case,"bytes",sizes,"med_ms",
          {k:round(v,3) for k,v in median.items()},flush=True)
    return row


def self_test():
    region="N37W119"
    img=(np.arange(128*192,dtype=np.int32)%999-470).astype(np.int16).reshape(128,192)
    info=metadata(region,img)
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        path=Path(root)/"data.bh60"
        path.write_bytes(encode_bh60(img,info))
        result,met=bh60_decode_full(path)
        assert np.array_equal(result,img) and met==info
        tile,_=bh60_window(path,1,2)
        assert np.array_equal(tile,img[32:64,64:96])
        try:
            with path.open("wb") as f:f.write(b"bad")
            bh60_decode_full(path)
        except ValueError:
            pass
        else: raise AssertionError("Truncated metadata was accepted")
        for codec in ("deflate","zstd","lzw"):
            tif=Path(root)/(codec+".tif")
            encode_tiff(tif,img,info,codec)
            verify_tiff(tif,img,info)
            assert np.array_equal(roi_tiff(tif,1,2),img[32:64,64:96])
    print("EXP60 SELF-TEST PASS: GeoTIFF CRS/transform/height and counted BH60 metadata")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--out",default="/tmp/blkh_exp60")
    ap.add_argument("--cache",default="/tmp/blkh_exp60_cache")
    ap.add_argument("--positions",type=int,default=24)
    ap.add_argument("--repeats",type=int,default=3)
    args=ap.parse_args()
    if args.self_test: self_test();return
    if not (4<=args.positions<=64 and 2<=args.repeats<=12):
        ap.error("Bad ROI sample parameters")
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    sources={}
    for region in REGIONS:
        if region in MORE_REGIONS:
            raw,source=read_holdout(region,Path(args.cache))
        else:
            raw,source=source_hgt(region,Path(args.cache))
        sources[region]=source
        for case,image in source_cases(raw):
            row=benchmark_one(region,case,image,root/region/case,
                              args.positions,args.repeats)
            rows.append(row)
        (root/"results.json").write_text(json.dumps({
            "experiment":60,"status":"CI pending until externally verified",
            "design":"32px lossless ROI window + complete EPSG:4326, nodata, meter units",
            "hardware_caveat":"Host-level GDAL caching, CI CPU, no physical TIFF read accounting",
            "regions":list(REGIONS),"sources":sources,"records":rows,
        },indent=2),encoding="utf-8")
    assert len(rows)==12
    print("EXP60 COMPLETED: 6 REAL HGT REGIONS × 2 GEOMETRIES × 4 CODECS",flush=True)


if __name__=="__main__":
    main()
