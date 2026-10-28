"""
BLKH Experiment 67 — can GDAL read ROI from whole-image JPEG2000?

Experiment 66 established that the full JPEG2000 codestream often occupies
40%+ fewer bytes than individually indexed BH65/J2K. Do NOT assume that a
single J2K codestream cannot support window reads. This experiment tests
GDAL JP2OpenJPEG (or compatible available driver) Window reads on the exact
source and same 32x32 requested ROI sequences.

IMPORTANT FAIRNESS: A .jp2 is extracted verbatim from Exp66's *counted*
geospatial wrapper (JSON CRS, affine, unit and signed int16 origin). We count
the ENTIRE wrapper size, not just the raw JP2 codestream. The standalone
.jp2 is for GDAL access; the wrapper is essential for reconstructing signed
elevations/georeferencing. This prototype is NOT a standard georeferenced JP2.
All reconstructed ROI samples are verified exactly against signed int16 DEM.

Four Exp66 sources are no longer unseen. Same work, no parameter tuning.
This measures warm GDAL cache vs uncached BH65; it is a system/API test, not
a compressor-algorithm speed proof.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import rasterio
from rasterio.windows import Window

from experiment_66_preregistered_holdout import (
    REGIONS, CODECS, source_hgt, compare_one,
    OpenJ2KGeo, PRELUDE, ROI_READS, digest,
)
from experiment_65_hybrid_pareto import OpenHybrid

MODES=("bh65_indexed", "j2k_indexed", "j2k_full_window", "tiff_zstd")


def split_full_j2k_wrapper(complete: bytes) -> tuple[dict, int, bytes]:
    if len(complete)<PRELUDE.size+8:
        raise ValueError("Truncated JPEG2000 georeferenced wrapper")
    magic,version,n=PRELUDE.unpack(complete[:PRELUDE.size])
    if magic!=b"J2F66" or version!=1 or not 0<n<16384:
        raise ValueError("Not an Exp66 full JPEG2000 georeferenced wrapper")
    meta=json.loads(complete[PRELUDE.size:PRELUDE.size+n])
    start=PRELUDE.size+n
    lo,hi=struct.unpack("<ii",complete[start:start+8])
    if not isinstance(meta,dict) or meta.get("image_shape")!=[256,256]:
        raise ValueError("Invalid stored JPEG2000 geographic metadata")
    if meta.get("crs")!="EPSG:4326" or meta.get("sample_unit")!="meter":
        raise ValueError("Missing correct horizontal CRS or height units")
    if lo>hi:
        raise ValueError("Invalid signed image origin")
    jpeg=complete[start+8:]
    if len(jpeg)<12:
        raise ValueError("Truncated JPEG2000 codestream")
    return meta,lo,jpeg


def read_full_j2k(ds, origin:int, row:int,col:int) -> np.ndarray:
    arr=ds.read(1,window=Window(col*32,row*32,32,32))
    restored=arr.astype(np.int32)+origin
    if restored.shape!=(32,32) or np.any(restored>32767) or np.any(restored<-32768):
        raise ValueError("Invalid JPEG2000 signed ROI")
    return restored.astype(np.int16)


def benchmark(region:str,original:np.ndarray,source:dict,root:Path)->dict:
    # Recreate the predeclared Experiment 66 exact serialized candidates, for
    # losslessness, sha256, same geometry/metadata, and comparable archive sizes.
    trial=compare_one(region,original,source,root)
    directory=root/region
    full_bytes=(directory/trial["full_j2k_nonindexed_file"]).read_bytes()
    info,lo,jpeg=split_full_j2k_wrapper(full_bytes)
    if info!=trial["geodata"]:
        raise AssertionError("Unfair J2K metadata mismatch")
    jp2=directory/"j2k_full_gdal_decoding.jp2"
    jp2.write_bytes(jpeg)
    samples={k:[] for k in MODES}
    with ExitStack() as stack:
        bh=stack.enter_context(OpenHybrid(directory/trial["file"]["bh65_fixed20"]))
        indexed=stack.enter_context(OpenJ2KGeo(directory/trial["file"]["j2k_indexed"]))
        jp2ds=stack.enter_context(rasterio.open(jp2))
        tiff=stack.enter_context(rasterio.open(directory/trial["file"]["tiff_zstd"]))
        if (jp2ds.width,jp2ds.height,jp2ds.count)!=(256,256,1):
            raise ValueError("GDAL cannot decode JPEG2000 source geometry")
        whole=jp2ds.read(1).astype(np.int32)+lo
        if not np.array_equal(whole,original):
            raise AssertionError("GDAL JPEG2000 full image not exactly lossless")
        if not np.array_equal(tiff.read(1),original):
            raise AssertionError("GeoTIFF not exactly lossless")
        ds_driver=jp2ds.driver
        jp2_block_shapes=jp2ds.block_shapes
        # Full-GDAL vs indexed codec read each query, same continuous handles.
        # GDAL may cache reconstructed blocks, whereas BH65 does not.
        for t,tile_id in enumerate(ROI_READS):
            y,x=divmod(tile_id,8)
            reference=original[y*32:(y+1)*32,x*32:(x+1)*32]
            order=MODES[t%len(MODES):]+MODES[:t%len(MODES)]
            for mode in order:
                start=time.perf_counter_ns()
                if mode=="bh65_indexed":
                    rec=bh.read(y,x)
                elif mode=="j2k_indexed":
                    rec=indexed.read(y,x)
                elif mode=="j2k_full_window":
                    rec=read_full_j2k(jp2ds,lo,y,x)
                else:
                    rec=tiff.read(1,window=Window(x*32,y*32,32,32))
                ms=(time.perf_counter_ns()-start)/1e6
                if not np.array_equal(rec,reference):
                    raise AssertionError("Window read not exactly original heights")
                samples[mode].append(ms)
    return {
        "region":region,"source":source,
        "input_sha256":digest(original.tobytes()),
        "counted_geospatial_bytes":len(full_bytes)-len(jpeg),
        "full_j2k_wrapper_bytes":len(full_bytes),
        "jp2_raw_bytes_for_gdal_only":len(jpeg),
        "full_j2k_sha256":digest(full_bytes),
        "gdal_driver":ds_driver,
        "jp2_block_shapes":jp2_block_shapes,
        "files":{
            "j2k_full_wrapper":trial["full_j2k_nonindexed_file"],
            "j2k_full_extracted":jp2.name,
            "bh65_fixed20":trial["file"]["bh65_fixed20"],
            "j2k_indexed":trial["file"]["j2k_indexed"],
            "tiff_zstd":trial["file"]["tiff_zstd"],
        },
        "complete_size_bytes":{
            "bh65_indexed":trial["sizes_bytes"]["bh65_fixed20"],
            "j2k_indexed":trial["sizes_bytes"]["j2k_indexed"],
            "j2k_full_window":len(full_bytes),
            "tiff_zstd":trial["sizes_bytes"]["tiff_zstd"],
        },
        "roi_mean_ms":{k:float(np.mean(v)) for k,v in samples.items()},
        "roi_median_ms":{k:float(np.median(v)) for k,v in samples.items()},
        "roi_p95_ms":{k:float(np.percentile(v,95)) for k,v in samples.items()},
        "roi_count":len(ROI_READS),
        "exact_lossless":True,
        "height_max_error_m":0,
        "full_j2k_has_direct_gdal_window_access":True,
        "full_j2k_dominates_bh65_size_and_mean_roi":(
            len(full_bytes)<trial["sizes_bytes"]["bh65_fixed20"]
            and np.mean(samples["j2k_full_window"])<=np.mean(samples["bh65_indexed"])
        ),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="/tmp/blkh_exp67")
    p.add_argument("--cache",default="/tmp/blkh_exp67_cache")
    args=p.parse_args()
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for region in REGIONS:
        img,source=source_hgt(region,Path(args.cache))
        result=benchmark(region,img,source,root)
        rows.append(result)
        print(region,"GDAL driver",result["gdal_driver"],
              "bytes",result["complete_size_bytes"],
              "window_mean_ms",{k:round(v,4) for k,v in result["roi_mean_ms"].items()},
              "full_J2K_dominates",result["full_j2k_dominates_bh65_size_and_mean_roi"],
              flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":67,
            "regions":list(REGIONS),
            "status":"reproduced only after independent CI verification",
            "scientific_goal":"verify actual GDAL window access on full J2K, not assume full image is non-ROI",
            "limitation":"wrapped JPEG2000 requires custom georeferenced signed-height metadata; JP2 itself not standalone GIS source",
            "cache_note":"GDAL full-JP2 windows can be cached unlike no-cache BH65; performance remains an API comparison",
            "results":rows,
        },indent=2),encoding="utf-8")
    assert len(rows)==4
    print("EXP67 DONE: FULL IMAGE JPEG2000 WINDOW ACCESS CONFIRMED",flush=True)


if __name__=="__main__":
    main()
