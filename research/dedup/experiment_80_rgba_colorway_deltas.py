"""
BLKH Experiment 80 — exact authored within-shape Blue-relative RGBA deltas.

All RGBA including hidden RGB at zero alpha is byte-exact. The source
is the SAME preregistered real Kenney CC0 source and ordering from Exp79;
four fixed predictors, and the best of ALL seven existing Exp79 codecs.
NOT an independent dataset or artist-version chronology.
See research/dedup/BHUH_EXP80_RGBA_COLORWAY_DELTA_PROTOCOL.md.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import platform
import struct
import time

import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import zstandard as zstd

from experiment_78_real_cc0_rgba_sprites import (
    CODECS as REFERENCE,fetch_zip,sha,pack_assets,base_manifest,encode_case,
    decode_case,encode_webp,decode_rgba,zcompress,zdecompress,
)
from experiment_79_authored_colorways import (
    COLORS,SHAPES,EXPECTED_SHA256,authored_colorways,
)

METHODS=("xor_webp","xor_zstd","mod_webp","mod_zstd")
MAGIC=b"BL80D"
HEAD=struct.Struct("<5sBI")
MAX_FILE=40_000_000
MAX_METADATA=250_000


def assert_groups(items:list[dict])->None:
    if len(items) not in (32,64,96) or len(items)%len(COLORS):
        raise ValueError("Unexpected shape/color sample count")
    for g in range(0,len(items),4):
        group=items[g:g+4]
        w=group[0]["meta"]["width"]
        h=group[0]["meta"]["height"]
        s=group[0]["meta"]["shape_suffix"]
        if ([(x["meta"]["colorway"]) for x in group]!=list(COLORS)
                or any(x["meta"]["shape_suffix"]!=s for x in group)
                or any(x["meta"]["width"]!=w or x["meta"]["height"]!=h
                       for x in group)
                or any(x["rgba"].shape!=(h,w,4) for x in group)):
            raise ValueError("Author colorways not shape-aligned; no image substitutions allowed")


def difference(items:list[dict],method:str)->list[np.ndarray]:
    if method not in METHODS:
        raise ValueError("Unknown preregistered residual mode")
    assert_groups(items)
    produced=[]
    for i in range(0,len(items),4):
        base=items[i]["rgba"]
        produced.append(base.copy())
        for item in items[i+1:i+4]:
            src=item["rgba"]
            if method.startswith("xor"):
                transformed=np.bitwise_xor(src,base)
            else:
                transformed=((src.astype(np.uint16)-base.astype(np.uint16))&255).astype(np.uint8)
            produced.append(np.ascontiguousarray(transformed))
    return produced


def residual_atlas(items:list[dict],method:str)->np.ndarray:
    residual=difference(items,method)
    substituted=[{"meta":source["meta"],"rgba":rgba}
                 for source,rgba in zip(items,residual)]
    return pack_assets(substituted)[0]


def encode(items:list[dict],manifest:dict,method:str)->bytes:
    if method not in METHODS:
        raise ValueError("Invalid source delta scheme")
    arr=residual_atlas(items,method)
    if list(arr.shape)!=manifest["atlas_shape_rgba"]:
        raise ValueError("Residual layout disagrees with original atlas")
    stream=(encode_webp(arr) if method.endswith("webp")
            else zcompress(arr.tobytes()))
    meta=json.dumps({**manifest,"codec":method},sort_keys=True,
                    separators=(",",":")).encode()
    if not 0<len(meta)<MAX_METADATA:
        raise ValueError("Oversized colorway metadata")
    return HEAD.pack(MAGIC,1,len(meta))+meta+stream


def decode(encoded:bytes)->tuple[list[np.ndarray],dict]:
    if not HEAD.size<len(encoded)<=MAX_FILE:
        raise ValueError("Invalid colorway delta file size")
    magic,version,n=HEAD.unpack(encoded[:HEAD.size])
    if magic!=MAGIC or version!=1 or not 0<n<MAX_METADATA or n+HEAD.size>=len(encoded):
        raise ValueError("Truncated/invalid BL80 header")
    meta=json.loads(encoded[HEAD.size:HEAD.size+n])
    mode=meta.get("codec")
    if (mode not in METHODS or meta.get("collection_size") not in (32,64,96)
            or len(meta.get("sprites",[]))!=meta["collection_size"]
            or meta.get("colors_fixed_order")!=list(COLORS)
            or meta.get("atlas_shape_rgba")!=[
                meta.get("atlas_height"),meta.get("atlas_width"),4]):
        raise ValueError("Wrong colorway schema")
    shape=tuple(meta["atlas_shape_rgba"])
    if not all(isinstance(x,int) and 0<x<=4096 for x in shape[:2]):
        raise ValueError("Source image dimensions out of bounds")
    payload=encoded[HEAD.size+n:]
    if mode.endswith("webp"):
        transformed=decode_rgba(payload,shape)
    else:
        raw=zdecompress(payload,int(np.prod(shape)))
        if len(raw)!=int(np.prod(shape)):
            raise ValueError("Bad residual decoded byte count")
        transformed=np.frombuffer(raw,dtype=np.uint8).reshape(shape)
    sprites=meta["sprites"]
    images=[]
    for group_start in range(0,len(sprites),4):
        info=sprites[group_start:group_start+4]
        if ([s.get("colorway") for s in info]!=list(COLORS)
                or len({s.get("shape_suffix") for s in info})!=1):
            raise ValueError("Malformed ordered colorway references")
        shapes={(s["height"],s["width"]) for s in info}
        if len(shapes)!=1:
            raise ValueError("Malformed colorway dimensions")
        base=None
        for c,sprite in enumerate(info):
            x,y,w,h=(sprite[k] for k in ("x","y","width","height"))
            if (not all(isinstance(k,int) for k in (x,y,w,h))
                    or min(x,y)<0 or min(w,h)<1
                    or x+w>shape[1] or y+h>shape[0]):
                raise ValueError("Out-of-range sprite atlas source crop")
            block=np.ascontiguousarray(transformed[y:y+h,x:x+w])
            if c==0:
                original=block.copy()
                base=original
            else:
                assert base is not None
                original=(np.bitwise_xor(block,base)
                          if mode.startswith("xor")
                          else ((block.astype(np.uint16)+base.astype(np.uint16))&255).astype(np.uint8))
            if sha(original.tobytes())!=sprite["pixel_rgba_sha256"]:
                raise ValueError("Original exact RGBA sprite checksum mismatch")
            images.append(original)
    # Also validate padding and exactly reconstructed RGB/alpha atlas.
    original_atlas=np.zeros(shape,dtype=np.uint8)
    for original,sprite in zip(images,sprites):
        x,y,w,h=(sprite[k] for k in ("x","y","width","height"))
        original_atlas[y:y+h,x:x+w]=original
    if sha(original_atlas.tobytes())!=meta["atlas_rgba_sha256"]:
        raise ValueError("Atlas padding/source hash failure")
    return images,meta


def exercise(items:list[dict],meta:dict,folder:Path)->dict:
    atlas,_=pack_assets(items)
    folder.mkdir(parents=True,exist_ok=True)
    with io.BytesIO() as f:
        Image.fromarray(atlas,mode="RGBA").save(f,format="PNG")
        png=f.getvalue()
    (folder/"actual_original_atlas.png").write_bytes(png)
    scores={}
    for method in METHODS+REFERENCE:
        start=time.perf_counter_ns()
        if method in METHODS:
            raw=encode(items,meta,method)
        else:
            raw,_=encode_case(atlas,items,meta,method)
        encode_s=(time.perf_counter_ns()-start)/1e9
        target=folder/(method+(".bl80" if method in METHODS else ".bl78"))
        target.write_bytes(raw)
        start=time.perf_counter_ns()
        restored,decoded_meta=(decode(target.read_bytes()) if method in METHODS
                               else decode_case(target.read_bytes()))
        decode_s=(time.perf_counter_ns()-start)/1e9
        if decoded_meta!={**meta,"codec":method} or len(restored)!=len(items):
            raise AssertionError("Wrong saved source variant metadata/ordering")
        if not all(np.array_equal(image["rgba"],out)
                   for image,out in zip(items,restored)):
            raise AssertionError("Source RGBA alpha or hidden colors changed")
        scores[method]={
            "file":target.name,
            "bytes":len(raw),"sha256":sha(raw),
            "encoding_seconds":encode_s,"decoding_seconds":decode_s,
        }
    ours,our_mode=min((scores[m]["bytes"],m) for m in METHODS)
    base,baseline_mode=min((scores[m]["bytes"],m) for m in REFERENCE)
    return {
        "shape_count":len(items)//4,
        "color_variant_images":len(items),
        "manifest":meta,
        "source_atlas_png":"actual_original_atlas.png",
        "source_atlas_png_sha256":sha(png),
        "actual_encoded_files":scores,
        "best_new_differential":our_mode,"best_new_differential_bytes":ours,
        "best_unchanged_exp79_reference":baseline_mode,
        "best_unchanged_reference_bytes":base,
        "new_delta_smaller_than_best_preexisting_format":ours<base,
        "exact_rgba_reconstruction":True,
    }


def self_test():
    rng=np.random.default_rng(80)
    items=[]
    for shape_index in range(8):
        h,w=20+shape_index,28+shape_index
        base=rng.integers(0,256,(h,w,4),dtype=np.uint8)
        for i,c in enumerate(COLORS):
            if i==0:rgba=base
            else:
                rgba=base.copy()
                rgba[:,:,i-1]=(rgba[:,:,i-1].astype(np.uint16)+i*13)&255
                rgba=rgba.astype(np.uint8)
            with io.BytesIO() as f:
                Image.fromarray(rgba,mode="RGBA").save(f,format="PNG")
                encoded=f.getvalue()
            items.append({
                "rgba":rgba,"raw":encoded,
                "meta":{
                    "filename":f"PNG/{c}/Default/selftest_{shape_index}.png",
                    "colorway":c,"shape_suffix":f"selftest_{shape_index}.png",
                    "source_png_sha256":sha(encoded),
                    "pixel_rgba_sha256":sha(rgba.tobytes()),
                    "width":w,"height":h,
                    "alpha_transparent_pixel_count":int(np.count_nonzero(rgba[:,:,3]==0)),
                    "alpha_partial_pixel_count":int(np.count_nonzero(
                        (rgba[:,:,3]>0)&(rgba[:,:,3]<255)))
                },
            })
    atlas,layout=pack_assets(items)
    m=base_manifest("0"*64,1200,items,atlas,layout)
    m["variant_semantics"]="selftest"
    m["colors_fixed_order"]=list(COLORS)
    m["shape_suffixes"]=[f"selftest_{i}.png" for i in range(8)]
    assert_groups(items)
    for mode in METHODS:
        blob=encode(items,m,mode)
        restored,original=decode(blob)
        assert original=={**m,"codec":mode}
        assert all(np.array_equal(item["rgba"],result)
                   for item,result in zip(items,restored))
        try:
            decode(blob[:-1])
        except (ValueError,zstd.ZstdError,OSError):
            pass
        else:
            raise AssertionError("Truncated alpha-aware delta accepted")
    print("EXP80 SELF-TEST PASS: all 4 reversible colorway transforms, alpha/hidden RGB exact")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",default="/tmp/blkh_exp80")
    p.add_argument("--cache",default="/tmp/blkh_exp78_cache")
    a=p.parse_args()
    if a.self_test:
        self_test();return
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    data,source_hash=fetch_zip(Path(a.cache))
    if source_hash!=EXPECTED_SHA256:
        raise ValueError("Unregistered Kenney CC0 source ZIP change")
    all_items,inventory=authored_colorways(data,source_hash,root)
    results=[]
    for shapes in SHAPES:
        subset=all_items[:shapes*4]
        assert_groups(subset)
        atlas,layout=pack_assets(subset)
        shared=base_manifest(source_hash,len(data),subset,atlas,layout)
        shared["variant_semantics"]="4 genuine Kenney-authored palette/colorways of each selected UI shape, NOT chronological revision files"
        shared["colors_fixed_order"]=list(COLORS)
        shared["shape_suffixes"]=inventory["selected_first_24_suffixes"][:shapes]
        row=exercise(subset,shared,root/f"shapes_{shapes:02d}")
        results.append(row)
        print("EXP80",shapes,"authored forms",shapes*4,"assets",
              "delta",row["best_new_differential"],
              row["best_new_differential_bytes"],
              "strongest_existing",row["best_unchanged_exp79_reference"],
              row["best_unchanged_reference_bytes"],
              "win",row["new_delta_smaller_than_best_preexisting_format"],flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":80,
            "protocol":"research/dedup/BHUH_EXP80_RGBA_COLORWAY_DELTA_PROTOCOL.md",
            "source_sha256":source_hash,
            "source":"Kenney UI CC0 original author variations",
            "color_order":list(COLORS),
            "shape_counts":list(SHAPES),
            "new_modes":list(METHODS),
            "reference_modes":list(REFERENCE),
            "source_reused_from_exp79_not_new_independent_holdout":True,
            "software":{"numpy":np.__version__,"pillow":PIL_VERSION,
                        "zstandard":zstd.__version__,"python":platform.python_version()},
            "results":results,
        },indent=2),encoding="utf8")
    assert len(results)==3
    print("EXP80 COMPLETED: 33 exact-lossless full files across three correlated artist colorway groups",flush=True)

if __name__=="__main__":
    main()
