"""
BLKH Experiment 82 — historical sprite edits across 3 real Pixelorama Git commits.

Actual modified same-path PNGs fetched at fixed first parent and commit SHA;
not synthesized colorways. All file bytes and decoded uint8 RGBA source pixel
hashes counted, exact RGBA under alpha=0 kept. Aggregate + per-commit
comparisons to strong original-PNG ZSTD and spatial atlas baselines.

Pre-registered: research/dedup/BHUH_EXP82_GIT_REAL_REVISIONS_PROTOCOL.md
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import platform
import struct
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import zstandard as zstd

from experiment_78_real_cc0_rgba_sprites import (
    TILE,MAX_ROW_WIDTH,sha,zcompress,zdecompress,
    encode_webp,decode_rgba,dictionary,decode_dict,
    encode_individual,decode_individual,
)

REPOSITORY="Orama-Interactive/Pixelorama"
REPOSITORY_URL="https://github.com/Orama-Interactive/Pixelorama"
LICENSE="MIT (Pixelorama repository root LICENSE)"
COMMIT_GROUPS=(
    ("pan","9f8c9807e789bf476f95ac7045409e943b5fc328",tuple(
        f"assets/graphics/{color}/tools/{name}.png"
        for color in ("blue_themes","dark_themes","light_themes")
        for name in ("pan","pan_l","pan_l_r","pan_r")
    )+("assets/graphics/cursor_icons/pan_cursor.png",)),
    ("curve","9611bf72631c3eaed8e4f0281a6628e069ecaa63",(
        "assets/graphics/tools/3dshapeedit.png",
        "assets/graphics/tools/cursors/3dshapeedit.png",
        "assets/graphics/tools/cursors/curvetool.png",
        "assets/graphics/tools/curvetool.png",
    )),
    ("isometric","13c2af3a2a53bfa1439d7a6ca0e3ddbd136a91b5",(
        "assets/graphics/tools/cursors/isometricboxtool.png",
        "assets/graphics/tools/isometricboxtool.png",
    )),
)
CANDIDATES=("xor_webp","xor_zstd","mod_webp","mod_zstd")
REFERENCES=("atlas_png","atlas_webp","atlas_raw_zstd",
            "original_png_zstd","optimized_png_zstd",
            "tile16_webp","tile16_zstd")
ALL_METHODS=CANDIDATES+REFERENCES
HEADER=struct.Struct("<5sBI")
MAGIC=b"BL82R"
MAX_SOURCE=5_000_000
MAX_ARCHIVE=25_000_000
MAX_METADATA=220_000


def request_bytes(url:str,max_length:int)->bytes:
    req=urllib.request.Request(url,headers={"User-Agent":"BLKH-exp81-reproducible-research/1.0"})
    with urllib.request.urlopen(req,timeout=45) as r:
        raw=r.read(max_length+1)
    if not 0<len(raw)<=max_length:
        raise ValueError("Source file/metadata outside preregistered safety bounds")
    return raw


def get_parent(commit_sha:str)->str:
    url=f"https://api.github.com/repos/{REPOSITORY}/commits/{commit_sha}"
    data=json.loads(request_bytes(url,300_000))
    if data.get("sha")!=commit_sha or len(data.get("parents",[]))!=1:
        raise ValueError("Pinned historical commit has ambiguous/invalid first parent")
    parent=data["parents"][0]["sha"]
    if not isinstance(parent,str) or len(parent)!=40:
        raise ValueError("Invalid historical source parent SHA")
    return parent


def source_png(ref:str,path:str)->tuple[bytes,np.ndarray]:
    url=(f"https://raw.githubusercontent.com/{REPOSITORY}/{ref}/"
         +urllib.parse.quote(path,safe="/"))
    raw=request_bytes(url,MAX_SOURCE)
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("GitHub historical file is not PNG")
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        if im.width<1 or im.height<1 or max(im.width,im.height)>1024:
            raise ValueError("Historical icon invalid geometry")
        pic=np.asarray(im.convert("RGBA"),dtype=np.uint8)
    return raw,pic


def discover(root:Path)->tuple[list[dict],dict]:
    cached=root/"sources"
    cached.mkdir(parents=True,exist_ok=True)
    items=[]
    per_commit={}
    for label,commit,paths in COMMIT_GROUPS:
        parent=get_parent(commit)
        count_pixel_edits=0
        status=[]
        for index,path in enumerate(sorted(paths)):
            pair=[]
            for role,ref in (("old",parent),("new",commit)):
                raw,rgba=source_png(ref,path)
                fn=cached/f"{label}_{index:02}_{role}.png"
                fn.write_bytes(raw)
                meta={
                    "source_repository":REPOSITORY,
                    "commit_label":label,
                    "commit_sha":commit,
                    "parent_sha":parent,
                    "file_path":path,
                    "pair_index":index,
                    "role":role,
                    "filename":f"{label}:{index}:{role}:{path}",
                    "source_png_sha256":sha(raw),
                    "pixel_rgba_sha256":sha(rgba.tobytes()),
                    "width":int(rgba.shape[1]),
                    "height":int(rgba.shape[0]),
                    "alpha_zero_pixels":int(np.count_nonzero(rgba[:,:,3]==0)),
                }
                pair.append({"rgba":rgba,"raw":raw,"meta":meta})
            same=(pair[0]["rgba"].shape==pair[1]["rgba"].shape and
                  np.array_equal(pair[0]["rgba"],pair[1]["rgba"]))
            if not same:count_pixel_edits+=1
            status.append({"path":path,"pixels_identical":same})
            items.extend(pair)
        per_commit[label]={
            "commit":commit,
            "first_parent":parent,
            "png_pairs_modified_in_git":len(paths),
            "png_pairs_really_changed_rgba":count_pixel_edits,
            "status":status,
        }
    inventory={
        "repository":REPOSITORY,"repository_license":LICENSE,
        "revision_groups":per_commit,"source_png_count":len(items),
        "real_file_pair_count":len(items)//2,
        "pixel_changed_pairs":sum(g["png_pairs_really_changed_rgba"]
                                  for g in per_commit.values()),
    }
    (root/"real_git_source_inventory.json").write_text(json.dumps(inventory,indent=2))
    if len(items)!=38:
        raise AssertionError("Frozen 19 pairs / 38 PNG versions violated")
    return items,inventory


def pack(items:list[dict])->tuple[np.ndarray,list[dict]]:
    x=y=0;row_height=0;max_width=0
    positions=[]
    for item in items:
        w,h=item["meta"]["width"],item["meta"]["height"]
        pw=((w+TILE-1)//TILE)*TILE
        ph=((h+TILE-1)//TILE)*TILE
        if pw>MAX_ROW_WIDTH:raise ValueError("One true source icon too wide")
        if x+pw>MAX_ROW_WIDTH:
            y+=row_height
            x=0;row_height=0
        positions.append({"x":x,"y":y,"width":w,"height":h,
                          "padded_width":pw,"padded_height":ph,
                          **item["meta"]})
        row_height=max(row_height,ph)
        max_width=max(max_width,x+pw)
        x+=pw
    width=max_width;height=y+row_height
    if width<=0 or height<=0 or width*height>3_000_000:
        raise ValueError("Actual historical source atlas too big")
    atlas=np.zeros((height,width,4),dtype=np.uint8)
    for item,position in zip(items,positions):
        x,y,w,h=(position[k] for k in ("x","y","width","height"))
        atlas[y:y+h,x:x+w]=item["rgba"]
    return atlas,positions


def residual_assets(items:list[dict],mode:str)->list[dict]:
    if mode not in CANDIDATES or len(items)%2:
        raise ValueError("Not a frozen pair delta method")
    out=[]
    for i in range(0,len(items),2):
        before,after=items[i:i+2]
        out.append({"meta":before["meta"],"rgba":before["rgba"]})
        old=before["rgba"]
        new=after["rgba"]
        reference=np.zeros_like(new)
        h=min(old.shape[0],new.shape[0]);w=min(old.shape[1],new.shape[1])
        reference[:h,:w]=old[:h,:w]
        values=(np.bitwise_xor(new,reference) if mode.startswith("xor_")
                else ((new.astype(np.uint16)-reference.astype(np.uint16))&255).astype(np.uint8))
        out.append({"meta":after["meta"],"rgba":values})
    return out


def manifest_for(items:list[dict],inventory:dict)->dict:
    atlas,locations=pack(items)
    return {
        "source_repository":REPOSITORY,
        "source_repo_url":REPOSITORY_URL,
        "source_license":LICENSE,
        "group_commit_provenance":{
            key:{"sha":v["commit"],"parent":v["first_parent"],
                 "png_pairs_modified_in_git":v["png_pairs_modified_in_git"],
                 "png_pairs_really_changed_rgba":v["png_pairs_really_changed_rgba"]}
            for key,v in inventory["revision_groups"].items()
        },
        "source_png_count":len(items),
        "historical_pair_count":len(items)//2,
        "target_rgba_definition":"exact decoded RGBA uint8, including hidden RGB; original PNG bytes in baseline",
        "paired_source_order":"group pan, curve, isometric; sorted paths each; old-parent then new-commit",
        "atlas_shape":list(atlas.shape),
        "atlas_sha256":sha(atlas.tobytes()),
        "source_sprites":locations,
    }


def encode(items:list[dict],meta:dict,method:str)->tuple[bytes,dict]:
    original_atlas,_=pack(items)
    if list(original_atlas.shape)!=meta["atlas_shape"]:
        raise ValueError("Original atlas shape changed")
    extras={}
    if method in CANDIDATES:
        transform,_=pack(residual_assets(items,method))
        if transform.shape!=original_atlas.shape:
            raise ValueError("Changed atlas geometry")
        content=(encode_webp(transform)
                 if method.endswith("webp") else zcompress(transform.tobytes()))
    elif method=="atlas_png":
        with io.BytesIO() as b:
            Image.fromarray(original_atlas,mode="RGBA").save(
                b,format="PNG",optimize=True,compress_level=9)
            content=b.getvalue()
    elif method=="atlas_webp":
        content=encode_webp(original_atlas)
    elif method=="atlas_raw_zstd":
        content=zcompress(original_atlas.tobytes())
    elif method in ("original_png_zstd","optimized_png_zstd"):
        content=encode_individual(items,method=="optimized_png_zstd")
    elif method in ("tile16_webp","tile16_zstd"):
        content,extras=dictionary(original_atlas,method)
    else:
        raise ValueError("Method not preregistered")
    metadata=json.dumps({**meta,"codec":method},sort_keys=True,
                        separators=(",",":")).encode()
    if not 0<len(metadata)<MAX_METADATA:
        raise ValueError("Full provenance manifest too large")
    return HEADER.pack(MAGIC,1,len(metadata))+metadata+content,extras


def decode(raw:bytes)->tuple[list[np.ndarray],dict]:
    if not HEADER.size<len(raw)<=MAX_ARCHIVE:
        raise ValueError("Full historical revision archive length invalid")
    magic,version,meta_length=HEADER.unpack(raw[:HEADER.size])
    if magic!=MAGIC or version!=1 or not 0<meta_length<MAX_METADATA:
        raise ValueError("Wrong historical PNG revision format")
    limit=HEADER.size+meta_length
    if limit>=len(raw):raise ValueError("Missing historical sprite content")
    meta=json.loads(raw[HEADER.size:limit])
    mode=meta.get("codec");sprites=meta.get("source_sprites")
    if (mode not in ALL_METHODS or not isinstance(sprites,list)
            or len(sprites)<2 or len(sprites)%2 or
            meta.get("source_png_count")!=len(sprites) or
            meta.get("historical_pair_count")!=len(sprites)//2):
        raise ValueError("Invalid revision-pair metadata")
    shape=tuple(meta["atlas_shape"])
    if len(shape)!=3 or shape[-1]!=4 or shape[0]%16 or shape[1]%16:
        raise ValueError("Invalid source color atlas shape")
    if shape[0]*shape[1]>3_000_000:
        raise ValueError("Malformed oversized historical atlas")
    content=raw[limit:]
    if mode in CANDIDATES:
        if mode.endswith("webp"):
            encoded=decode_rgba(content,shape)
        else:
            dec=zdecompress(content,int(np.prod(shape)))
            if len(dec)!=int(np.prod(shape)):
                raise ValueError("Corrupt RGBA delta content")
            encoded=np.frombuffer(dec,dtype=np.uint8).reshape(shape)
    elif mode in ("atlas_webp","atlas_png"):
        encoded=decode_rgba(content,shape)
    elif mode=="atlas_raw_zstd":
        dec=zdecompress(content,int(np.prod(shape)))
        if len(dec)!=int(np.prod(shape)):raise ValueError("Wrong RGBA atlas byte count")
        encoded=np.frombuffer(dec,dtype=np.uint8).reshape(shape)
    elif mode in ("tile16_webp","tile16_zstd"):
        encoded=decode_dict(content,mode,shape)
    elif mode in ("original_png_zstd","optimized_png_zstd"):
        wrapped={**meta,"sprites":sprites}
        result=decode_individual(content,wrapped)
        encoded=np.zeros(shape,dtype=np.uint8)
        for image,sprite in zip(result,sprites):
            x,y,w,h=(sprite[k] for k in ("x","y","width","height"))
            encoded[y:y+h,x:x+w]=image
    else:raise ValueError("Unknown compressed method")
    results=[]
    for i,sprite in enumerate(sprites):
        x,y,w,h=(sprite[k] for k in ("x","y","width","height"))
        if (not all(isinstance(v,int) for v in (x,y,w,h))
                or min(x,y)<0 or min(w,h)<1
                or x+w>shape[1] or y+h>shape[0]):
            raise ValueError("Historical sprite out of bounds")
        image=np.ascontiguousarray(encoded[y:y+h,x:x+w])
        if mode in CANDIDATES and i%2:
            prior=results[i-1]
            reference=np.zeros_like(image)
            hh=min(image.shape[0],prior.shape[0])
            ww=min(image.shape[1],prior.shape[1])
            reference[:hh,:ww]=prior[:hh,:ww]
            image=(np.bitwise_xor(image,reference) if mode.startswith("xor_")
                   else ((image.astype(np.uint16)+reference.astype(np.uint16))&255).astype(np.uint8))
        if sha(image.tobytes())!=sprite["pixel_rgba_sha256"]:
            raise ValueError("Failed to reproduce actual historical PNG RGBA bytes")
        results.append(image)
    atlas=np.zeros(shape,dtype=np.uint8)
    for image,sprite in zip(results,sprites):
        x,y,w,h=(sprite[k] for k in ("x","y","width","height"))
        atlas[y:y+h,x:x+w]=image
    if sha(atlas.tobytes())!=meta["atlas_sha256"]:
        raise ValueError("Reconstructed historical full source atlas incorrect")
    return results,meta


def run_group(items:list[dict],inventory:dict,name:str,root:Path)->dict:
    atlas,locations=pack(items)
    meta=manifest_for(items,inventory)
    folder=root/name
    folder.mkdir(parents=True,exist_ok=True)
    results={}
    with io.BytesIO() as b:
        Image.fromarray(atlas,mode="RGBA").save(b,format="PNG")
        png=b.getvalue()
    (folder/"historical_rgba_source_atlas.png").write_bytes(png)
    for method in ALL_METHODS:
        started=time.perf_counter_ns()
        stream,extras=encode(items,meta,method)
        elapsed_encode=(time.perf_counter_ns()-started)/1e9
        target=folder/(method+".bl81")
        target.write_bytes(stream)
        started=time.perf_counter_ns()
        images,manifest=decode(target.read_bytes())
        elapsed_decode=(time.perf_counter_ns()-started)/1e9
        if (manifest!={**meta,"codec":method} or
            not all(np.array_equal(x["rgba"],y)
                    for x,y in zip(items,images))):
            raise AssertionError("Saved file was not original pixel exact")
        results[method]={
            "file":target.name,
            "bytes":len(stream),"sha256":sha(stream),
            "encode_seconds":elapsed_encode,"decode_seconds":elapsed_decode,
            **extras,
        }
    best,method=min((v["bytes"],k) for k,v in results.items() if k in CANDIDATES)
    base,reference=min((v["bytes"],k) for k,v in results.items() if k in REFERENCES)
    return {
        "source_commit_group":name,
        "source_pairs":len(items)//2,
        "pixel_changed_pairs":sum(
            x["rgba"].shape!=y["rgba"].shape or not np.array_equal(x["rgba"],y["rgba"])
            for x,y in zip(items[::2],items[1::2])),
        "source_rgba_atlas_file":"historical_rgba_source_atlas.png",
        "source_rgba_atlas_png_sha256":sha(png),
        "shared_metadata":meta,
        "methods":results,
        "best_new_delta":method,"best_new_delta_bytes":best,
        "strongest_existing_format":reference,
        "strongest_existing_bytes":base,
        "delta_smaller_than_all_prior_references":best<base,
        "all_original_png_rgba_reconstructed_exactly":True,
    }


def self_test():
    rng=np.random.default_rng(81)
    samples=[]
    for k in range(3):
        old=rng.integers(0,256,(20+k,28+k,4),dtype=np.uint8)
        new=rng.integers(0,256,(22+k,26+k,4),dtype=np.uint8)
        for role,rgba in (("old",old),("new",new)):
            with io.BytesIO() as b:
                Image.fromarray(rgba,mode="RGBA").save(b,format="PNG")
                raw=b.getvalue()
            samples.append({
                "rgba":rgba,"raw":raw,
                "meta":{"width":rgba.shape[1],"height":rgba.shape[0],
                        "source_png_sha256":sha(raw),
                        "pixel_rgba_sha256":sha(rgba.tobytes()),
                        "role":role,"filename":str(k)+role}
            })
    inventory={"revision_groups":{"test":{"commit":"0"*40,
                                            "first_parent":"1"*40,
                                            "png_pairs_modified_in_git":3,
                                            "png_pairs_really_changed_rgba":3}}}
    meta=manifest_for(samples,inventory)
    for mode in ALL_METHODS:
        packed,_=encode(samples,meta,mode)
        images,source=decode(packed)
        assert source=={**meta,"codec":mode}
        assert all(np.array_equal(x["rgba"],y)
                   for x,y in zip(samples,images))
        if mode in CANDIDATES:
            try:decode(packed[:-1])
            except (ValueError,zstd.ZstdError,OSError):pass
            else:raise AssertionError("Truncated historical delta accepted")
    print("EXP82 SELFTEST PASS: 11 exact codecs, changed sprite dimensions, original source PNG data and all hidden RGBA")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="/tmp/blkh_exp82")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:self_test();return
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    all_assets,inventory=discover(root)
    observations=[]
    sets=[("all_19_pairs",all_assets)]
    for label,commit,paths in COMMIT_GROUPS:
        part=[asset for asset in all_assets
              if asset["meta"]["commit_label"]==label]
        if len(part)!=2*len(paths):raise ValueError("Missing historical Git revision")
        sets.append((label,part))
    for name,items in sets:
        row=run_group(items,inventory,name,root)
        observations.append(row)
        print("EXP82",name,"real pixel-changed revisions",row["pixel_changed_pairs"],
              "of",row["source_pairs"],
              "delta",row["best_new_delta"],row["best_new_delta_bytes"],
              "best reference",row["strongest_existing_format"],
              row["strongest_existing_bytes"],
              "win",row["delta_smaller_than_all_prior_references"],flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":82,
            "protocol":"research/dedup/BHUH_EXP82_GIT_REAL_REVISIONS_PROTOCOL.md",
            "source_inventory":inventory,
            "data_source":"Orama-Interactive/Pixelorama MIT Git source, 3 historical commits",
            "source_revision_pairs":19,
            "source_images":38,
            "candidate_codecs":list(CANDIDATES),
            "reference_codecs":list(REFERENCES),
            "versions":{"python":platform.python_version(),
                        "numpy":np.__version__,"pillow":PIL_VERSION,
                        "zstd":zstd.__version__},
            "results":observations,
        },indent=2))
    assert len(observations)==4 and len(all_assets)==38
    wins=sum(r["delta_smaller_than_all_prior_references"]
             for r in observations[1:])
    criterion=(observations[0]["delta_smaller_than_all_prior_references"]
               and wins>=2)
    print("EXP82 RESULT: 19 actual Git-modified image pairs, four independent complete archives, source-aware delta gate",criterion,
          "per-commit wins",wins,flush=True)


if __name__=="__main__":
    main()
