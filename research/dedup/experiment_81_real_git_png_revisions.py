"""
BLKH Exp81 — exact RGB+alpha historical sprite edits from immutable Git commits.

Actual historical PNG file changes in Pixel-Boy NinjaAdventure source repo,
not generated revisions, no cherry-picking by compression outcomes. Count
source SHA + commit + path + all PNG/data bytes with matched metadata in
eight self-contained file variants per subset. No loss of transparent RGB.
Precommitted protocol: BHUH_EXP81_REAL_GIT_REVISIONS_PROTOCOL.md.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import struct
import subprocess
import time

import numpy as np
from PIL import Image, features, __version__ as PIL_VERSION
import zstandard as zstd

SOURCE_REPO="https://github.com/pixel-boy/NinjaAdventure.git"
SOURCE_HEAD="6ac78232d5aedcc85ce5f27d060ea92366f7c24a"
SOURCE_PAGE="https://github.com/pixel-boy/NinjaAdventure"
SOURCE_LICENSE="CC0 via original Ninja Adventure art release by Pixel-Boy and AAA"
ALLOWED=("content/character/","content/ui/","content/destroyable/",
         "content/particle/","content/map/")
MAX_PAIRS=24
COMMIT_COUNT=12
PREFERRED_PREFIXES=(4,8,16,24)
ATLAS_MAX_WIDTH=1024
BLOCK=16
MAX_SOURCE_PNG_BYTES=2_500_000
MAX_ATLAS_PIXELS=8_000_000
MAX_ARCHIVE=50_000_000
HEAD=struct.Struct("<5sBI")
MAGIC=b"BL81R"
SCHEMA=1
MODES=("xor_webp","mod_webp","xor_zstd","mod_zstd",
       "original_png_zstd","atlas_webp","atlas_png","atlas_raw_zstd")


def sha(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def git(cwd:Path,*args:str)->bytes:
    p=subprocess.run(["git","-C",str(cwd),*args],stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,timeout=110)
    if p.returncode!=0:
        raise RuntimeError("Git source audit failed: "+
                           p.stderr.decode(errors="replace")[:1200])
    return p.stdout


def get_clone(folder:Path)->Path:
    source=folder/"NinjaAdventure"
    folder.mkdir(parents=True,exist_ok=True)
    if not (source/".git").is_dir():
        p=subprocess.run(["git","clone","--quiet","--no-checkout",
                         "--filter=blob:none",SOURCE_REPO,str(source)],
                         capture_output=True,timeout=180)
        if p.returncode:
            raise RuntimeError("Pinned public source repository clone failed: "+
                               p.stderr.decode(errors="replace")[:1200])
    git(source,"checkout","--quiet",SOURCE_HEAD)
    actual=git(source,"rev-parse","HEAD").decode().strip()
    if actual!=SOURCE_HEAD:
        raise AssertionError("Unpinned external Git HEAD")
    return source


def png_rgba(raw:bytes)->np.ndarray:
    if len(raw)>MAX_SOURCE_PNG_BYTES or raw[:8]!=b"\x89PNG\r\n\x1a\n":
        raise ValueError("Source file not a supported PNG")
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        w,h=im.size
        if not (16<=w<=512 and 16<=h<=512):
            raise ValueError("PNG source dimensions outside frozen range")
        return np.ascontiguousarray(np.asarray(im.convert("RGBA"),dtype=np.uint8))


def scan(source:Path,root:Path)->tuple[list[dict],dict]:
    commits=git(source,"rev-list",f"--max-count={COMMIT_COUNT}",
                "--first-parent",SOURCE_HEAD).decode().strip().splitlines()
    if len(commits)!=COMMIT_COUNT:
        raise ValueError("Cannot traverse all 12 fixed historical commits")
    inventory=[]
    accepted=[]
    for i,(child,parent) in enumerate(zip(commits,commits[1:])):
        paths=git(source,"diff","--name-only","--diff-filter=M",
                  parent,child,"--","*.png").decode().strip().splitlines()
        for path in sorted(paths):
            item={"transition_index_newest_first":i,
                  "parent_commit":parent,"child_commit":child,"source_path":path}
            if not path.startswith(ALLOWED) or path.lower().endswith(
                    ("sheet.png","spritesheet.png","preview.png")):
                item["rejection"]="outside predeclared sprite paths"
                inventory.append(item)
                continue
            try:
                before_bytes=git(source,"show",f"{parent}:{path}")
                after_bytes=git(source,"show",f"{child}:{path}")
                before=png_rgba(before_bytes)
                after=png_rgba(after_bytes)
                if before.shape!=after.shape:
                    raise ValueError("Original Git PNG dimensions changed")
                if np.array_equal(before,after):
                    raise ValueError("Original RGBA pixels unchanged after PNG-only edit")
                item.update({
                    "before_original_png_sha256":sha(before_bytes),
                    "after_original_png_sha256":sha(after_bytes),
                    "before_rgba_sha256":sha(before.tobytes()),
                    "after_rgba_sha256":sha(after.tobytes()),
                    "width":int(before.shape[1]),"height":int(before.shape[0]),
                    "changed_channel_values":int(np.count_nonzero(before!=after)),
                    "diff_is_actual_rgba":True,
                })
                accepted.append({"meta":item,"before":before,"after":after,
                                 "before_png":before_bytes,"after_png":after_bytes})
                item["rejection"]=None
            except (ValueError,RuntimeError,OSError) as exc:
                item["rejection"]=str(exc)
            inventory.append(item)
    accepted=accepted[:MAX_PAIRS]
    info={"source_repository":SOURCE_PAGE,"source_head_sha":SOURCE_HEAD,
          "source_first_parent_commits":commits,
          "tested_consecutive_parent_child_transitions":COMMIT_COUNT-1,
          "all_considered_modified_pngs":inventory,
          "valid_edited_rgba_pairs_before_first24_limit":sum(
              i["rejection"] is None for i in inventory),
          "selected_pairs_count":len(accepted),
          "selected_pairs":[x["meta"] for x in accepted],
          "source_license":SOURCE_LICENSE}
    root.mkdir(parents=True,exist_ok=True)
    (root/"source_inventory.json").write_text(json.dumps(info,indent=2))
    if len(accepted)<1:
        raise RuntimeError("No genuine same-path modified RGBA sprite PNG pairs; source audit saved")
    return accepted,info


def pack_pair_sprites(pairs:list[dict],transformed:str|None):
    if transformed not in (None,"xor","mod"):
        raise ValueError("Unknown reversible transform")
    sprites=[]
    x=y=0
    max_height=0
    width=0
    for pair_number,pair in enumerate(pairs):
        if pair["before"].shape!=pair["after"].shape:
            raise ValueError("Versioned PNG shape drift unsupported")
        original=pair["before"]
        modified=pair["after"]
        if transformed=="xor":
            modified=np.bitwise_xor(modified,original)
        elif transformed=="mod":
            modified=((modified.astype(np.uint16)-original.astype(np.uint16))&255).astype(np.uint8)
        for version,rgba,raw in (
            ("before",original,pair["before_png"]),
            ("after",modified,pair["after_png"])):
            h,w,_=rgba.shape
            padded_w=(w+BLOCK-1)//BLOCK*BLOCK
            padded_h=(h+BLOCK-1)//BLOCK*BLOCK
            if padded_w>ATLAS_MAX_WIDTH:
                raise ValueError("Source sprite too wide")
            if x+padded_w>ATLAS_MAX_WIDTH:
                y+=max_height
                x=max_height=0
            sprites.append({
                "pair_index":pair_number,"version":version,
                "width":w,"height":h,"x":x,"y":y,
                "meta":pair["meta"],"raw_source_png":raw,
                "original_rgba":pair[version],
            })
            width=max(width,x+padded_w)
            max_height=max(max_height,padded_h)
            x+=padded_w
    height=y+max_height
    if width*height>MAX_ATLAS_PIXELS or not width or not height:
        raise ValueError("Source atlas pixel budget exceeded")
    atlas=np.zeros((height,width,4),dtype=np.uint8)
    for s in sprites:
        pos=s["x"],s["y"];w,h=s["width"],s["height"]
        pair=pairs[s["pair_index"]]
        raw=(pair["before"] if s["version"]=="before"
             else pair["after"])
        if transformed and s["version"]=="after":
            raw=(np.bitwise_xor(raw,pair["before"]) if transformed=="xor"
                 else ((raw.astype(np.uint16)-pair["before"].astype(np.uint16))&255).astype(np.uint8))
        atlas[pos[1]:pos[1]+h,pos[0]:pos[0]+w]=raw
    return atlas,sprites


def manifest_for(pairs:list[dict])->dict:
    atlas,sprites=pack_pair_sprites(pairs,None)
    return {
        "source_repo":SOURCE_PAGE,
        "source_fixed_head_commit":SOURCE_HEAD,
        "source_license":SOURCE_LICENSE,
        "source_pairs":[p["meta"] for p in pairs],
        "pair_count":len(pairs),
        "atlas_shape":list(atlas.shape),
        "original_rgba_atlas_sha256":sha(atlas.tobytes()),
        "sprite_positions":[{k:s[k] for k in (
            "pair_index","version","width","height","x","y")}
            for s in sprites],
        "max_packing_row_width":ATLAS_MAX_WIDTH,
        "padded_pixel_rgba":[0,0,0,0],
        "bit_depth":"uint8 RGBA, including hidden transparent RGB",
    }


def compress(data:bytes)->bytes:
    return zstd.ZstdCompressor(level=9).compress(data)


def decompress(data:bytes,limit:int)->bytes:
    return zstd.ZstdDecompressor().decompress(data,max_output_size=limit)


def encode_webp(atlas:np.ndarray)->bytes:
    with io.BytesIO() as file:
        Image.fromarray(atlas).save(file,format="WEBP",lossless=True,exact=True,method=6)
        return file.getvalue()


def encode_png(atlas:np.ndarray)->bytes:
    with io.BytesIO() as file:
        Image.fromarray(atlas).save(file,format="PNG",optimize=True,compress_level=9)
        return file.getvalue()


def decode_image(raw:bytes,shape:tuple[int,int,int])->np.ndarray:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        result=np.asarray(image.convert("RGBA"),dtype=np.uint8)
    if result.shape!=shape:
        raise ValueError("Stored image dimensions changed")
    return result


def encode_case(pairs:list[dict],manifest:dict,mode:str)->bytes:
    if mode not in MODES:
        raise ValueError("Unexpected compression mode")
    transform=("xor" if mode.startswith("xor_") else
               "mod" if mode.startswith("mod_") else None)
    atlas,_=pack_pair_sprites(pairs,transform)
    if list(atlas.shape)!=manifest["atlas_shape"]:
        raise ValueError("Unmatched changed version atlas")
    if mode=="original_png_zstd":
        body=bytearray()
        for p in pairs:
            for key in ("before_png","after_png"):
                item=p[key]
                body.extend(struct.pack("<I",len(item)))
                body.extend(item)
        packed=compress(bytes(body))
    elif mode.endswith("webp") or mode=="atlas_webp":
        packed=encode_webp(atlas)
        if not np.array_equal(decode_image(packed,atlas.shape),atlas):
            raise ValueError("WebP failed exact original RGBA including transparent colors")
    elif mode=="atlas_png":
        packed=encode_png(atlas)
    else:
        packed=compress(atlas.tobytes())
    metadata=json.dumps({**manifest,"codec":mode,"payload_sha256":sha(packed)},sort_keys=True,
                        separators=(",",":")).encode()
    if not 0<len(metadata)<180_000:
        raise ValueError("Metadata too large")
    return HEAD.pack(MAGIC,SCHEMA,len(metadata))+metadata+packed


def decode_case(blob:bytes)->tuple[list[np.ndarray],dict]:
    if not HEAD.size<len(blob)<=MAX_ARCHIVE:
        raise ValueError("Truncated or oversized archive")
    magic,version,n=HEAD.unpack(blob[:HEAD.size])
    if magic!=MAGIC or version!=SCHEMA or not 0<n<180_000 or HEAD.size+n>len(blob):
        raise ValueError("Bad historical asset file")
    meta=json.loads(blob[HEAD.size:HEAD.size+n])
    mode=meta.get("codec")
    pair_count=meta.get("pair_count")
    if mode not in MODES or not isinstance(pair_count,int) or not 1<=pair_count<=MAX_PAIRS:
        raise ValueError("Invalid source version count")
    if len(meta.get("source_pairs",[]))!=pair_count:
        raise ValueError("Missing historical Git versions")
    positions=meta.get("sprite_positions")
    if not isinstance(positions,list) or len(positions)!=2*pair_count:
        raise ValueError("Invalid asset atlas mapping")
    shape=tuple(meta["atlas_shape"])
    if len(shape)!=3 or shape[2]!=4 or int(np.prod(shape))>MAX_ATLAS_PIXELS*4:
        raise ValueError("Invalid lossless RGBA output size")
    encoded=blob[HEAD.size+n:]
    if sha(encoded)!=meta.get("payload_sha256"):
        raise ValueError("Saved codec payload was truncated or modified")
    if mode=="original_png_zstd":
        input_data=decompress(encoded,pair_count*2*(MAX_SOURCE_PNG_BYTES+4))
        off=0
        images=[]
        for p in meta["source_pairs"]:
            for version in ("before","after"):
                if off+4>len(input_data):
                    raise ValueError("Truncated source PNG length")
                size=struct.unpack_from("<I",input_data,off)[0];off+=4
                if size<8 or size>MAX_SOURCE_PNG_BYTES or off+size>len(input_data):
                    raise ValueError("Invalid source PNG bytes")
                data=input_data[off:off+size];off+=size
                if sha(data)!=p[version+"_original_png_sha256"]:
                    raise ValueError("Original PNG SHA mismatch")
                images.append(png_rgba(data))
        if off!=len(input_data):
            raise ValueError("Unexpected source PNG padding")
    else:
        if mode.endswith("webp") or mode=="atlas_webp":
            atlas=decode_image(encoded,shape)
        elif mode=="atlas_png":
            atlas=decode_image(encoded,shape)
        else:
            decoded=decompress(encoded,int(np.prod(shape)))
            if len(decoded)!=int(np.prod(shape)):
                raise ValueError("Wrong source pixel byte count")
            atlas=np.frombuffer(decoded,dtype=np.uint8).reshape(shape)
        cropped=[]
        for entry in positions:
            x,y,w,h=(entry[k] for k in ("x","y","width","height"))
            if not all(isinstance(z,int) for z in (x,y,w,h)):
                raise ValueError("Invalid source sprite coords")
            if min(x,y)<0 or min(w,h)<1 or x+w>shape[1] or y+h>shape[0]:
                raise ValueError("Out of atlas bounds")
            cropped.append(np.ascontiguousarray(atlas[y:y+h,x:x+w]))
        images=[]
        for i in range(pair_count):
            before=cropped[2*i]
            after=cropped[2*i+1]
            if mode.startswith("xor_"):
                after=np.bitwise_xor(after,before)
            elif mode.startswith("mod_"):
                after=((after.astype(np.uint16)+before.astype(np.uint16))&255).astype(np.uint8)
            images.extend((before,after))
    if len(images)!=2*pair_count:
        raise ValueError("Missing source asset versions")
    source_atlas=np.zeros(shape,dtype=np.uint8)
    for i,(image,position) in enumerate(zip(images,positions)):
        p=meta["source_pairs"][i//2]
        version="before" if i%2==0 else "after"
        if sha(image.tobytes())!=p[version+"_rgba_sha256"]:
            raise ValueError("Decoded historical RGBA values changed")
        x,y,w,h=(position[k] for k in ("x","y","width","height"))
        if image.shape!=(h,w,4):
            raise ValueError("Historical sprite decoded geometry wrong")
        source_atlas[y:y+h,x:x+w]=image
    if sha(source_atlas.tobytes())!=meta["original_rgba_atlas_sha256"]:
        raise ValueError("Historical RGBA atlas checksum mismatch")
    return images,{k:v for k,v in meta.items() if k!="payload_sha256"}


def self_test():
    rng=np.random.default_rng(81)
    pairs=[]
    for i in range(4):
        first=rng.integers(0,256,(32+i,28+i,4),dtype=np.uint8)
        second=first.copy()
        second[i:i+3,i:i+3] = rng.integers(0,256,(3,3,4),dtype=np.uint8)
        def encode_png_local(image):
            with io.BytesIO() as io_file:
                Image.fromarray(image).save(io_file,format="PNG")
                return io_file.getvalue()
        b,a=encode_png_local(first),encode_png_local(second)
        p={
            "source_path":f"content/ui/test_{i}.png",
            "parent_commit":"0"*40,"child_commit":"1"*40,
            "before_rgba_sha256":sha(first.tobytes()),
            "after_rgba_sha256":sha(second.tobytes()),
            "before_original_png_sha256":sha(b),
            "after_original_png_sha256":sha(a),
            "width":first.shape[1],"height":first.shape[0],
        }
        pairs.append({"before":first,"after":second,
                      "before_png":b,"after_png":a,"meta":p})
    m=manifest_for(pairs)
    for mode in MODES:
        coded=encode_case(pairs,m,mode)
        decoded,manifest=decode_case(coded)
        assert manifest=={**m,"codec":mode}
        assert all(np.array_equal(decoded[2*i],pair["before"]) and
                   np.array_equal(decoded[2*i+1],pair["after"])
                   for i,pair in enumerate(pairs))
        try:decode_case(coded[:-1])
        except (OSError,ValueError,zstd.ZstdError):
            pass
        else:raise AssertionError("Truncated saved file decoded without error")
    print("EXP81 SELFTEST PASS: 4 real-source-format historic pairs, all eight complete codec modes exact including hidden alpha RGB")


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--self-test",action="store_true")
    a.add_argument("--out",default="/tmp/blkh_exp81")
    a.add_argument("--source-cache",default="/tmp/blkh_exp81_source")
    args=a.parse_args()
    if args.self_test:
        self_test();return
    if not features.check("webp"):
        raise RuntimeError("Strict RGBA exact lossless WebP required")
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    source=get_clone(Path(args.source_cache))
    accepted,inventory=scan(source,root)
    samples=[s for s in PREFERRED_PREFIXES if s<=len(accepted)]
    if not samples:samples=[len(accepted)]
    records=[]
    for n in samples:
        chosen=accepted[:n]
        manifest=manifest_for(chosen)
        directory=root/f"pairs_{n:02d}"
        directory.mkdir(parents=True,exist_ok=True)
        modes={}
        for mode in MODES:
            start=time.perf_counter_ns()
            encoded=encode_case(chosen,manifest,mode)
            encode_s=(time.perf_counter_ns()-start)/1e9
            path=directory/(mode+".bl81")
            path.write_bytes(encoded)
            start=time.perf_counter_ns()
            images,decoded=decode_case(path.read_bytes())
            decode_s=(time.perf_counter_ns()-start)/1e9
            if decoded!={**manifest,"codec":mode}:
                raise AssertionError("Original source commit mapping changed")
            for i,p in enumerate(chosen):
                if not (np.array_equal(images[2*i],p["before"]) and
                        np.array_equal(images[2*i+1],p["after"])):
                    raise AssertionError("Original author edited RGBA mismatch")
            modes[mode]={"file":path.name,"bytes":len(encoded),"sha256":sha(encoded),
                         "encode_seconds":encode_s,"decode_seconds":decode_s}
        smallest,method=min((modes[k]["bytes"],k)
                            for k in MODES if k in MODES[:4])
        best,reference=min((modes[k]["bytes"],k) for k in MODES[4:])
        result={
            "historically_edited_png_pairs":n,
            "independent_source_repository_count":1,
            "manifest":manifest,
            "all_original_rgba_exact":True,
            "codec_archives":modes,
            "best_delta_codec":method,"best_delta_bytes":smallest,
            "best_conventional_codec":reference,"best_conventional_bytes":best,
            "delta_smaller_than_strongest_reference":smallest<best,
        }
        records.append(result)
        print("EXP81",n,"genuine authored Git PNG edits","delta",method,smallest,
              "reference",reference,best,
              "win",result["delta_smaller_than_strongest_reference"],flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":81,"preregistration":"research/dedup/BHUH_EXP81_REAL_GIT_REVISIONS_PROTOCOL.md",
            "source_repo":SOURCE_PAGE,"pinned_head_commit":SOURCE_HEAD,
            "source_audit_sample_count":len(accepted),
            "evaluation_prefixes":samples,
            "scope":"one historical CC0 game project; authored edits, not generated, non-independent correlated changes",
            "software":{"python":platform.python_version(),"numpy":np.__version__,
                        "pillow":PIL_VERSION,"zstandard":zstd.__version__},
            "results":records,
        },indent=2))
    print("EXP81 DONE: genuine same-path historical PNG changes, all full files independently reconstructed",flush=True)


if __name__=="__main__":
    main()
