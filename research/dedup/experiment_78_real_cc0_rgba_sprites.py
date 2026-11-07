"""
BLKH Experiment 78 — REAL CC0 Kenney UI PNG assets; exact RGBA pixel codec.

One archived Kenney CC0 pack, deterministic first 96 eligible originals.
Compare 16px content-addressed tile dictionary with strong exact, packed
PNG/WebP atlas and zstd(original PNG file bytes) baselines. Preserve *all*
RGBA bytes including hidden RGB when alpha==0: Pillow WebP exact=True.
All complete-file sizes and source SHA256 included. 32/64/96 nested sets,
NOT three independent creative families. No authored revision histories.

Preregistered protocol: research/dedup/BHUH_EXP78_REAL_CC0_RGBA_PROTOCOL.md
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import struct
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image, __version__ as PIL_VERSION, features
import zstandard as zstd

SOURCE_URL="https://opengameart.org/sites/default/files/kenney_ui-pack.zip"
SOURCE_LANDING="https://opengameart.org/content/ui-pack"
LICENSE="CC0-1.0"
NESTED=(32,64,96)
TILE=16
MAX_ROW_WIDTH=1024
MAX_PNG_BYTES=5_000_000
MAX_ZIP_BYTES=20_000_000
MAX_ATLAS_PIXELS=6_000_000
MAX_FILE_BYTES=40_000_000
MAGIC=b"BL78A"
HEAD=struct.Struct("<5sBI")
DICT=struct.Struct("<II")
CODECS=("tile16_zstd","tile16_webp","atlas_png","atlas_webp",
        "atlas_raw_zstd","original_png_zstd","optimized_png_zstd")


def sha(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()


def zcompress(b:bytes)->bytes:
    return zstd.ZstdCompressor(level=9).compress(b)


def zdecompress(raw:bytes,limit:int)->bytes:
    return zstd.ZstdDecompressor().decompress(raw,max_output_size=limit)


def fetch_zip(cache:Path)->tuple[bytes,str]:
    cache.mkdir(parents=True,exist_ok=True)
    zipfile_path=cache/"kenney_ui-pack.zip"
    if zipfile_path.exists():
        blob=zipfile_path.read_bytes()
    else:
        req=urllib.request.Request(SOURCE_URL,headers={
            "User-Agent":"BLKH-reproducible-scientific-experiment-78/1.0"
        })
        with urllib.request.urlopen(req,timeout=120) as r:
            blob=r.read(MAX_ZIP_BYTES+1)
        if not (100_000<len(blob)<=MAX_ZIP_BYTES):
            raise ValueError("Unexpected source ZIP size")
        zipfile_path.write_bytes(blob)
    if not (100_000<len(blob)<=MAX_ZIP_BYTES) or not zipfile.is_zipfile(io.BytesIO(blob)):
        raise ValueError("Missing/invalid fixed source Kenney ZIP")
    return blob,sha(blob)


def discover(zip_bytes:bytes,zip_hash:str,root:Path)->list[dict]:
    candidates=[]
    inventory=[]
    excluded=Counter()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        all_names=sorted(x for x in z.namelist() if x.lower().endswith(".png"))
        for name in all_names:
            lowered=name.lower()
            if any(key in lowered for key in
                   ("spritesheet","preview","sample","sheet","tilemap")):
                excluded["sheet_preview"]=excluded["sheet_preview"]+1
                continue
            member=z.getinfo(name)
            if member.file_size>MAX_PNG_BYTES:
                excluded["oversized_file"]+=1
                continue
            raw=z.read(name)
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    h,w=im.height,im.width
                    if not (16<=w<=512 and 16<=h<=512):
                        excluded["dimensions"]+=1
                        continue
                    pic=np.asarray(im.convert("RGBA"),dtype=np.uint8)
            except Exception:
                excluded["unreadable_png"]+=1
                continue
            if not np.any(pic[:,:,3]<255):
                excluded["opaque"]+=1
                continue
            entry={
                "filename":name,"source_png_sha256":sha(raw),
                "pixel_rgba_sha256":sha(pic.tobytes()),
                "width":w,"height":h,
                "alpha_transparent_pixel_count":int(np.count_nonzero(pic[:,:,3]==0)),
                "alpha_partial_pixel_count":int(np.count_nonzero(
                    (pic[:,:,3]>0)&(pic[:,:,3]<255))),
            }
            inventory.append(entry)
            candidates.append({"meta":entry,"raw":raw,"rgba":pic})
    report={
        "source_url":SOURCE_URL,"zip_sha256":zip_hash,
        "source_zip_bytes":len(zip_bytes),
        "all_png_members":len(all_names),
        "valid_alpha_pngs":len(candidates),
        "excluded_summary":dict(excluded),
        "eligible_sorted_members":[s["meta"] for s in candidates],
        "selected_first_96":[s["meta"]["filename"] for s in candidates[:96]],
    }
    root.mkdir(parents=True,exist_ok=True)
    (root/"source_inventory.json").write_text(json.dumps(report,indent=2))
    if len(candidates)<max(NESTED):
        raise ValueError("Preregistered selection failed: fewer than 96 qualifying source assets")
    return candidates[:max(NESTED)]


def pack_assets(items:list[dict])->tuple[np.ndarray,dict]:
    if len(items) not in NESTED:
        raise ValueError("Invalid registered sample count")
    positions=[]
    x=y=0
    row_height=0
    max_width=0
    for item in items:
        w=item["meta"]["width"];h=item["meta"]["height"]
        padded_w=(w+TILE-1)//TILE*TILE
        padded_h=(h+TILE-1)//TILE*TILE
        if padded_w>MAX_ROW_WIDTH:
            raise ValueError("Sprite wider than shelf")
        if x+padded_w>MAX_ROW_WIDTH:
            y+=row_height
            x=0;row_height=0
        positions.append({"x":x,"y":y,"width":w,"height":h,
                          "padded_width":padded_w,"padded_height":padded_h})
        max_width=max(max_width,x+padded_w)
        row_height=max(row_height,padded_h)
        x+=padded_w
    height=y+row_height
    width=max_width
    if width%TILE or height%TILE or width*height>MAX_ATLAS_PIXELS:
        raise ValueError("Preregistered RGBA atlas exceeds space limits")
    atlas=np.zeros((height,width,4),dtype=np.uint8)
    for item,pos in zip(items,positions):
        x,y,w,h=pos["x"],pos["y"],pos["width"],pos["height"]
        atlas[y:y+h,x:x+w]=item["rgba"]
        assert np.array_equal(atlas[y:y+h,x:x+w],item["rgba"])
    return atlas,{"atlas_width":width,"atlas_height":height,
                  "sprites":[{**item["meta"],**pos}
                             for item,pos in zip(items,positions)]}


def base_manifest(source_hash:str,source_bytes:int,items:list[dict],
                  atlas:np.ndarray,layout:dict)->dict:
    return {
        "title":"Kenney UI Pack — original CC0 alpha sprites",
        "source_url":SOURCE_URL,
        "source_landing":SOURCE_LANDING,
        "license":LICENSE,
        "source_zip_sha256":source_hash,
        "source_zip_size_bytes":source_bytes,
        "collection_size":len(items),
        "atlas_shape_rgba":[*atlas.shape],
        "atlas_rgba_sha256":sha(atlas.tobytes()),
        "tile_side":TILE,
        "packing":"lexicographic precommitted sorted originals, ordered fixed shelf max width 1024",
        **layout,
    }


def encode_webp(rgba:np.ndarray)->bytes:
    with io.BytesIO() as f:
        Image.fromarray(rgba,mode="RGBA").save(
            f,format="WEBP",lossless=True,method=6,exact=True)
        return f.getvalue()


def decode_rgba(data:bytes,shape:tuple[int,int,int])->np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        result=np.asarray(im.convert("RGBA"),dtype=np.uint8)
    if result.shape!=tuple(shape):
        raise ValueError("Lossless image geometry mismatch")
    return result


def dictionary(atlas:np.ndarray,kind:str)->tuple[bytes,dict]:
    h,w,_=atlas.shape
    if h%TILE or w%TILE:
        raise ValueError("Atlas not tile-aligned")
    seen={}
    tiles=[]
    ids=[]
    for y in range(0,h,TILE):
        for x in range(0,w,TILE):
            tile=atlas[y:y+TILE,x:x+TILE].tobytes()
            hash_value=sha(tile)
            if hash_value not in seen:
                seen[hash_value]=len(tiles)
                tiles.append(tile)
            elif tiles[seen[hash_value]]!=tile:
                raise ValueError("Content hash collision")
            ids.append(seen[hash_value])
    if not 1<=len(tiles)<65536:
        raise ValueError("Number of dictionary tiles exceeds uint16 index")
    if kind=="tile16_zstd":
        packed=zcompress(b"".join(tiles))
        columns=0
    elif kind=="tile16_webp":
        columns=16
        rows=(len(tiles)+columns-1)//columns
        packed_canvas=np.zeros((rows*TILE,columns*TILE,4),dtype=np.uint8)
        for i,tile in enumerate(tiles):
            row,col=divmod(i,columns)
            packed_canvas[row*TILE:(row+1)*TILE,
                          col*TILE:(col+1)*TILE]=np.frombuffer(
                tile,dtype=np.uint8).reshape(TILE,TILE,4)
        packed=encode_webp(packed_canvas)
        if not np.array_equal(decode_rgba(packed,packed_canvas.shape),packed_canvas):
            raise ValueError("WebP dictionary changed exact RGBA including transparent RGB")
    else:
        raise ValueError("Invalid dictionary mode")
    references=np.asarray(ids,dtype="<u2").tobytes()
    result=DICT.pack(len(tiles),columns)+references+packed
    return result,{
        "unique_tiles":len(tiles),
        "tile_references":len(ids),
        "duplicate_tile_references":len(ids)-len(tiles),
        "tile_index_bytes":len(references),
        "dictionary_compressed_bytes":len(packed),
    }


def decode_dict(blob:bytes,kind:str,shape:tuple[int,int,int])->np.ndarray:
    h,w,c=shape
    if c!=4 or h%TILE or w%TILE or len(blob)<DICT.size:
        raise ValueError("Bad dictionary/shape")
    n,cols=DICT.unpack(blob[:DICT.size])
    count=(h//TILE)*(w//TILE)
    if not 1<=n<=count or n>=65536:
        raise ValueError("Invalid number of dictionary tiles")
    end=DICT.size+count*2
    if len(blob)<end:
        raise ValueError("Incomplete tile references")
    ids=np.frombuffer(blob,dtype="<u2",count=count,offset=DICT.size)
    if np.any(ids>=n):
        raise ValueError("Invalid dictionary tile indices")
    encoded=blob[end:]
    if kind=="tile16_zstd":
        if cols!=0:raise ValueError("Wrong zstd dictionary layout")
        raw=zdecompress(encoded,n*TILE*TILE*4)
        if len(raw)!=n*TILE*TILE*4:
            raise ValueError("Decoded tile bytes do not match")
        tiles=np.frombuffer(raw,dtype=np.uint8).reshape(n,TILE,TILE,4)
    elif kind=="tile16_webp":
        if cols!=16:raise ValueError("Wrong WebP dictionary layout")
        rows=(n+cols-1)//cols
        canvas=decode_rgba(encoded,(rows*TILE,cols*TILE,4))
        tiles=np.stack([canvas[(j//cols)*TILE:(j//cols+1)*TILE,
                               (j%cols)*TILE:(j%cols+1)*TILE]
                        for j in range(n)])
    else:raise ValueError("Bad tile dictionary codec")
    out=np.empty(shape,dtype=np.uint8)
    for i,id_value in enumerate(ids):
        y,x=divmod(i,w//TILE)
        out[y*TILE:(y+1)*TILE,x*TILE:(x+1)*TILE]=tiles[id_value]
    return out


def encode_individual(items:list[dict],optimized:bool)->bytes:
    whole=bytearray()
    for entry in items:
        if optimized:
            with io.BytesIO() as f:
                Image.fromarray(entry["rgba"],mode="RGBA").save(
                    f,format="PNG",optimize=True,compress_level=9)
                raw=f.getvalue()
        else:
            raw=entry["raw"]
        whole.extend(struct.pack("<I",len(raw)))
        whole.extend(raw)
    return zcompress(bytes(whole))


def decode_individual(data:bytes,manifest:dict)->list[np.ndarray]:
    # Bound by every original source's max 5MB PNG member + size prefixes.
    limit=(len(manifest["sprites"])*(MAX_PNG_BYTES+4))
    blob=zdecompress(data,limit)
    offset=0
    pictures=[]
    for sprite in manifest["sprites"]:
        if len(blob)-offset<4:
            raise ValueError("Truncated PNG member length")
        n=struct.unpack_from("<I",blob,offset)[0]
        offset+=4
        if n>MAX_PNG_BYTES or n>len(blob)-offset:
            raise ValueError("Invalid PNG member payload")
        raw=blob[offset:offset+n]
        offset+=n
        if manifest.get("codec")=="original_png_zstd" and sha(raw)!=sprite["source_png_sha256"]:
            raise ValueError("Original source PNG bytes modified")
        pictures.append(decode_rgba(raw,(sprite["height"],sprite["width"],4)))
    if offset!=len(blob):
        raise ValueError("Trailing uncounted PNG archive members")
    return pictures


def wrap(mode:str,manifest:dict,payload:bytes)->bytes:
    if mode not in CODECS:
        raise ValueError("Invalid registered method")
    raw=json.dumps({**manifest,"codec":mode},sort_keys=True,
                   separators=(",",":")).encode()
    if not 0<len(raw)<250_000:
        raise ValueError("Oversized asset source manifest")
    return HEAD.pack(MAGIC,1,len(raw))+raw+payload


def unwrap(blob:bytes)->tuple[str,dict,bytes]:
    if not HEAD.size<len(blob)<=MAX_FILE_BYTES:
        raise ValueError("Truncated/oversized BL78 archive")
    m,v,n=HEAD.unpack(blob[:HEAD.size])
    if m!=MAGIC or v!=1 or not 0<n<250_000 or len(blob)<HEAD.size+n:
        raise ValueError("Invalid source manifest/header")
    meta=json.loads(blob[HEAD.size:HEAD.size+n])
    if (meta.get("codec") not in CODECS
            or meta.get("collection_size") not in NESTED
            or len(meta.get("sprites",[]))!=meta["collection_size"]
            or len(meta.get("atlas_shape_rgba",[]))!=3
            or meta.get("atlas_shape_rgba")!=[
                meta.get("atlas_height"),meta.get("atlas_width"),4]):
        raise ValueError("Invalid BL78 layout metadata")
    return meta["codec"],meta,blob[HEAD.size+n:]


def encode_case(atlas:np.ndarray,items:list[dict],meta:dict,
                codec:str)->tuple[bytes,dict]:
    stats={}
    if codec in ("tile16_zstd","tile16_webp"):
        binary,stats=dictionary(atlas,codec)
    elif codec=="atlas_png":
        with io.BytesIO() as f:
            Image.fromarray(atlas,mode="RGBA").save(
                f,format="PNG",optimize=True,compress_level=9)
            binary=f.getvalue()
    elif codec=="atlas_webp":
        binary=encode_webp(atlas)
    elif codec=="atlas_raw_zstd":
        binary=zcompress(atlas.tobytes())
    elif codec=="original_png_zstd":
        binary=encode_individual(items,False)
    elif codec=="optimized_png_zstd":
        binary=encode_individual(items,True)
    else:
        raise ValueError("Invalid fixed codec mode")
    return wrap(codec,meta,binary),stats


def decode_case(blob:bytes)->tuple[list[np.ndarray],dict]:
    codec,meta,payload=unwrap(blob)
    shape=tuple(meta["atlas_shape_rgba"])
    if codec in ("tile16_zstd","tile16_webp"):
        atlas=decode_dict(payload,codec,shape)
    elif codec in ("atlas_png","atlas_webp"):
        atlas=decode_rgba(payload,shape)
    elif codec=="atlas_raw_zstd":
        raw=zdecompress(payload,int(np.prod(shape)))
        if len(raw)!=int(np.prod(shape)):
            raise ValueError("Incorrect source atlas length")
        atlas=np.frombuffer(raw,dtype=np.uint8).reshape(shape)
    elif codec in ("original_png_zstd","optimized_png_zstd"):
        frames=decode_individual(payload,meta)
        atlas=np.zeros(shape,dtype=np.uint8)
        for f,sprite in zip(frames,meta["sprites"]):
            x,y,w,h=(sprite["x"],sprite["y"],sprite["width"],sprite["height"])
            atlas[y:y+h,x:x+w]=f
    else:raise ValueError("Invalid compressed model")
    if sha(atlas.tobytes())!=meta["atlas_rgba_sha256"]:
        raise ValueError("Padded source atlas pixel check failed")
    result=[]
    for sprite in meta["sprites"]:
        x,y,w,h=(sprite["x"],sprite["y"],sprite["width"],sprite["height"])
        if min(x,y,w,h)<0 or x+w>shape[1] or y+h>shape[0]:
            raise ValueError("Invalid source atlas crop bound")
        cropped=np.ascontiguousarray(atlas[y:y+h,x:x+w])
        if cropped.shape!=(h,w,4) or sha(cropped.tobytes())!=sprite["pixel_rgba_sha256"]:
            raise ValueError("Exact source RGBA—including hidden transparent RGB—was corrupted")
        result.append(cropped)
    return result,meta


def self_test():
    rng=np.random.default_rng(78)
    sprites=[]
    for i in range(32):
        w,h=(22+(i%7)*3,26+(i%5)*4)
        rgba=rng.integers(0,256,(h,w,4),dtype=np.uint8)
        rgba[:,:,3]=(rgba[:,:,3]//4)*4
        with io.BytesIO() as f:
            Image.fromarray(rgba,mode="RGBA").save(f,format="PNG")
            raw=f.getvalue()
        sprites.append({
            "rgba":rgba,"raw":raw,
            "meta":{"filename":f"synthetic_{i}.png",
                    "source_png_sha256":sha(raw),"pixel_rgba_sha256":sha(rgba.tobytes()),
                    "width":w,"height":h,"alpha_transparent_pixel_count":0,
                    "alpha_partial_pixel_count":0}
        })
    atlas,layout=pack_assets(sprites)
    manifest=base_manifest("0"*64,100_000,sprites,atlas,layout)
    for mode in CODECS:
        blob,_=encode_case(atlas,sprites,manifest,mode)
        decoded,meta=decode_case(blob)
        assert len(decoded)==32 and all(np.array_equal(a["rgba"],b)
                                        for a,b in zip(sprites,decoded))
        assert meta["codec"]==mode
        if mode.startswith("tile"):
            try:
                decode_case(blob[:-1])
            except (ValueError,zstd.ZstdError,OSError):
                pass
            else:
                raise AssertionError("Truncated tile container was accepted")
    print("EXP78 SELFTEST PASS: all 7 saved full codecs exact on 32 RGBA sprites with hidden RGB")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",default="/tmp/blkh_exp78")
    ap.add_argument("--cache",default="/tmp/blkh_exp78_cache")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:
        self_test();return
    if not features.check("webp"):
        raise RuntimeError("WebP lossless needed for fair conventional atlas")
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    zipped,source_hash=fetch_zip(Path(a.cache))
    images=discover(zipped,source_hash,root)
    report=[]
    for n in NESTED:
        items=images[:n]
        atlas,layout=pack_assets(items)
        shared=base_manifest(source_hash,len(zipped),items,atlas,layout)
        dest=root/f"sprites_{n:03}"
        dest.mkdir(parents=True,exist_ok=True)
        with io.BytesIO() as f:
            Image.fromarray(atlas,mode="RGBA").save(f,format="PNG")
            (dest/"original_rgba_atlas.png").write_bytes(f.getvalue())
        scores={}
        for mode in CODECS:
            started=time.perf_counter_ns()
            raw,stats=encode_case(atlas,items,shared,mode)
            elapsed_encode=(time.perf_counter_ns()-started)/1e9
            file=dest/(mode+".bl78")
            file.write_bytes(raw)
            started=time.perf_counter_ns()
            reconstructed,manifest=decode_case(file.read_bytes())
            elapsed_decode=(time.perf_counter_ns()-started)/1e9
            if manifest!={**shared,"codec":mode} or len(reconstructed)!=n:
                raise AssertionError("Corrupt source manifest or sprite count")
            if not all(np.array_equal(orig["rgba"],got)
                       for orig,got in zip(items,reconstructed)):
                raise AssertionError("Source sprite RGBA not reconstructed exactly")
            scores[mode]={
                "filename":file.name,
                "complete_bytes":len(raw),"sha256":sha(raw),
                "encode_seconds":elapsed_encode,"decode_seconds":elapsed_decode,
                **stats,
            }
        size,chosen=min((scores[m]["complete_bytes"],m)
                        for m in CODECS if m.startswith("tile"))
        comp,comp_codec=min((scores[m]["complete_bytes"],m)
                            for m in CODECS if not m.startswith("tile"))
        row={
            "source_collection":"Kenney UI Pack, one correlated CC0 family",
            "sprite_count":n,
            "source_zip_sha256":source_hash,
            "source_zip_bytes":len(zipped),
            "exact_rgba_manifest":shared,
            "all_sprites_exactly_decoded":True,
            "source_rgba_atlas":"original_rgba_atlas.png",
            "source_rgba_atlas_sha256":sha((dest/"original_rgba_atlas.png").read_bytes()),
            "codecs":scores,
            "best_indexed_dedup":chosen,
            "best_indexed_dedup_bytes":size,
            "best_conventional":comp_codec,
            "best_conventional_bytes":comp,
            "custom_smaller_than_best_conventional":size<comp,
        }
        report.append(row)
        print("EXP78",n,"sprites",
              "dedup",chosen,size,"standard",comp_codec,comp,
              "unique_tiles",scores["tile16_zstd"]["unique_tiles"],
              "win",row["custom_smaller_than_best_conventional"],flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":78,
            "protocol":"research/dedup/BHUH_EXP78_REAL_CC0_RGBA_PROTOCOL.md",
            "source_zip_sha256":source_hash,
            "source_landing":SOURCE_LANDING,
            "license":"CC0-1.0",
            "nested_subset_sizes":list(NESTED),
            "sample_warning":"Only one artist's historical UI sprite pack; not three independent artist families or authored revision histories",
            "software":{"python":platform.python_version(),
                       "numpy":np.__version__,"pillow":PIL_VERSION,
                       "zstandard":zstd.__version__},
            "results":report,
        },indent=2),encoding="utf8")
    print("EXP78 DONE: 32/64/96 actual sourced CC0 UI sprites, ALL 21 complete lossless archives decoded",flush=True)


if __name__=="__main__":
    main()
