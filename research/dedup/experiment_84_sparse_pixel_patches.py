"""
BLKH Exp84 — exact sparse changed-pixel patch archives for TRUE authored Git edits.

No synthetic edits or source selection changes. Exact same original
Pixelorama 19 PNG old/new pairs as Exp83P; compare to all eleven archived
codecs and independently implemented bsdiff4 binary delta reference.
Preregistered protocol BHUH_EXP84_SPARSE_PIXEL_PROTOCOL.md.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import platform
import struct
import time

import bsdiff4
import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import zstandard as zstd

from experiment_83_pixelorama_git_history import (
    COMMIT_GROUPS, discover, pack, manifest_for,
    encode as legacy_encode,decode as legacy_decode,
    CANDIDATES as OLD_CANDIDATES,REFERENCES as OLD_REFERENCES,
    sha,
)
from experiment_78_real_cc0_rgba_sprites import (
    encode_webp,decode_rgba,zcompress,zdecompress
)

NEW=("sparse_webpold_zstd","sparse_pngold_zstd")
BINARY="bsdiff_png_zstd"
OLD=OLD_CANDIDATES+OLD_REFERENCES
ALL=NEW+(BINARY,)+OLD
HEAD=struct.Struct("<5sBI")
PART=struct.Struct("<II")
MAGIC=b"BL84S"
MAX_ARCHIVE=25_000_000
MAX_METADATA=250_000
MAX_SOURCE_FILE=5_000_000


def shared_manifest(items:list[dict],inventory:dict)->dict:
    if not items or len(items)%2:
        raise ValueError("Full old/new historical pair count required")
    meta=manifest_for(items,inventory)
    old_atlas,old_positions=pack(items[::2])
    meta["old_atlas_shape"]=list(old_atlas.shape)
    meta["old_atlas_positions"]=old_positions
    meta["old_atlas_rgba_sha256"]=sha(old_atlas.tobytes())
    return meta


def masked_data(items:list[dict])->tuple[bytes,dict]:
    output=bytearray()
    changed_total=pixel_total=0
    changes_per_pair=[]
    for pair in range(0,len(items),2):
        old,new=items[pair:pair+2]
        source=new["rgba"]
        prior=old["rgba"]
        h,w,_=source.shape
        reference=np.zeros_like(source)
        oh=min(h,prior.shape[0]);ow=min(w,prior.shape[1])
        reference[:oh,:ow]=prior[:oh,:ow]
        changed=np.any(source!=reference,axis=2).reshape(-1)
        mask=np.packbits(changed,bitorder="little").tobytes()
        rgba=np.ascontiguousarray(source.reshape(-1,4)[changed]).tobytes()
        count=int(changed.sum())
        output.extend(struct.pack("<I",count))
        output.extend(mask)
        output.extend(rgba)
        changed_total+=count
        pixel_total+=h*w
        changes_per_pair.append(count)
    return bytes(output),{
        "full_target_pixel_count":pixel_total,
        "changed_pixel_count":changed_total,
        "changed_pixel_fraction":changed_total/pixel_total,
        "individual_pair_changed_pixels":changes_per_pair,
        "pixel_change_mask_definition":"1 bit/pixel, row-major, little endian bit position; 4 exact RGBA bytes per changed pixel",
    }


def restore_masks(raw:bytes,old:list[np.ndarray],meta:dict)->list[np.ndarray]:
    sprites=meta["source_sprites"]
    if len(old)*2!=len(sprites):
        raise ValueError("Original old sprite count inconsistent")
    at=0
    decoded=[]
    for pair,prior in enumerate(old):
        new_info=sprites[2*pair+1]
        h=new_info["height"];w=new_info["width"]
        if not(1<=h<=1024 and 1<=w<=1024):
            raise ValueError("Bad new historical image geometry")
        n=h*w;mask_bytes=(n+7)//8
        if len(raw)-at<4+mask_bytes:raise ValueError("Truncated source patch mask")
        count=struct.unpack_from("<I",raw,at)[0];at+=4
        packed=raw[at:at+mask_bytes];at+=mask_bytes
        if n%8 and (packed[-1]>>(n%8)):
            raise ValueError("Nonzero trailing mask padding")
        changed=np.unpackbits(np.frombuffer(packed,dtype=np.uint8),
                              bitorder="little")[:n].astype(bool)
        if count!=int(changed.sum()) or count>n:
            raise ValueError("Bad changed pixel count")
        if len(raw)-at<count*4:
            raise ValueError("Truncated source replacement RGBA pixel array")
        values=np.frombuffer(raw,dtype=np.uint8,count=count*4,
                             offset=at).reshape(count,4)
        at+=count*4
        rebuilt=np.zeros((h,w,4),dtype=np.uint8)
        oh=min(h,prior.shape[0]);ow=min(w,prior.shape[1])
        rebuilt[:oh,:ow]=prior[:oh,:ow]
        rebuilt.reshape(-1,4)[changed]=values
        decoded.append(rebuilt)
    if at!=len(raw):raise ValueError("Trailing unconsumed changed-pixel bytes")
    return decoded


def encode_old_original_pngs(items:list[dict])->bytes:
    raw=bytearray()
    for item in items[::2]:
        png=item["raw"]
        if len(png)>MAX_SOURCE_FILE:raise ValueError("Oversized original historical PNG")
        raw.extend(struct.pack("<I",len(png)))
        raw.extend(png)
    return zcompress(bytes(raw))


def old_original_pngs_decode(data:bytes,meta:dict)->list[np.ndarray]:
    refs=meta["source_sprites"][::2]
    data=zdecompress(data,len(refs)*(MAX_SOURCE_FILE+4))
    cursor=0;images=[]
    for info in refs:
        if len(data)-cursor<4:raise ValueError("Truncated old PNG source length")
        size=struct.unpack_from("<I",data,cursor)[0];cursor+=4
        if size<8 or size>MAX_SOURCE_FILE or size>len(data)-cursor:
            raise ValueError("Invalid old historical PNG member")
        orig=data[cursor:cursor+size];cursor+=size
        if sha(orig)!=info["source_png_sha256"]:
            raise ValueError("Historical old PNG byte hash mismatch")
        images.append(decode_rgba(orig,(info["height"],info["width"],4)))
    if cursor!=len(data):raise ValueError("Additional old source bytes")
    return images


def old_atlas_decode(blob:bytes,meta:dict)->list[np.ndarray]:
    shape=tuple(meta["old_atlas_shape"])
    if len(shape)!=3 or shape[2]!=4 or np.prod(shape)>3_000_000:
        raise ValueError("Old atlas dimensions invalid")
    base=decode_rgba(blob,shape)
    if sha(base.tobytes())!=meta["old_atlas_rgba_sha256"]:
        raise ValueError("Old atlas RGBA checksum mismatch")
    out=[]
    for old in meta["old_atlas_positions"]:
        x,y,w,h=(old[z] for z in ("x","y","width","height"))
        if min(x,y)<0 or min(w,h)<1 or x+w>shape[1] or y+h>shape[0]:
            raise ValueError("Old sprite location outside old atlas")
        out.append(np.ascontiguousarray(base[y:y+h,x:x+w]))
    return out


def binary_originals(items:list[dict])->bytes:
    packed=bytearray()
    for old,new in zip(items[::2],items[1::2]):
        original=old["raw"];target=new["raw"]
        if len(original)>MAX_SOURCE_FILE or len(target)>MAX_SOURCE_FILE:
            raise ValueError("Git source PNG too big")
        patch=bsdiff4.diff(original,target)
        packed.extend(PART.pack(len(original),len(patch)))
        packed.extend(original);packed.extend(patch)
    return zcompress(bytes(packed))


def decode_binary_originals(data:bytes,meta:dict)->list[np.ndarray]:
    raw=zdecompress(data,(len(meta["source_sprites"])//2)*(2*MAX_SOURCE_FILE+8))
    parts=[];cursor=0
    for old_info,new_info in zip(meta["source_sprites"][::2],
                                 meta["source_sprites"][1::2]):
        if len(raw)-cursor<PART.size:
            raise ValueError("Incomplete binary PNG patch lengths")
        length,psize=PART.unpack_from(raw,cursor);cursor+=PART.size
        if length>MAX_SOURCE_FILE or psize>MAX_SOURCE_FILE*2:
            raise ValueError("Oversized historical PNG source/binary delta")
        if len(raw)-cursor<length+psize:
            raise ValueError("Truncated binary source or bsdiff patch")
        oldpng=raw[cursor:cursor+length];cursor+=length
        delta=raw[cursor:cursor+psize];cursor+=psize
        if sha(oldpng)!=old_info["source_png_sha256"]:
            raise ValueError("Old historical PNG bytes modified")
        newpng=bsdiff4.patch(oldpng,delta)
        if sha(newpng)!=new_info["source_png_sha256"]:
            raise ValueError("New original PNG bytes cannot be recovered")
        parts.append(decode_rgba(oldpng,(old_info["height"],old_info["width"],4)))
        parts.append(decode_rgba(newpng,(new_info["height"],new_info["width"],4)))
    if cursor!=len(raw):raise ValueError("Trailing unaccounted bsdiff patch bytes")
    return parts


def encode(items:list[dict],meta:dict,mode:str)->tuple[bytes,dict]:
    if mode not in NEW+(BINARY,):
        raise ValueError("Unknown experimental real source codec")
    if mode in NEW:
        patch,stats=masked_data(items)
        if mode=="sparse_webpold_zstd":
            old,positions=pack(items[::2])
            if list(old.shape)!=meta["old_atlas_shape"] or positions!=meta["old_atlas_positions"]:
                raise ValueError("Source old atlas geometry changed")
            baseline=encode_webp(old)
        else:
            baseline=encode_old_original_pngs(items)
        changes=zcompress(patch)
        streams=PART.pack(len(baseline),len(changes))+baseline+changes
        stats={**stats,"old_source_encoded_bytes":len(baseline),
               "compressed_patch_bytes":len(changes)}
    else:
        streams=binary_originals(items)
        stats={"bsdiff_zstd_stream_bytes":len(streams)}
    metadata=json.dumps({**meta,"codec":mode},sort_keys=True,
                        separators=(",",":")).encode()
    if len(metadata)>=MAX_METADATA:
        raise ValueError("Historical source metadata too large")
    return HEAD.pack(MAGIC,1,len(metadata))+metadata+streams,stats


def decode(archive:bytes)->tuple[list[np.ndarray],dict]:
    if not HEAD.size<len(archive)<=MAX_ARCHIVE:
        raise ValueError("Bad archive length")
    magic,version,length=HEAD.unpack(archive[:HEAD.size])
    if magic!=MAGIC or version!=1 or not 0<length<MAX_METADATA:
        raise ValueError("Wrong BL84 version")
    end=HEAD.size+length
    if len(archive)<=end:raise ValueError("Missing source history binary stream")
    meta=json.loads(archive[HEAD.size:end])
    mode=meta.get("codec")
    if mode not in NEW+(BINARY,) or len(meta.get("source_sprites",[]))%2:
        raise ValueError("Unsupported real history source codec or metadata")
    binary=archive[end:]
    if mode==BINARY:
        images=decode_binary_originals(binary,meta)
    else:
        if len(binary)<PART.size:raise ValueError("Truncated source and patch lengths")
        nold,nchanges=PART.unpack_from(binary)
        if nold<1 or nchanges<1 or PART.size+nold+nchanges!=len(binary):
            raise ValueError("Invalid complete source-stream length")
        begin=PART.size
        oldraw=binary[begin:begin+nold]
        changedraw=binary[begin+nold:]
        olds=(old_atlas_decode(oldraw,meta)
              if mode=="sparse_webpold_zstd"
              else old_original_pngs_decode(oldraw,meta))
        pixel_count=sum(ref["width"]*ref["height"]
                        for ref in meta["source_sprites"][1::2])
        # masks=ceil(pixel_count/8) per image + 4 byte count + max 4 bytes/pixel
        ceiling=sum(4+(ref["width"]*ref["height"]+7)//8+
                    ref["width"]*ref["height"]*4
                    for ref in meta["source_sprites"][1::2])
        patches=zdecompress(changedraw,ceiling)
        new=restore_masks(patches,olds,meta)
        images=[src for pair in zip(olds,new) for src in pair]
    if len(images)!=len(meta["source_sprites"]):
        raise ValueError("Wrong source history sprite count")
    full=np.zeros(tuple(meta["atlas_shape"]),dtype=np.uint8)
    for source,info in zip(images,meta["source_sprites"]):
        x,y,w,h=(info[k] for k in ("x","y","width","height"))
        if (source.shape!=(h,w,4) or sha(source.tobytes())!=info["pixel_rgba_sha256"]
                or min(x,y)<0 or min(w,h)<1
                or x+w>full.shape[1] or y+h>full.shape[0]):
            raise ValueError("Decoded original Git sprite bytes do not match")
        full[y:y+h,x:x+w]=source
    if sha(full.tobytes())!=meta["atlas_sha256"]:
        raise ValueError("Full file original RGBA atlas checksum mismatch")
    return images,meta


def self_test()->None:
    rng=np.random.default_rng(84)
    assets=[]
    for i in range(3):
        original=rng.integers(0,256,(20+i,27+i,4),dtype=np.uint8)
        changed=np.pad(original,((0,2),(0,2),(0,0)))[:22+i,:29+i].copy()
        changed[i:i+2,i:i+2]=rng.integers(0,256,(2,2,4),dtype=np.uint8)
        for role,arr in (("old",original),("new",changed)):
            with io.BytesIO() as f:
                Image.fromarray(arr,mode="RGBA").save(f,format="PNG")
                raw=f.getvalue()
            assets.append({"rgba":arr,"raw":raw,"meta":{
                "width":arr.shape[1],"height":arr.shape[0],
                "filename":f"test{i}:{role}","role":role,
                "source_png_sha256":sha(raw),
                "pixel_rgba_sha256":sha(arr.tobytes()),
            }})
    inv={"revision_groups":{"test":{
        "commit":"0"*40,"first_parent":"1"*40,
        "png_pairs_modified_in_git":3,"png_pairs_really_changed_rgba":3,
    }}}
    meta=shared_manifest(assets,inv)
    for codec in NEW+(BINARY,):
        encoded,_=encode(assets,meta,codec)
        output,provenance=decode(encoded)
        assert provenance=={**meta,"codec":codec}
        assert all(np.array_equal(x["rgba"],y) for x,y in zip(assets,output))
        if codec in NEW:
            try:decode(encoded[:-1])
            except (ValueError,zstd.ZstdError,OSError):pass
            else:raise AssertionError("Truncated sparse archive accepted")
    print("EXP84 SELFTEST PASS: both sparse masks and actual PNG bsdiff byte-exact, changed image sizes, RGBA alpha channels")


def evaluate(items:list[dict],inv:dict,group:str,root:Path)->dict:
    meta=shared_manifest(items,inv)
    dest=root/group;dest.mkdir(parents=True,exist_ok=True)
    files={}
    for mode in ALL:
        t=time.perf_counter_ns()
        if mode in NEW+(BINARY,):
            stream,details=encode(items,meta,mode)
        else:
            stream,details=legacy_encode(items,meta,mode)
        encode_secs=(time.perf_counter_ns()-t)/1e9
        path=dest/(mode+".bl84")
        path.write_bytes(stream)
        t=time.perf_counter_ns()
        saved=path.read_bytes()
        returned,info=(decode(saved) if mode in NEW+(BINARY,)
                       else legacy_decode(saved))
        decode_secs=(time.perf_counter_ns()-t)/1e9
        if info!={**meta,"codec":mode} or not all(
                np.array_equal(original["rgba"],actual)
                for original,actual in zip(items,returned)):
            raise AssertionError("Historical output not exact RGBA byte-verified")
        files[mode]={
            "filename":path.name,"bytes":len(stream),"sha256":sha(stream),
            "encode_seconds":encode_secs,"decode_seconds":decode_secs,**details,
        }
    custom,custom_codec=min((files[m]["bytes"],m) for m in NEW)
    baseline,basecodec=min((files[m]["bytes"],m) for m in (BINARY,)+OLD)
    _,patch_stats=masked_data(items)
    return {
        "group":group,"real_old_new_png_pairs":len(items)//2,
        "shared_source_manifest":meta,
        "changed_rgba_pixel_statistics":patch_stats,
        "methods":files,
        "best_sparse_codec":custom_codec,"best_sparse_bytes":custom,
        "best_of_12_strong_references":basecodec,
        "best_reference_bytes":baseline,
        "sparse_smaller_than_best_reference":custom<baseline,
        "all_original_rgba_exact":True,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="/tmp/blkh_exp84")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:self_test();return
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    items,inventory=discover(root)
    assert len(items)==38
    groups=[("all_19_pairs",items)]
    for label,commit,paths in COMMIT_GROUPS:
        subset=[a for a in items if a["meta"]["commit_label"]==label]
        assert len(subset)==len(paths)*2
        groups.append((label,subset))
    report=[]
    for name,sources in groups:
        row=evaluate(sources,inventory,name,root)
        report.append(row)
        print("EXP84",name,"real changed RGBA pixel fraction",
              round(row["changed_rgba_pixel_statistics"]["changed_pixel_fraction"],5),
              "sparse",row["best_sparse_codec"],row["best_sparse_bytes"],
              "strongest of 12",row["best_of_12_strong_references"],
              row["best_reference_bytes"],"win",
              row["sparse_smaller_than_best_reference"],flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":84,
            "protocol":"research/dedup/BHUH_EXP84_SPARSE_PIXEL_PROTOCOL.md",
            "source":"Real pinned Pixelorama Git edits, same studied 19 source pairs as Exp83P",
            "source_is_new_holdout":False,
            "pair_count":19,"source_image_count":38,
            "new_sparse_candidates":list(NEW),
            "bsdiff_external_baseline":BINARY,
            "other_prior_references":list(OLD),
            "versions":{"python":platform.python_version(),
                        "numpy":np.__version__,"Pillow":PIL_VERSION,
                        "zstd":zstd.__version__},
            "results":report,
        },indent=2))
    assert len(report)==4
    percommit=sum(r["sparse_smaller_than_best_reference"] for r in report[1:])
    passed=report[0]["sparse_smaller_than_best_reference"] and percommit>=2
    print("EXP84 FINAL 19 real pair aggregate + >=2/3 history groups success",passed,
          "percommit wins",percommit,flush=True)


if __name__=="__main__":
    main()
