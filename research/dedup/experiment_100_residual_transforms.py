"""Exp100: exact RGBA residual coding (raw / XOR / mod-256 subtraction).

Frozen two motion shift selectors x three residual representations on real,
already studied Wesnoth Git PNG history. Retrospective; NOT independent.
All original source metadata and RGBA including hidden RGB are preserved.
"""
from __future__ import annotations
import argparse
import json
import time
import resource
from pathlib import Path

import numpy as np
from experiment_78_real_cc0_rgba_sprites import encode_webp
import experiment_83_pixelorama_git_history as base
import experiment_84_sparse_pixel_patches as sparse
import experiment_87_wesnoth_real_maintenance as source
import experiment_92_motion_shift_patch as motion
from experiment_95_compressed_shift_retrospective import choose_by_compressed_patch

MAGIC=b"BL100"
MODE="temporal_residual8_webpold_zstd"
HEAD=motion.HEAD
LENGTHS=motion.LENGTHS
RECORD=motion.RECORD
TRANSFORMS={"raw":1,"xor":2,"sub":3}
VERSIONS={v:k for k,v in TRANSFORMS.items()}

def residual(new,prior,mode):
    if mode=="raw":
        return new
    if mode=="xor":
        return np.bitwise_xor(new,prior)
    if mode=="sub":
        return np.subtract(new,prior,dtype=np.uint8)
    raise ValueError("Unknown residual mode")

def inverse(value,prior,mode):
    if mode=="raw":
        return value
    if mode=="xor":
        return np.bitwise_xor(value,prior)
    if mode=="sub":
        return np.add(value,prior,dtype=np.uint8)
    raise ValueError("Unknown inverse residual mode")

def plan_shifts(items):
    plans={}
    pairs=list(zip(items[::2],items[1::2]))
    plans["mismatch"]=[motion.select_shift(a["rgba"],b["rgba"]) for a,b in pairs]
    plans["compressed"]=[choose_by_compressed_patch(a["rgba"],b["rgba"]) for a,b in pairs]
    return plans

def encode(items,manifest,shifts,mode,old_webp):
    if mode not in TRANSFORMS or len(items)<2 or len(items)%2 or len(shifts)*2!=len(items):
        raise ValueError("Incomplete old/new source pairs or unexpected residual mode")
    old_atlas,positions=base.pack(items[::2])
    if list(old_atlas.shape)!=manifest["old_atlas_shape"] or positions!=manifest["old_atlas_positions"]:
        raise ValueError("Old atlas metadata inconsistent")
    patch=bytearray()
    count_pixels=0
    for (a,b),(dx,dy,count) in zip(zip(items[::2],items[1::2]),shifts):
        old,new=a["rgba"],b["rgba"]
        predicted=motion.predicted(old,new.shape,dx,dy)
        mask=np.any(predicted!=new,axis=2).reshape(-1)
        n=int(mask.sum())
        if n!=count:raise ValueError("Shift changed-pixel count drift")
        prior=np.ascontiguousarray(predicted.reshape(-1,4)[mask])
        target=np.ascontiguousarray(new.reshape(-1,4)[mask])
        values=residual(target,prior,mode)
        patch.extend(RECORD.pack(dx,dy,n))
        patch.extend(np.packbits(mask,bitorder="little").tobytes())
        patch.extend(values.tobytes())
        count_pixels+=n
    zpatch=sparse.zcompress(bytes(patch))
    metadata=json.dumps({**manifest,"codec":MODE},
                        separators=(",",":"),sort_keys=True).encode()
    if not 0<len(metadata)<sparse.MAX_METADATA:
        raise ValueError("Oversized source manifest")
    stream=(HEAD.pack(MAGIC,TRANSFORMS[mode],len(metadata))+metadata+
            LENGTHS.pack(len(old_webp),len(zpatch))+old_webp+zpatch)
    if len(stream)>sparse.MAX_ARCHIVE:
        raise ValueError("Full archive exceeds source safety limit")
    return stream,{"changed_pixels":count_pixels,
                   "raw_patch_bytes":len(patch),
                   "compressed_patch_bytes":len(zpatch)}

def decode(data):
    if not HEAD.size<len(data)<=sparse.MAX_ARCHIVE:
        raise ValueError("Bad complete archive length")
    magic,version,nmeta=HEAD.unpack_from(data)
    if magic!=MAGIC or version not in VERSIONS or not 0<nmeta<sparse.MAX_METADATA:
        raise ValueError("Wrong version or magic")
    mode=VERSIONS[version]
    at=HEAD.size+nmeta
    if len(data)<=at+LENGTHS.size:
        raise ValueError("Truncated full source stream")
    meta=json.loads(data[HEAD.size:at])
    sprites=meta.get("source_sprites")
    if (meta.get("codec")!=MODE or not isinstance(sprites,list) or
            not sprites or len(sprites)%2 or
            meta.get("historical_pair_count")!=len(sprites)//2):
        raise ValueError("Invalid source revision manifest")
    nold,npatch=LENGTHS.unpack_from(data,at)
    at+=LENGTHS.size
    if nold<1 or npatch<1 or at+nold+npatch!=len(data):
        raise ValueError("Corrupt compressed stream lengths")
    old=sparse.old_atlas_decode(data[at:at+nold],meta)
    ceiling=0
    for item in sprites[1::2]:
        h,w=item["height"],item["width"]
        if not all(isinstance(v,int) for v in (h,w)) or not (1<=h<=1024 and 1<=w<=1024):
            raise ValueError("Invalid output geometry")
        ceiling+=RECORD.size+(h*w+7)//8+4*h*w
    raw=sparse.zdecompress(data[at+nold:],ceiling)
    ptr=0
    newer=[]
    for prior,info in zip(old,sprites[1::2]):
        h,w=info["height"],info["width"]
        pixels=h*w
        masklen=(pixels+7)//8
        if len(raw)-ptr<RECORD.size+masklen:
            raise ValueError("Missing residual record")
        dx,dy,count=RECORD.unpack_from(raw,ptr)
        ptr+=RECORD.size
        packed=raw[ptr:ptr+masklen]
        ptr+=masklen
        if pixels%8 and (packed[-1]>>(pixels%8)):
            raise ValueError("Invalid residual mask padding")
        mask=np.unpackbits(np.frombuffer(packed,dtype=np.uint8),
                           bitorder="little")[:pixels].astype(bool)
        if count>pixels or int(mask.sum())!=count or len(raw)-ptr<4*count:
            raise ValueError("Invalid residual pixel count")
        values=np.frombuffer(raw,dtype=np.uint8,count=count*4,
                             offset=ptr).reshape(count,4)
        ptr+=4*count
        result=motion.predicted(prior,(h,w,4),dx,dy)
        flat=result.reshape(-1,4)
        oldvalue=flat[mask]
        flat[mask]=inverse(values,oldvalue,mode)
        newer.append(result)
    if ptr!=len(raw):
        raise ValueError("Unconsumed residual stream")
    images=[im for pair in zip(old,newer) for im in pair]
    shape=tuple(meta["atlas_shape"])
    if (len(shape)!=3 or shape[2]!=4 or min(shape[:2])<1 or
            shape[0]*shape[1]>3_000_000):
        raise ValueError("Source atlas shape out of bound")
    atlas=np.zeros(shape,dtype=np.uint8)
    if len(images)!=len(sprites):
        raise ValueError("Source sprites count mismatch")
    for src,info in zip(images,sprites):
        x,y,w,h=(info[k] for k in ("x","y","width","height"))
        if (src.shape!=(h,w,4) or x<0 or y<0 or x+w>shape[1] or y+h>shape[0] or
                base.sha(src.tobytes())!=info["pixel_rgba_sha256"]):
            raise ValueError("Wrong exact RGBA reconstruction")
        atlas[y:y+h,x:x+w]=src
    if base.sha(atlas.tobytes())!=meta["atlas_sha256"]:
        raise ValueError("Wrong whole-source atlas checksum")
    return images,meta

def self_test():
    rng=np.random.default_rng(100)
    a=rng.integers(0,256,(11,12,4),dtype=np.uint8)
    b=a.copy();b[2,3]=[255,20,30,0]  # hidden RGB is material
    c=rng.integers(0,256,(16,16,4),dtype=np.uint8)
    d=c[2:15,1:14].copy();d[4,6]=[8,1,2,0]
    items=[]
    for i,(prior,new) in enumerate(((a,b),(c,d))):
        for j,rgba in enumerate((prior,new)):
            items.append({"rgba":rgba,"raw":b"test",
                          "meta":{"width":rgba.shape[1],"height":rgba.shape[0],
                                  "filename":f"test:{i}:{j}",
                                  "role":"old" if j==0 else "new",
                                  "pixel_rgba_sha256":base.sha(rgba.tobytes()),
                                  "source_png_sha256":"0"*64}})
    atlas,positions=base.pack(items)
    old_atlas,oldpos=base.pack(items[::2])
    meta={"source_sprites":positions,"historical_pair_count":2,
          "atlas_shape":list(atlas.shape),"atlas_sha256":base.sha(atlas.tobytes()),
          "old_atlas_shape":list(old_atlas.shape),
          "old_atlas_positions":oldpos,
          "old_atlas_rgba_sha256":base.sha(old_atlas.tobytes())}
    encoded_old=encode_webp(old_atlas)
    plans=plan_shifts(items)
    for selector in ("mismatch","compressed"):
        for mode in TRANSFORMS:
            stream,_=encode(items,meta,plans[selector],mode,encoded_old)
            rebuilt,recovered=decode(stream)
            assert recovered=={**meta,"codec":MODE}
            assert all(np.array_equal(x["rgba"],y) for x,y in zip(items,rebuilt))
            for corrupt in (stream[:-1],stream[:9]):
                try:decode(corrupt)
                except Exception:pass
                else:raise AssertionError("Truncated residual archive accepted")
    print("EXP100 SELFTEST PASS: 6 exact full archives, hidden RGBA, cropped motion",flush=True)

def execute(out):
    out.mkdir(parents=True,exist_ok=True)
    source.configure_sources()
    source.source_license(out)
    items,inventory=source.discover(out)
    assert len(items)==42 and inventory["real_file_pair_count"]==21
    cases=[("all_21_pairs",items)]
    for label,_,paths in source.SOURCE_GROUPS:
        subset=[x for x in items if x["meta"]["commit_label"]==label]
        assert len(subset)==2*len(paths)
        cases.append((label,subset))
    rows=[]
    for name,subset in cases:
        reference=sparse.evaluate(subset,inventory,name,out)
        meta=reference["shared_source_manifest"]
        old_atlas,positions=base.pack(subset[::2])
        assert list(old_atlas.shape)==meta["old_atlas_shape"] and positions==meta["old_atlas_positions"]
        old_bytes=encode_webp(old_atlas)
        start=time.perf_counter()
        plans=plan_shifts(subset)
        selection_seconds=time.perf_counter()-start
        methods={}
        for selector,shifts in plans.items():
            for mode in TRANSFORMS:
                began=time.perf_counter()
                stream,stats=encode(subset,meta,shifts,mode,old_bytes)
                elapsed=time.perf_counter()-began
                folder=out/name;folder.mkdir(parents=True,exist_ok=True)
                filename=f"residual_{selector}_{mode}.bl100"
                (folder/filename).write_bytes(stream)
                images,restored=decode(stream)
                assert restored=={**meta,"codec":MODE}
                assert all(np.array_equal(x["rgba"],y) for x,y in zip(subset,images))
                methods[f"{selector}_{mode}"]=dict(filename=filename,bytes=len(stream),
                    sha256=base.sha(stream),encode_seconds=round(elapsed,6),**stats)
        best=min(methods,key=lambda key:(methods[key]["bytes"],key))
        prior=min(reference["best_sparse_bytes"],reference["best_reference_bytes"])
        rows.append(dict(group=name,real_pairs=len(subset)//2,
            manifest=meta,selection_seconds=round(selection_seconds,6),
            best_candidate=best,best_candidate_bytes=methods[best]["bytes"],
            best_reference_bytes=prior,
            candidate_minus_reference_bytes=methods[best]["bytes"]-prior,
            candidate_methods=methods,reference_results=reference,exact_rgba=True))
        print("EXP100",name,"best",best,methods[best]["bytes"],"reference",prior,
              "delta",methods[best]["bytes"]-prior,flush=True)
    data={"experiment":100,"retrospective":True,"independent_holdout":False,
          "release_eligible":False,"source_repository":source.SOURCE_REPO,
          "source_pairs":21,"total_persisted_archives":80,
          "peak_process_maxrss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "results":rows}
    (out/"results100.json").write_text(json.dumps(data,indent=2)+"\n")
    print("EXP100 complete: retrospective, no public release claim",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="/tmp/blkh_exp100")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:self_test()
    else:execute(Path(args.out))
