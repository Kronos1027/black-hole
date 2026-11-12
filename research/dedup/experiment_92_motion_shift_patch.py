"""BLKH Exp92 frozen exact lossless ±8 integer-motion RGBA patch codec.

Fixed method/protocol in BHUH_EXP92_TEMPORAL_SHIFT8_ENDLESS_SKY_PROTOCOL.md.
The shift candidates, mismatch-only scoring and tie break were frozen before
loading original source pixel values; this method is not a PNG-byte codec.
"""
from __future__ import annotations

import json
import struct
import time

import numpy as np
from experiment_78_real_cc0_rgba_sprites import encode_webp
import experiment_83_pixelorama_git_history as base
import experiment_84_sparse_pixel_patches as sparse

MODE = "temporal_shift8_webpold_zstd"
MAGIC = b"BL92M"
HEAD = struct.Struct("<5sBI")
LENGTHS = struct.Struct("<II")
RECORD = struct.Struct("<bbI")
SHIFT_LIMIT = 8
SHIFTS = tuple(sorted(((dx,dy) for dx in range(-SHIFT_LIMIT,SHIFT_LIMIT+1)
                      for dy in range(-SHIFT_LIMIT,SHIFT_LIMIT+1)),
                     key=lambda shift: (abs(shift[0])+abs(shift[1]),
                                        abs(shift[1]),abs(shift[0]),shift[1],shift[0])))
assert len(SHIFTS) == 289 and SHIFTS[0] == (0,0)


def predicted(old: np.ndarray, shape: tuple[int,int,int], dx: int,dy: int) -> np.ndarray:
    if not -SHIFT_LIMIT <= dx <= SHIFT_LIMIT or not -SHIFT_LIMIT <= dy <= SHIFT_LIMIT:
        raise ValueError("Source motion translation exceeds frozen bound")
    if len(shape)!=3 or shape[2]!=4 or min(shape[:2])<1 or old.ndim!=3 or old.shape[2]!=4:
        raise ValueError("Bad old/new source RGB/A shape")
    h,w,_ = shape
    oh,ow,_ = old.shape
    result = np.zeros(shape,dtype=np.uint8)
    x0=max(0,-dx)
    y0=max(0,-dy)
    x1=min(w,ow-dx)
    y1=min(h,oh-dy)
    if x1>x0 and y1>y0:
        result[y0:y1,x0:x1] = old[y0+dy:y1+dy,x0+dx:x1+dx]
    return result


def select_shift(old: np.ndarray, new: np.ndarray) -> tuple[int,int,int]:
    best = (new.shape[0]*new.shape[1]+1,0,0)
    for dx,dy in SHIFTS:
        mask=np.any(predicted(old,new.shape,dx,dy)!=new,axis=2)
        mismatches=int(np.count_nonzero(mask))
        if mismatches < best[0]:
            best=(mismatches,dx,dy)
            if mismatches==0:
                break
    return best[1],best[2],best[0]


def patch_bytes(items: list[dict]) -> tuple[bytes,dict]:
    if not items or len(items)%2:
        raise ValueError("Nonempty paired original PNG sources required")
    result=bytearray()
    shifts=[]
    changed_pixels=0
    target_pixels=0
    for old,new in zip(items[::2],items[1::2]):
        original=old["rgba"]
        target=new["rgba"]
        dx,dy,count=select_shift(original,target)
        reference=predicted(original,target.shape,dx,dy)
        mask=np.any(reference!=target,axis=2).reshape(-1)
        assert int(mask.sum())==count
        result.extend(RECORD.pack(dx,dy,count))
        result.extend(np.packbits(mask,bitorder="little").tobytes())
        result.extend(np.ascontiguousarray(target.reshape(-1,4)[mask]).tobytes())
        shifts.append({"dx":dx,"dy":dy,"changed_pixel_count":count})
        changed_pixels+=count
        target_pixels+=int(target.shape[0]*target.shape[1])
    return bytes(result),{
        "per_pair_motion":shifts,
        "nonzero_motion_pairs":sum(1 for v in shifts if v["dx"] or v["dy"]),
        "motion_after_pixel_difference_fraction":changed_pixels/target_pixels,
        "changed_pixel_count":changed_pixels,
        "target_pixel_count":target_pixels,
    }


def restore_patch(data: bytes, old: list[np.ndarray], metadata: dict) -> list[np.ndarray]:
    info=metadata["source_sprites"]
    if len(old)*2!=len(info):
        raise ValueError("Old image count incompatible with full metadata")
    restored=[]
    at=0
    for index,prior in enumerate(old):
        item=info[2*index+1]
        h,w=(item[z] for z in ("height","width"))
        if not all(isinstance(v,int) for v in (h,w)) or not (1<=h<=1024 and 1<=w<=1024):
            raise ValueError("Target PNG geometry out of bounds")
        n=h*w
        mask_bytes=(n+7)//8
        if len(data)-at < RECORD.size+mask_bytes:
            raise ValueError("Incomplete motion record or change mask")
        dx,dy,count=RECORD.unpack_from(data,at)
        at+=RECORD.size
        mask=data[at:at+mask_bytes]
        at+=mask_bytes
        if n%8 and (mask[-1] >> (n%8)):
            raise ValueError("Nonzero padding bits after real change mask")
        bits=np.unpackbits(np.frombuffer(mask,dtype=np.uint8),
                           bitorder="little")[:n].astype(bool)
        if count > n or count != int(np.count_nonzero(bits)):
            raise ValueError("Motion patch count doesn't match change mask")
        if len(data)-at < 4*count:
            raise ValueError("Truncated changed RGBA values")
        delta=np.frombuffer(data,dtype=np.uint8,count=4*count,
                            offset=at).reshape(count,4)
        at+=4*count
        base=predicted(prior,(h,w,4),dx,dy)
        base.reshape(-1,4)[bits]=delta
        restored.append(base)
    if at!=len(data):
        raise ValueError("Unconsumed appended motion patch bytes")
    return restored


def encode(items: list[dict], full_meta: dict) -> tuple[bytes,dict]:
    old,positions=base.pack(items[::2])
    if list(old.shape)!=full_meta["old_atlas_shape"] or positions!=full_meta["old_atlas_positions"]:
        raise ValueError("Original historical old-source atlas drift")
    old_bytes=encode_webp(old)
    patches,stats=patch_bytes(items)
    deltas=sparse.zcompress(patches)
    json_bytes=json.dumps({**full_meta,"codec":MODE},
                          sort_keys=True,separators=(",",":")).encode()
    if not 0<len(json_bytes)<sparse.MAX_METADATA:
        raise ValueError("Original full source manifest size invalid")
    stream=(HEAD.pack(MAGIC,1,len(json_bytes))+json_bytes
            +LENGTHS.pack(len(old_bytes),len(deltas))+old_bytes+deltas)
    if len(stream)>sparse.MAX_ARCHIVE:
        raise ValueError("Full motion-containing archive exceeds source safety bound")
    return stream,{**stats,"old_source_encoded_bytes":len(old_bytes),
                   "compressed_motion_patch_bytes":len(deltas)}


def decode(archive: bytes) -> tuple[list[np.ndarray],dict]:
    if not HEAD.size<len(archive)<=sparse.MAX_ARCHIVE:
        raise ValueError("Complete motion archive size invalid")
    magic,version,n=HEAD.unpack_from(archive)
    if magic!=MAGIC or version!=1 or not 0<n<sparse.MAX_METADATA:
        raise ValueError("Wrong exact lossless motion codec version")
    start=HEAD.size+n
    if len(archive)<=start+LENGTHS.size:
        raise ValueError("No original old source image and motion stream")
    meta=json.loads(archive[HEAD.size:start])
    if meta.get("codec")!=MODE or not isinstance(meta.get("source_sprites"),list):
        raise ValueError("Unsupported historical motion manifest")
    originals=meta["source_sprites"]
    if not originals or len(originals)%2:
        raise ValueError("Broken original historical pair manifest")
    nold,npatch=LENGTHS.unpack_from(archive,start)
    at=start+LENGTHS.size
    if nold<1 or npatch<1 or at+nold+npatch!=len(archive):
        raise ValueError("Truncated original old WebP and Zstd motion stream")
    old=sparse.old_atlas_decode(archive[at:at+nold],meta)
    limit=sum(RECORD.size+(info["height"]*info["width"]+7)//8
              +4*info["height"]*info["width"]
              for info in originals[1::2])
    raw=sparse.zdecompress(archive[at+nold:],limit)
    newer=restore_patch(raw,old,meta)
    images=[a for pair in zip(old,newer) for a in pair]
    shape=tuple(meta["atlas_shape"])
    if (len(shape)!=3 or shape[2]!=4 or shape[0]*shape[1]>3_000_000
            or min(shape[:2])<1):
        raise ValueError("Historic source atlas geometry invalid")
    complete=np.zeros(shape,dtype=np.uint8)
    for frame,info in zip(images,originals):
        x,y,w,h=(info[k] for k in ("x","y","width","height"))
        if (frame.shape!=(h,w,4) or x<0 or y<0 or
                x+w>shape[1] or y+h>shape[0] or
                base.sha(frame.tobytes())!=info["pixel_rgba_sha256"]):
            raise ValueError("Actual source historical RGBA image checksum/geometry failure")
        complete[y:y+h,x:x+w]=frame
    if base.sha(complete.tobytes())!=meta["atlas_sha256"]:
        raise ValueError("Source complete atlas checksum mismatch")
    return images,meta


def self_test() -> None:
    rng=np.random.default_rng(92)
    fixtures=[]
    a=rng.integers(0,255,(20,27,4),dtype=np.uint8)
    b=a[3:17,5:21].copy()
    b[2,4]=rng.integers(0,255,(4,),dtype=np.uint8)
    c=np.zeros((16,16,4),dtype=np.uint8)
    d=c.copy()
    d[2,2]=[10,20,30,0]  # hidden RGB beneath transparent alpha is data!
    for i,(old,new) in enumerate(((a,b),(c,d))):
        dx,dy,n=select_shift(old,new)
        expect=predicted(old,new.shape,dx,dy)
        assert int(np.count_nonzero(np.any(expect!=new,axis=2)))==n
        for j,pic in enumerate((old,new)):
            fixtures.append({"rgba":pic,"meta":{
                "width":pic.shape[1],"height":pic.shape[0],
                "pixel_rgba_sha256":base.sha(pic.tobytes()),
                "filename":f"selftest:{i}:{j}","role":"old" if not j else "new",
                "source_png_sha256":"x"*64,
            },"raw":b"test-original-png"})
    old_atlas,positions=base.pack(fixtures[::2])
    atlas,pos=base.pack(fixtures)
    dummy={"atlas_shape":list(atlas.shape),"atlas_sha256":base.sha(atlas.tobytes()),
           "source_sprites":pos,"old_atlas_shape":list(old_atlas.shape),
           "old_atlas_positions":positions,
           "old_atlas_rgba_sha256":base.sha(old_atlas.tobytes())}
    encoded,details=encode(fixtures,dummy)
    reread,meta=decode(encoded)
    assert meta=={**dummy,"codec":MODE}
    assert all(np.array_equal(x["rgba"],y) for x,y in zip(fixtures,reread))
    assert details["nonzero_motion_pairs"]>=1
    for corrupt in (encoded[:-1],encoded[:12],encoded[:HEAD.size+2]):
        try:decode(corrupt)
        except (ValueError,RuntimeError,Exception):
            pass
        else:
            raise AssertionError("Truncated motion record unexpectedly accepted")
    print("EXP92 NEW MOTION CODEC SELFTEST PASS: 2 source pairs, crop translation, hidden alpha RGB, full metadata/archive exact decoding",flush=True)
