"""
BLKH Exp73 — trained static entropy tables + reversible Paeth + true rANS.

The PROBABILITY MODEL (six context frequency tables) is fitted on three
photos only, then frozen before compressing 2 more real source photos.
The compression method is classic RANS; it is NOT a new neural compressor.
Complete 5-photo model-and-payload bundle is compared to *joint packed* and
individually saved PNG/WebP/JPEG2000 lossless baselines, all using the same
counted metadata. All output pixels are decoded from saved bytes.

See: research/conditional/BHUH_EXP73_PROTOCOL.md
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import struct
import time
import zlib

import numpy as np
from PIL import Image, __version__ as PIL_VERSION, features
import skimage
from skimage import data

TRAIN=("astronaut","coffee","chelsea")
HOLDOUT=("hubble_deep_field","immunohistochemistry")
SOURCES=TRAIN+HOLDOUT
SIDE=128
CHANNELS=3
P=12
TOTAL=1<<P
L=1<<23
EDGE_THRESHOLD=24
TABLE_COUNT=CHANNELS*2
TABLE_BYTES=TABLE_COUNT*256*2
HEAD=struct.Struct("<5sBI")
MAGIC=b"BL73R"
BASE=b"BL73B"
VERSION=1
SIZE_CAP=2_000_000
BASES=("png","webp_lossless","jpeg2000_lossless")
DATASET_NOTES="scikit-image standard sample images; upstream image provenance/licensing applies"

def sha(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()

def get_photo(name:str)->tuple[np.ndarray,dict]:
    if name not in SOURCES:
        raise ValueError("Photo outside precommitted source list")
    image=np.asarray(getattr(data,name)())
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError("Source not uint8 RGB")
    h,w,_=image.shape
    y=(h-SIDE)//2;x=(w-SIDE)//2
    crop=np.ascontiguousarray(image[y:y+SIDE,x:x+SIDE])
    assert crop.shape==(SIDE,SIDE,3)
    return crop,{
        "source":name,
        "full_source_shape":list(image.shape),
        "original_photo_sha256":sha(image.tobytes()),
        "crop_yx":[y,x],
        "crop_shape":[SIDE,SIDE,3],
        "crop_sha256":sha(crop.tobytes()),
        "trained_probability_source":name in TRAIN,
    }

def paeth(left:int,up:int,up_left:int)->int:
    p=left+up-up_left
    a=abs(p-left);b=abs(p-up);c=abs(p-up_left)
    return left if a<=b and a<=c else up if b<=c else up_left

def context(left:int,up:int,channel:int)->int:
    return CHANNELS*int(abs(left-up)>EDGE_THRESHOLD)+channel

def rgb_to_tokens(image:np.ndarray):
    h,w,_=image.shape
    if (h,w)!=(SIDE,SIDE) or image.dtype!=np.uint8:
        raise ValueError("RGB size mismatch")
    toks=[]
    residuals=bytearray()
    for y in range(h):
        for x in range(w):
            for ch in range(CHANNELS):
                left=int(image[y,x-1,ch]) if x else 0
                up=int(image[y-1,x,ch]) if y else 0
                ul=int(image[y-1,x-1,ch]) if x and y else 0
                sym=(int(image[y,x,ch])-paeth(left,up,ul)) & 255
                toks.append((context(left,up,ch),sym))
                residuals.append(sym)
    return toks,bytes(residuals)

def normalize_counts(counts:np.ndarray)->np.ndarray:
    if counts.shape!=(TABLE_COUNT,256):
        raise ValueError("Expected exactly 6 residual histograms")
    out=np.zeros((TABLE_COUNT,256),dtype=np.uint16)
    for ch in range(TABLE_COUNT):
        mass=counts[ch].astype(np.float64)
        if np.any(mass<0) or not mass.sum():
            raise ValueError("Invalid counts")
        ideal=mass/mass.sum()*(TOTAL-256)
        raw=np.floor(ideal).astype(np.int64)+1
        remaining=TOTAL-int(raw.sum())
        if remaining<0 or remaining>256:
            raise ValueError("Bad frequency normalization")
        fractional=ideal-np.floor(ideal)
        order=np.argsort(-fractional,kind="stable")
        raw[order[:remaining]]+=1
        if raw.sum()!=TOTAL or np.any(raw<=0):
            raise AssertionError("Normalized rANS table invalid")
        out[ch]=raw.astype(np.uint16)
    return out

def train_tables(images:dict[str,np.ndarray])->tuple[np.ndarray,dict]:
    if set(images)!=set(TRAIN):
        raise ValueError("Probability model trained with wrong source set")
    counts=np.ones((TABLE_COUNT,256),dtype=np.int64)
    tokens=0
    for name in TRAIN:
        seq,_=rgb_to_tokens(images[name])
        tokens+=len(seq)
        for ctx,sym in seq:
            counts[ctx,sym]+=1
    freq=normalize_counts(counts)
    return freq,{
        "training_sources":list(TRAIN),
        "total_training_residuals":tokens,
        "contexts":TABLE_COUNT,
        "table_sum_per_context":TOTAL,
        "symbols_per_context":256,
        "predictor":"Paeth, zeros at image border",
        "edge_threshold":EDGE_THRESHOLD,
        "entropy_coder":"standard byte-renormalized 32-bit rANS",
        "byte_order":"little_endian",
    }

def prepare_freq(freq:np.ndarray):
    if freq.shape!=(TABLE_COUNT,256) or freq.dtype!=np.uint16:
        raise ValueError("Invalid frequency table shape/dtype")
    if np.any(freq==0) or np.any(freq.sum(axis=1)!=TOTAL):
        raise ValueError("All model tables must be normalized and positive")
    starts=np.cumsum(freq.astype(np.int32),axis=1)-freq.astype(np.int32)
    inverse=np.empty((TABLE_COUNT,TOTAL),dtype=np.uint8)
    for c in range(TABLE_COUNT):
        for s in range(256):
            a=int(starts[c,s]);b=a+int(freq[c,s])
            inverse[c,a:b]=s
    return starts,inverse

def rans_encode(tokens:list[tuple[int,int]],freq:np.ndarray,starts:np.ndarray)->bytes:
    state=L
    byte_stack=bytearray()
    for c,s in reversed(tokens):
        f=int(freq[c,s]);start=int(starts[c,s])
        maximum=((L>>P)<<8)*f
        while state>=maximum:
            byte_stack.append(state&255)
            state>>=8
        state=((state//f)<<P)+(state%f)+start
        if state>=1<<32:
            raise ValueError("rANS state overflow")
    return state.to_bytes(4,"little")+bytes(reversed(byte_stack))

def rans_decode_rgb(stream:bytes,freq:np.ndarray,starts:np.ndarray,
                    inverse:np.ndarray)->np.ndarray:
    if not 4<=len(stream)<=SIZE_CAP:
        raise ValueError("rANS payload length invalid")
    state=int.from_bytes(stream[:4],"little")
    offset=4
    result=np.empty((SIDE,SIDE,3),dtype=np.uint8)
    for y in range(SIDE):
        for x in range(SIDE):
            for ch in range(CHANNELS):
                left=int(result[y,x-1,ch]) if x else 0
                up=int(result[y-1,x,ch]) if y else 0
                ul=int(result[y-1,x-1,ch]) if x and y else 0
                c=context(left,up,ch)
                slot=state&(TOTAL-1)
                sym=int(inverse[c,slot])
                state=int(freq[c,sym])*(state>>P)+slot-int(starts[c,sym])
                while state<L:
                    if offset>=len(stream):
                        raise ValueError("Truncated rANS encoded payload")
                    state=(state<<8)|stream[offset]
                    offset+=1
                result[y,x,ch]=(paeth(left,up,ul)+sym)&255
    if offset!=len(stream) or state!=L:
        raise ValueError("Unconsumed/invalid trailing rANS state")
    return result

def model_bundle(images:dict[str,np.ndarray],manifest:dict)->tuple[bytes,dict]:
    freq,learned=train_tables(images)
    starts,_=prepare_freq(freq)
    table=freq.astype("<u2",copy=False).tobytes()
    payload=bytearray()
    details={}
    for name in SOURCES:
        seq,_=rgb_to_tokens(images[name])
        start=time.perf_counter()
        bitstream=rans_encode(seq,freq,starts)
        seconds=time.perf_counter()-start
        payload.extend(struct.pack("<I",len(bitstream)))
        payload.extend(bitstream)
        details[name]={
            "payload_bytes_excluding_shared_model":len(bitstream),
            "payload_sha256":sha(bitstream),
            "rans_encoding_seconds":seconds,
        }
    meta={**manifest,"model":learned,"model_table_sha256":sha(table)}
    metabytes=json.dumps(meta,sort_keys=True,separators=(",",":")).encode()
    raw=HEAD.pack(MAGIC,VERSION,len(metabytes))+metabytes+table+payload
    return raw,{
        "shared_model_table_bytes":len(table),
        "source_entropy_details":details,
        "shared_model_table_sha256":sha(table),
        "counted_complete_bundle_bytes":len(raw),
    }

def decode_bundle(blob:bytes)->tuple[dict[str,np.ndarray],dict]:
    if len(blob)<HEAD.size+TABLE_BYTES or len(blob)>SIZE_CAP:
        raise ValueError("Invalid complete learned entropy bundle length")
    magic,version,mlen=HEAD.unpack(blob[:HEAD.size])
    if magic!=MAGIC or version!=VERSION or not 0<mlen<32768:
        raise ValueError("Unsupported BL73 archive")
    mstart=HEAD.size
    end=mstart+mlen
    if len(blob)<end+TABLE_BYTES:
        raise ValueError("Incomplete model/manifest")
    manifest=json.loads(blob[mstart:end])
    if (not isinstance(manifest,dict)
        or manifest.get("sources")!=list(SOURCES)
        or manifest.get("format")!="uint8_RGB_128x128"):
        raise ValueError("Source manifest mismatch")
    table=blob[end:end+TABLE_BYTES]
    if sha(table)!=manifest["model_table_sha256"]:
        raise ValueError("Corrupted shared probability model")
    freq=np.frombuffer(table,dtype="<u2").astype(np.uint16).reshape(TABLE_COUNT,256)
    starts,inverse=prepare_freq(freq)
    offset=end+TABLE_BYTES
    result={}
    for name in SOURCES:
        if len(blob)-offset<4:
            raise ValueError("Missing RGB entropy stream length")
        count=struct.unpack("<I",blob[offset:offset+4])[0]
        offset+=4
        if count<4 or count>SIZE_CAP or count>len(blob)-offset:
            raise ValueError("Truncated/oversized RGB entropy stream")
        coded=blob[offset:offset+count]
        offset+=count
        image=rans_decode_rgb(coded,freq,starts,inverse)
        if sha(image.tobytes())!=manifest["sources_manifest"][name]["crop_sha256"]:
            raise ValueError("Decoded image checksum mismatch")
        result[name]=image
    if offset!=len(blob):
        raise ValueError("Unexpected trailing bundle bytes")
    return result,manifest

def paeth_unresidual(raw:bytes)->np.ndarray:
    if len(raw)!=SIDE*SIDE*CHANNELS:
        raise ValueError("Invalid residual byte array size")
    out=np.empty((SIDE,SIDE,CHANNELS),dtype=np.uint8)
    cursor=0
    for y in range(SIDE):
        for x in range(SIDE):
            for c in range(CHANNELS):
                l=int(out[y,x-1,c]) if x else 0
                u=int(out[y-1,x,c]) if y else 0
                ul=int(out[y-1,x-1,c]) if x and y else 0
                out[y,x,c]=(paeth(l,u,ul)+raw[cursor])&255
                cursor+=1
    return out

def zlib_bundle(images:dict[str,np.ndarray],manifest:dict)->tuple[bytes,dict]:
    chunks=bytearray()
    part_sizes={}
    for name in SOURCES:
        _,resid=rgb_to_tokens(images[name])
        compressed=zlib.compress(resid,9)
        chunks.extend(struct.pack("<I",len(compressed)))
        chunks.extend(compressed)
        part_sizes[name]=len(compressed)
    meta=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
    return HEAD.pack(b"BL73Z",VERSION,len(meta))+meta+chunks,part_sizes

def decode_zlib_bundle(blob:bytes)->dict[str,np.ndarray]:
    magic,ver,n=HEAD.unpack(blob[:HEAD.size])
    if magic!=b"BL73Z" or ver!=VERSION:raise ValueError("Not a zlib ablation bundle")
    manifest=json.loads(blob[HEAD.size:HEAD.size+n])
    ptr=HEAD.size+n
    results={}
    for name in SOURCES:
        if ptr+4>len(blob):raise ValueError("Truncated zlib length")
        count=struct.unpack("<I",blob[ptr:ptr+4])[0];ptr+=4
        raw=blob[ptr:ptr+count];ptr+=count
        decompressor=zlib.decompressobj()
        restored=decompressor.decompress(raw,SIDE*SIDE*3+1)
        if len(restored)!=SIDE*SIDE*3 or not decompressor.eof or decompressor.unused_data:
            raise ValueError("Invalid zlib residual chunk")
        img=paeth_unresidual(restored)
        if sha(img.tobytes())!=manifest["sources_manifest"][name]["crop_sha256"]:
            raise ValueError("Zlib reconstructed wrong pixels")
        results[name]=img
    if ptr!=len(blob):raise ValueError("Trailing zlib archive bytes")
    return results

def encode_baseline(image:np.ndarray,codec:str)->bytes:
    with io.BytesIO() as f:
        im=Image.fromarray(image)
        if codec=="png":
            im.save(f,format="PNG",optimize=True,compress_level=9)
        elif codec=="webp_lossless":
            im.save(f,format="WEBP",lossless=True,method=6)
        elif codec=="jpeg2000_lossless":
            im.save(f,format="JPEG2000",irreversible=False)
        else:raise ValueError("Unsupported exact codec")
        return f.getvalue()

def decode_baseline(raw:bytes,shape:tuple[int,int,int])->np.ndarray:
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        arr=np.asarray(im.convert("RGB"),dtype=np.uint8)
    if arr.shape!=shape:
        raise ValueError("Invalid decoded baseline dimensions")
    return arr

def make_manifest(images:dict[str,np.ndarray],source_details:dict)->dict:
    return {
        "sources":list(SOURCES),
        "sources_manifest":source_details,
        "trained_on":list(TRAIN),
        "held_out_from_probability_training":list(HOLDOUT),
        "format":"uint8_RGB_128x128",
        "atlas_layout":"one row, five original RGB crops side by side",
        "atlas_shape":[SIDE,5*SIDE,3],
        "codec_schema":"BL73R-v1",
        "data_notes":DATASET_NOTES,
    }

def benchmark(root:Path)->dict:
    images={};sources={}
    for name in SOURCES:
        image,detail=get_photo(name)
        images[name]=image
        sources[name]=detail
    manifest=make_manifest(images,sources)
    root.mkdir(parents=True,exist_ok=True)
    for name in SOURCES:
        Image.fromarray(images[name]).save(root/f"source_{name}.png")
    start=time.perf_counter()
    model,info=model_bundle(images,manifest)
    enc_s=time.perf_counter()-start
    learnt_file=root/"learned_rans_all_five.bl73"
    learnt_file.write_bytes(model)
    start=time.perf_counter()
    output,decoded_manifest=decode_bundle(learnt_file.read_bytes())
    dec_s=time.perf_counter()-start
    assert decoded_manifest["sources_manifest"]==sources
    assert all(np.array_equal(images[k],output[k]) for k in SOURCES)
    start=time.perf_counter()
    zdata,zdetails=zlib_bundle(images,manifest)
    zlib_enc_s=time.perf_counter()-start
    (root/"paeth_zlib_all_five.bl73").write_bytes(zdata)
    start=time.perf_counter()
    zdec=decode_zlib_bundle((root/"paeth_zlib_all_five.bl73").read_bytes())
    zlib_dec_s=time.perf_counter()-start
    assert all(np.array_equal(images[k],zdec[k]) for k in SOURCES)
    meta=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
    encoded={}
    baseline_files={}
    atlas=np.concatenate([images[n] for n in SOURCES],axis=1)
    for codec in BASES:
        per_source=[]
        for name in SOURCES:
            coded=encode_baseline(images[name],codec)
            saved=HEAD.pack(BASE,VERSION,len(meta))+meta+coded
            path=root/f"{codec}_{name}.bl73"
            path.write_bytes(saved)
            rec=decode_baseline(coded,images[name].shape)
            if not np.array_equal(rec,images[name]):
                raise AssertionError("Conventional individual image not lossless")
            per_source.append(len(saved))
            baseline_files[path.name]={
                "bytes":len(saved),"sha256":sha(saved),
                "original_photo":name,
            }
        atcoded=encode_baseline(atlas,codec)
        atpath=root/f"{codec}_joint_atlas.bl73"
        atfile=HEAD.pack(BASE,VERSION,len(meta))+meta+atcoded
        atpath.write_bytes(atfile)
        rec=decode_baseline(atcoded,atlas.shape)
        if not np.array_equal(rec,atlas):
            raise AssertionError("Joint atlas not lossless")
        baseline_files[atpath.name]={"bytes":len(atfile),"sha256":sha(atfile),
                                     "original_photo":"five-photo joint atlas"}
        encoded[codec]={
            "individual_complete_files_total_bytes":sum(per_source),
            "joint_atlas_complete_bytes":len(atfile),
            "best_actual_complete_bytes":min(sum(per_source),len(atfile)),
        }
    best,selected=min(
        (item["best_actual_complete_bytes"],codec)
        for codec,item in encoded.items())
    return {
        "experiment":73,
        "protocol":"research/conditional/BHUH_EXP73_PROTOCOL.md",
        "software":{"python":platform.python_version(),
                    "numpy":np.__version__,"pillow":PIL_VERSION,
                    "scikit_image":skimage.__version__},
        "source_independence":"5 original photos, trained probabilities on first 3 only",
        "source_manifest":sources,
        "all_lossless_verified":True,
        "learned_rans":{
            "filename":learnt_file.name,
            "sha256":sha(model),"complete_bytes":len(model),
            "training_sources":list(TRAIN),
            "holdout_sources":list(HOLDOUT),
            "encode_seconds_including_training":enc_s,
            "decode_seconds":dec_s,
            **info,
        },
        "paeth_zlib_ablation":{
            "filename":"paeth_zlib_all_five.bl73",
            "sha256":sha(zdata),"complete_bytes":len(zdata),
            "per_image_residual_stream_bytes":zdetails,
            "encode_seconds":zlib_enc_s,"decode_seconds":zlib_dec_s,
        },
        "conventional":encoded,
        "baseline_files":baseline_files,
        "smallest_conventional_complete_bundle":selected,
        "smallest_conventional_complete_bytes":best,
        "learned_smaller_than_best_lossless_baseline":len(model)<best,
        "model_to_best_baseline_ratio":len(model)/best,
        "note":"The model is an offline-fitted frequency table and standard rANS, not a neural network; model bytes must never be ignored.",
    }

def self_test():
    rng=np.random.default_rng(73)
    counts=rng.integers(0,3000,(TABLE_COUNT,256),dtype=np.int64)+1
    freq=normalize_counts(counts)
    start,inv=prepare_freq(freq)
    for n in (1,2,17,111,5000):
        seq=[(int(rng.integers(0,TABLE_COUNT)),int(rng.integers(0,256)))
             for _ in range(n)]
        stream=rans_encode(seq,freq,start)
        state=int.from_bytes(stream[:4],"little");pos=4;actual=[]
        for c,_ in seq:
            slot=state&(TOTAL-1);sym=int(inv[c,slot])
            actual.append(sym)
            state=int(freq[c,sym])*(state>>P)+slot-int(start[c,sym])
            while state<L:
                state=(state<<8)|stream[pos];pos+=1
        assert [s for c,s in seq]==actual and state==L and pos==len(stream)
    imgs={s:rng.integers(0,256,(SIDE,SIDE,3),dtype=np.uint8) for s in SOURCES}
    src={s:{"crop_sha256":sha(imgs[s].tobytes())} for s in SOURCES}
    manifest=make_manifest(imgs,src)
    coded,detail=model_bundle(imgs,manifest)
    decoded,m=decode_bundle(coded)
    assert all(np.array_equal(decoded[k],imgs[k]) for k in SOURCES)
    try:
        decode_bundle(coded[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupt rANS bundle accepted")
    zb,_=zlib_bundle(imgs,manifest)
    zd=decode_zlib_bundle(zb)
    assert all(np.array_equal(zd[k],imgs[k]) for k in SOURCES)
    print("EXP73 SELF-TEST PASS: byte-exact rANS, all 6 learned context tables, zlib control and corruption rejection")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",default="/tmp/blkh_exp73")
    a=p.parse_args()
    if a.self_test:
        self_test();return
    if not features.check("webp") or not features.check("jpg_2000"):
        raise RuntimeError("Fair lossless WebP and JPEG2000 baselines mandatory")
    root=Path(a.out)
    result=benchmark(root)
    (root/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print("EXP73 RESULT: learned rANS all 5 sources",result["learned_rans"]["complete_bytes"],
          "Paeth+zlib",result["paeth_zlib_ablation"]["complete_bytes"],
          "best conventional",result["smallest_conventional_complete_bundle"],
          result["smallest_conventional_complete_bytes"],
          "learned_size_win",result["learned_smaller_than_best_lossless_baseline"],flush=True)
    print("EXP73 TRAINED ONLY ON",list(TRAIN),"HOLDOUT PROBABILITY MODEL",list(HOLDOUT),flush=True)

if __name__=="__main__":
    main()
