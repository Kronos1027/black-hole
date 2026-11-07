"""
Experiment 77 — native source bit depth, lossless temporal prediction.

Two previously studied *real* microscopy sequences; no conversion to RGB8
or normalization of original nickel uint16. Compare 4 temporal residual
methods to actual FFV1 Matroska, PNG/JPEG2000/TIFF-ZSTD chronological atlases
and direct ZSTD. All 9 complete formats independently decoded from saved
file bytes, with 100% source integer equality.

Protocol: research/dedup/BHUH_EXP77_NATIVE_16BIT_PROTOCOL.md.
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

import brotli
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import skimage
from skimage import data
from PIL import Image, features, __version__ as PIL_VERSION
import zstandard as zstd

SOURCES=("protein_transport","nickel_solidification")
CODECS=("mod_zstd","xor_zstd","mod_brotli","xor_brotli",
        "atlas_png","atlas_jp2","atlas_tiff_zstd","ffv1_mkv","raw_zstd")
TEMPORAL=CODECS[:4]
BASELINES=CODECS[4:]
MAGIC=b"BL77N"
HEADER=struct.Struct("<5sBI")
TIMES=8
SIDE=128
FRAME_MS=100
MAX_META=16_384
MAX_OUTPUT=5_000_000
LOSSLESS_ZSTD_LEVEL=9


def digest(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def pixel_type(bits:int):
    if bits==8:return np.dtype("u1")
    if bits==16:return np.dtype("<u2")
    raise ValueError("Unsupported scientific pixel bit depth")


def raw_sequence(name:str)->tuple[np.ndarray,dict]:
    if name not in SOURCES:
        raise ValueError("Source outside protocol")
    source=np.asarray(getattr(data,name)())
    if name=="protein_transport":
        if source.ndim!=4 or source.shape[0]<TIMES or source.shape[1]<1:
            raise ValueError("Expected real protein transport movie")
        samples=source[:TIMES,0]
        channel=0
        expect_bits=8
    else:
        if source.ndim!=3 or source.shape[0]<TIMES:
            raise ValueError("Expected real nickel radiograph movie")
        samples=source[:TIMES]
        channel=None
        expect_bits=16
    if samples.ndim!=3 or min(samples.shape[1:])<SIDE:
        raise ValueError("Invalid original temporal acquisition geometry")
    bits=int(source.dtype.itemsize*8)
    if not np.issubdtype(source.dtype,np.unsignedinteger) or bits!=expect_bits:
        raise ValueError("Original scientific sensor bit depth mismatch")
    y,x=((samples.shape[1]-SIDE)//2,(samples.shape[2]-SIDE)//2)
    original_crop=np.ascontiguousarray(samples[:,y:y+SIDE,x:x+SIDE])
    # Encoding is unconditionally little-endian; this is a reversible byte
    # order conversion and NEVER changes native sample values or bit width.
    frames=np.asarray(original_crop,dtype=pixel_type(bits))
    if frames.shape!=(TIMES,SIDE,SIDE):
        raise ValueError("Invalid selected source crop")
    manifest={
        "source":name,
        "full_source_shape":list(source.shape),
        "full_original_source_sha256":digest(source.tobytes()),
        "full_source_dtype":str(source.dtype),
        "selected_original_channel":channel,
        "selected_timepoint_indices":list(range(TIMES)),
        "original_crop_offset_yx":[y,x],
        "original_crop_bytes_sha256":digest(original_crop.tobytes()),
        "native_sample_bits":bits,
        "native_dtype_little_endian":"uint8" if bits==8 else "uint16le",
        "frame_shape":[TIMES,SIDE,SIDE],
        "source_frame_sha256":[digest(frame.tobytes()) for frame in frames],
        "complete_canonical_frames_sha256":digest(frames.tobytes()),
        "source_native_min":int(frames.min()),
        "source_native_max":int(frames.max()),
        "frame_duration_ms":FRAME_MS,
        "frame_duration_notice":"100ms a benchmark display convention, not original scientific acquisition timing",
        "raw_native_scientific_pixels_preserved":True,
    }
    return frames,manifest


def frame_atlas(frames:np.ndarray)->np.ndarray:
    if frames.shape!=(TIMES,SIDE,SIDE):
        raise ValueError("Invalid frame geometry")
    return frames.reshape(2,4,SIDE,SIDE).transpose(0,2,1,3).reshape(SIDE*2,SIDE*4)


def from_atlas(atlas:np.ndarray,bits:int)->np.ndarray:
    if atlas.shape!=(SIDE*2,SIDE*4):
        raise ValueError("Wrong grayscale atlas dimensions")
    return atlas.reshape(2,SIDE,4,SIDE).transpose(0,2,1,3).reshape(
        TIMES,SIDE,SIDE).astype(pixel_type(bits))


def predict(frames:np.ndarray,mode:str)->np.ndarray:
    if mode not in TEMPORAL:
        raise ValueError("Not a fixed temporal candidate")
    bits=frames.dtype.itemsize*8
    prev=np.concatenate((np.zeros_like(frames[:1]),frames[:-1]),axis=0)
    if mode.startswith("xor_"):
        return np.bitwise_xor(frames,prev)
    mask=(1<<bits)-1
    return ((frames.astype(np.uint32)-prev.astype(np.uint32))&mask).astype(pixel_type(bits))


def restore(deltas:np.ndarray,mode:str)->np.ndarray:
    if mode not in TEMPORAL or deltas.shape!=(TIMES,SIDE,SIDE):
        raise ValueError("Invalid temporal reconstruction")
    bits=deltas.dtype.itemsize*8
    mask=(1<<bits)-1
    frames=np.empty_like(deltas)
    previous=np.zeros_like(deltas[0])
    for t in range(TIMES):
        if mode.startswith("xor_"):
            frames[t]=np.bitwise_xor(deltas[t],previous)
        else:
            frames[t]=((deltas[t].astype(np.uint32)+
                        previous.astype(np.uint32))&mask).astype(pixel_type(bits))
        previous=frames[t]
    return frames


def ffmpeg(commands:list[str],input_bytes:bytes)->bytes:
    proc=subprocess.run(commands,input=input_bytes,stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,timeout=90,check=False)
    if proc.returncode or not proc.stdout:
        raise RuntimeError("ffmpeg native-depth codec failure: "+
                           proc.stderr.decode(errors="replace")[:1800])
    return proc.stdout


def ffv1_encode(frames:np.ndarray)->bytes:
    fmt="gray" if frames.dtype.itemsize==1 else "gray16le"
    cmd=[
        "ffmpeg","-hide_banner","-loglevel","error","-nostdin",
        "-f","rawvideo","-pixel_format",fmt,
        "-video_size",f"{SIDE}x{SIDE}","-framerate","10",
        "-i","pipe:0","-frames:v",str(TIMES),
        "-c:v","ffv1","-level","3","-pix_fmt",fmt,
        "-threads","1","-f","matroska","pipe:1"
    ]
    return ffmpeg(cmd,frames.tobytes())


def ffv1_decode(blob:bytes,bits:int)->np.ndarray:
    fmt="gray" if bits==8 else "gray16le"
    command=[
        "ffmpeg","-hide_banner","-loglevel","error","-nostdin",
        "-i","pipe:0","-map","0:v:0","-frames:v",str(TIMES),
        "-f","rawvideo","-pix_fmt",fmt,
        "-threads","1","pipe:1"
    ]
    raw=ffmpeg(command,blob)
    expected=TIMES*SIDE*SIDE*(bits//8)
    if len(raw)!=expected:
        raise ValueError("FFV1 decoded wrong native source frame count/bit depth")
    return np.frombuffer(raw,dtype=pixel_type(bits)).copy().reshape(TIMES,SIDE,SIDE)


def compress(frames:np.ndarray,codec:str)->bytes:
    if codec not in CODECS:
        raise ValueError("Mode outside frozen method list")
    if codec in TEMPORAL or codec=="raw_zstd":
        values=predict(frames,codec) if codec in TEMPORAL else frames
        raw=values.tobytes()
        if codec.endswith("zstd"):
            return zstd.ZstdCompressor(level=LOSSLESS_ZSTD_LEVEL).compress(raw)
        return brotli.compress(raw,quality=9)
    if codec in ("atlas_png","atlas_jp2"):
        atlas=frame_atlas(frames)
        if frames.dtype.itemsize==2:
            atlas=atlas.astype("<u2")
        with io.BytesIO() as b:
            im=Image.fromarray(atlas)
            if codec=="atlas_png":
                im.save(b,format="PNG",optimize=True,compress_level=9)
            else:
                im.save(b,format="JPEG2000",irreversible=False)
            return b.getvalue()
    if codec=="atlas_tiff_zstd":
        atlas=frame_atlas(frames)
        with MemoryFile() as m:
            with m.open(driver="GTiff",height=SIDE*2,width=SIDE*4,count=1,
                        dtype="uint8" if frames.dtype.itemsize==1 else "uint16",
                        compress="ZSTD",predictor=2,tiled=True,
                        blockxsize=32,blockysize=32) as ds:
                ds.write(atlas,1)
            return m.read()
    if codec=="ffv1_mkv":
        return ffv1_encode(frames)
    raise ValueError("Unrecognized codec mode")


def decompress(payload:bytes,codec:str,bits:int)->np.ndarray:
    if codec not in CODECS or bits not in (8,16):
        raise ValueError("Unsupported codec/depth")
    expected=TIMES*SIDE*SIDE*(bits//8)
    if codec in TEMPORAL or codec=="raw_zstd":
        if codec.endswith("zstd"):
            values=zstd.ZstdDecompressor().decompress(payload,max_output_size=expected)
        else:
            decoder=brotli.Decompressor()
            body=bytearray()
            for start in range(0,len(payload),4096):
                body.extend(decoder.process(payload[start:start+4096]))
                if len(body)>expected:raise ValueError("Overlong temporal Brotli stream")
            if not decoder.is_finished():
                raise ValueError("Incomplete temporal Brotli stream")
            values=bytes(body)
        if len(values)!=expected:
            raise ValueError("Decoded native frame byte size mismatch")
        typed=np.frombuffer(values,dtype=pixel_type(bits)).copy().reshape(
            TIMES,SIDE,SIDE)
        return restore(typed,codec) if codec in TEMPORAL else typed
    if codec in ("atlas_png","atlas_jp2"):
        with Image.open(io.BytesIO(payload)) as im:
            im.load()
            arr=np.asarray(im)
        if not np.issubdtype(arr.dtype,np.integer):
            raise ValueError("Raster image not integer pixels")
        if (bits==16 and not np.array_equal(arr,arr.astype(np.uint16))) or (
            bits==8 and (np.any(arr>255) or np.any(arr<0))):
            raise ValueError("Pixel precision corrupted by standard codec")
        return from_atlas(arr.astype(pixel_type(bits)),bits)
    if codec=="atlas_tiff_zstd":
        with MemoryFile(payload) as m:
            with m.open() as ds:
                if ds.count!=1 or ds.dtypes[0] != ("uint8" if bits==8 else "uint16"):
                    raise ValueError("TIFF lost native sample bit depth")
                arr=ds.read(1)
        return from_atlas(arr,bits)
    if codec=="ffv1_mkv":
        return ffv1_decode(payload,bits)
    raise ValueError("Unexpected decoding method")


def archive(frames:np.ndarray,manifest:dict,codec:str)->bytes:
    meta=json.dumps({**manifest,"codec":codec},sort_keys=True,
                    separators=(",",":")).encode()
    if not 0<len(meta)<MAX_META:
        raise ValueError("Invalid source manifest")
    return HEADER.pack(MAGIC,1,len(meta))+meta+compress(frames,codec)


def decode_archive(blob:bytes)->tuple[np.ndarray,dict]:
    if not HEADER.size<len(blob)<=MAX_OUTPUT:
        raise ValueError("Truncated/oversized full archive")
    magic,version,length=HEADER.unpack(blob[:HEADER.size])
    if magic!=MAGIC or version!=1 or not 0<length<MAX_META:
        raise ValueError("Bad BL77 container schema")
    end=HEADER.size+length
    if end>len(blob):
        raise ValueError("Truncated source manifest")
    manifest=json.loads(blob[HEADER.size:end])
    codec=manifest.get("codec")
    bits=manifest.get("native_sample_bits")
    if (codec not in CODECS or bits not in (8,16) or
            manifest.get("frame_shape")!=[TIMES,SIDE,SIDE] or
            manifest.get("source") not in SOURCES or
            manifest.get("raw_native_scientific_pixels_preserved") is not True):
        raise ValueError("Invalid native video metadata")
    frames=decompress(blob[end:],codec,bits)
    if (frames.dtype.itemsize*8)!=bits:
        raise ValueError("Native scientific bit depth lost")
    if digest(frames.tobytes())!=manifest.get("complete_canonical_frames_sha256"):
        raise ValueError("Source original complete frame sequence checksum mismatch")
    hashes=manifest.get("source_frame_sha256")
    if not isinstance(hashes,list) or len(hashes)!=TIMES:
        raise ValueError("Missing native per-frame hashes")
    for i in range(TIMES):
        if digest(frames[i].tobytes())!=hashes[i]:
            raise ValueError("Original native scientific frame mismatch")
    return frames,manifest


def self_test():
    rng=np.random.default_rng(77)
    for bits in (8,16):
        src=rng.integers(0,1<<bits,size=(TIMES,SIDE,SIDE),
                         dtype=pixel_type(bits))
        meta={
            "source":SOURCES[0],"native_sample_bits":bits,
            "frame_shape":[TIMES,SIDE,SIDE],
            "raw_native_scientific_pixels_preserved":True,
            "complete_canonical_frames_sha256":digest(src.tobytes()),
            "source_frame_sha256":[digest(im.tobytes()) for im in src],
        }
        assert np.array_equal(from_atlas(frame_atlas(src),bits),src)
        for codec in CODECS:
            coded=archive(src,meta,codec)
            rec,m=decode_archive(coded)
            assert m["codec"]==codec and np.array_equal(rec,src)
            if codec in TEMPORAL:
                assert np.array_equal(restore(predict(src,codec),codec),src)
                try:
                    decode_archive(coded[:-1])
                except (ValueError,zstd.ZstdError,brotli.error,RuntimeError,OSError):
                    pass
                else:
                    raise AssertionError("Truncated temporal stream falsely accepted")
    print("EXP77 SELFTEST PASS: 18 random uint8 and uint16 native-depth exact codecs (including FFV1, JPEG2000 and TIFF ZSTD)")


def run_source(name:str,root:Path)->dict:
    frames,manifest=raw_sequence(name)
    folder=root/name
    folder.mkdir(parents=True,exist_ok=True)
    cases={}
    for codec in CODECS:
        start=time.perf_counter_ns()
        coded=archive(frames,manifest,codec)
        elapsed_encode=(time.perf_counter_ns()-start)/1e9
        path=folder/(codec+".bl77")
        path.write_bytes(coded)
        start=time.perf_counter_ns()
        actual,decoded_manifest=decode_archive(path.read_bytes())
        elapsed_decode=(time.perf_counter_ns()-start)/1e9
        if not np.array_equal(actual,frames) or decoded_manifest!={**manifest,"codec":codec}:
            raise AssertionError("Saved archive lost original scientific source")
        cases[codec]={
            "filename":path.name,"complete_bytes":len(coded),
            "sha256":digest(coded),"encode_seconds":elapsed_encode,
            "decode_seconds":elapsed_decode,
        }
        print("EXP77",name,codec,len(coded),"bytes",flush=True)
    choice,best=min((cases[c]["complete_bytes"],c) for c in TEMPORAL)
    reference,ref_codec=min((cases[c]["complete_bytes"],c) for c in BASELINES)
    return {
        "source":name,
        "scientific_metadata":manifest,
        "input_dtype_bits":frames.dtype.itemsize*8,
        "original_canonical_sha256":digest(frames.tobytes()),
        "codec_files":cases,
        "best_temporal":best,"best_temporal_bytes":choice,
        "best_native_lossless_baseline":ref_codec,
        "best_native_lossless_bytes":reference,
        "temporal_smaller_than_all_references":bool(choice<reference),
        "exact_native_bit_depth_reconstruction":True,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",default="/tmp/blkh_exp77")
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    if not features.check("jpg_2000"):
        raise RuntimeError("Lossless JPEG2000 required; cannot omit strong baseline")
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for source in SOURCES:
        entry=run_source(source,root)
        rows.append(entry)
        (root/"results.json").write_text(json.dumps({
            "experiment":77,
            "protocol":"research/dedup/BHUH_EXP77_NATIVE_16BIT_PROTOCOL.md",
            "sources":list(SOURCES),
            "methods":list(CODECS),
            "software":{"python":platform.python_version(),
                        "numpy":np.__version__,
                        "scikit_image":skimage.__version__,
                        "pillow":PIL_VERSION,"rasterio":rasterio.__version__,
                        "zstandard":zstd.__version__},
            "warning":"Original native scientific pixel values (uint8/uint16) are preserved exactly; sources already explored in Exp75/76",
            "results":rows,
        },indent=2),encoding="utf-8")
        print("EXP77 SUMMARY",source,"temporal",entry["best_temporal"],
              entry["best_temporal_bytes"],"baseline",entry["best_native_lossless_baseline"],
              entry["best_native_lossless_bytes"],"wins",
              entry["temporal_smaller_than_all_references"],flush=True)
    assert len(rows)==2
    print("EXP77 COMPLETE: TWO ORIGINAL REAL SCIENTIFIC TEMPORAL SEQUENCES LOSSLESS IN NATIVE BIT DEPTH",flush=True)


if __name__=="__main__":
    main()
