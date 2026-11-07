"""
BLKH Experiment 76 — reversible temporal XOR and mod-256 differences.

Real image sequences, exactly the same predeclared uint8 RGB derived inputs
as Exp75 (original raw uint16 scientific files are NOT reconstructed).
Four temporal predictors and seven baseline codec variants, all source frames
and metadata exact. No motion vectors, filters, training, hidden state or
novel codec claim.

Protocol: research/dedup/BHUH_EXP76_PROTOCOL.md
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import platform
import struct
import time
import zlib

import numpy as np
from PIL import Image, __version__ as PIL_VERSION
from experiment_75_real_temporal_frames import SOURCES,raw_time_sequence
from experiment_74_cross_frame_exact_blocks import (
    MODES as REFERENCE_MODES,COUNT,SIDE,pack,unpack,
    encode_case as standard_encode,decode_case as standard_decode,sha,
)

NEW_MODES=("xor_webp","xor_zlib","mod_webp","mod_zlib")
MAGIC=b"BL76T"
HEAD=struct.Struct("<5sBI")
ENCODING_LIMIT=8_000_000


def residuals(frames:np.ndarray,mode:str)->np.ndarray:
    if frames.shape!=(COUNT,SIDE,SIDE,3) or frames.dtype!=np.uint8:
        raise ValueError("Invalid uint8 RGB temporal source")
    if mode not in NEW_MODES:
        raise ValueError("Invalid temporal predictor")
    previous=np.concatenate([np.zeros_like(frames[:1]),frames[:-1]],axis=0)
    if mode.startswith("xor_"):
        return np.bitwise_xor(frames,previous)
    return ((frames.astype(np.int16)-previous.astype(np.int16))&255).astype(np.uint8)


def invert(delta:np.ndarray,mode:str)->np.ndarray:
    if delta.shape!=(COUNT,SIDE,SIDE,3) or delta.dtype!=np.uint8:
        raise ValueError("Invalid compressed prediction residual")
    result=np.empty_like(delta)
    prev=np.zeros_like(delta[0])
    for i in range(COUNT):
        current=(np.bitwise_xor(delta[i],prev) if mode.startswith("xor_")
                 else ((delta[i].astype(np.uint16)+prev.astype(np.uint16))&255).astype(np.uint8))
        result[i]=current
        prev=current
    return result


def save_model(frames:np.ndarray,manifest:dict,mode:str)->bytes:
    if mode not in NEW_MODES:
        raise ValueError("Predictor outside protocol")
    delta=residuals(frames,mode)
    if mode.endswith("_webp"):
        with io.BytesIO() as f:
            Image.fromarray(pack(delta)).save(
                f,format="WEBP",lossless=True,method=6)
            packed=f.getvalue()
    else:
        packed=zlib.compress(delta.tobytes(),9)
    metabytes=json.dumps({**manifest,"codec":mode},sort_keys=True,
                         separators=(",",":")).encode()
    if not 0<len(metabytes)<16384:
        raise ValueError("Invalid temporal metadata length")
    return HEAD.pack(MAGIC,1,len(metabytes))+metabytes+packed


def open_model(blob:bytes)->tuple[np.ndarray,dict]:
    if not HEAD.size<=len(blob)<=ENCODING_LIMIT:
        raise ValueError("Truncated/oversized temporal file")
    magic,version,length=HEAD.unpack(blob[:HEAD.size])
    if magic!=MAGIC or version!=1 or not 0<length<16384:
        raise ValueError("Invalid time-series container")
    if HEAD.size+length>len(blob):
        raise ValueError("Truncated temporal manifest")
    meta=json.loads(blob[HEAD.size:HEAD.size+length])
    mode=meta.get("codec")
    if (mode not in NEW_MODES or
        meta.get("frame_size_rgb")!=[SIDE,SIDE,3] or
        meta.get("frame_count")!=COUNT or
        meta.get("source") not in SOURCES):
        raise ValueError("Incorrect source/predictor schema")
    payload=blob[HEAD.size+length:]
    if mode.endswith("_webp"):
        with Image.open(io.BytesIO(payload)) as f:
            f.load()
            image=np.asarray(f.convert("RGB"),dtype=np.uint8)
        delta=unpack(image)
    else:
        dec=zlib.decompressobj()
        raw=dec.decompress(payload,COUNT*SIDE*SIDE*3+1)
        if (len(raw)!=COUNT*SIDE*SIDE*3 or not dec.eof or
                dec.unused_data or dec.unconsumed_tail):
            raise ValueError("Bad/overflowing temporal residual stream")
        delta=np.frombuffer(raw,dtype=np.uint8).reshape(COUNT,SIDE,SIDE,3)
    restored=invert(delta,mode)
    hashes=meta.get("frame_rgb_sha256")
    if not isinstance(hashes,list) or len(hashes)!=COUNT:
        raise ValueError("Incomplete temporal source integrity manifest")
    for i,frame in enumerate(restored):
        if sha(frame.tobytes())!=hashes[i]:
            raise ValueError("Lossy or corrupt temporal frame")
    return restored,meta


def self_test():
    rng=np.random.default_rng(76)
    sample=rng.integers(0,256,(COUNT,SIDE,SIDE,3),dtype=np.uint8)
    manifest={
        "source":SOURCES[0],
        "frame_count":COUNT,
        "frame_size_rgb":[SIDE,SIDE,3],
        "frame_rgb_sha256":[sha(frame.tobytes()) for frame in sample],
    }
    for mode in NEW_MODES:
        res=residuals(sample,mode)
        assert np.array_equal(invert(res,mode),sample)
        binary=save_model(sample,manifest,mode)
        restored,meta=open_model(binary)
        assert meta=={**manifest,"codec":mode}
        assert np.array_equal(restored,sample)
        try:
            open_model(binary[:-1])
        except (ValueError,zlib.error,OSError):
            pass
        else:
            raise AssertionError("Truncated temporal model accepted")
    print("EXP76 SELFTEST PASS: 4 byte-exact true temporal XOR/modular codecs")


def run_one(source:str,root:Path)->dict:
    frames,manifest=raw_time_sequence(source)
    folder=root/source
    folder.mkdir(parents=True,exist_ok=True)
    info={}
    for mode in NEW_MODES+REFERENCE_MODES:
        began=time.perf_counter_ns()
        if mode in NEW_MODES:
            raw=save_model(frames,manifest,mode)
            stats={}
        else:
            raw,stats=standard_encode(frames,manifest,mode)
        encoding_time=(time.perf_counter_ns()-began)/1e9
        ext=".bl76" if mode in NEW_MODES else ".bl74"
        path=folder/(mode+ext)
        path.write_bytes(raw)
        began=time.perf_counter_ns()
        decoded,meta=(open_model(path.read_bytes())
                      if mode in NEW_MODES else standard_decode(path.read_bytes()))
        decoding_time=(time.perf_counter_ns()-began)/1e9
        if not np.array_equal(decoded,frames) or meta!={**manifest,"codec":mode}:
            raise AssertionError("Saved temporal model did not restore source")
        info[mode]={
            "file":path.name,"complete_bytes":len(raw),
            "sha256":sha(raw),"encode_seconds":encoding_time,
            "decode_seconds":decoding_time,**stats,
        }
    reference_size,reference_mode=min(
        (info[m]["complete_bytes"],m)
        for m in REFERENCE_MODES if m not in ("raw_zlib","webp_dictionary")
    )
    candidate_size,candidate_mode=min(
        (info[m]["complete_bytes"],m) for m in NEW_MODES
    )
    return {
        "source":source,
        "original_scientific_source_dtype":manifest["original_movie_dtype"],
        "global_conversion_warning":"Lossless only on derived uint8 RGB frames, not original uint16 scientific data",
        "original_source_manifest":manifest,
        "original_derived_frame_hash":sha(frames.tobytes()),
        "codecs":info,
        "best_temporal_mode":candidate_mode,
        "best_temporal_complete_bytes":candidate_size,
        "best_reference_mode":reference_mode,
        "best_reference_complete_bytes":reference_size,
        "temporal_smaller_than_best_reference":bool(candidate_size<reference_size),
        "all_saved_frames_exactly_restored":True,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="/tmp/blkh_exp76")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    records=[]
    for source in SOURCES:
        row=run_one(source,root)
        records.append(row)
        print("EXP76",source,
              "best_temporal",row["best_temporal_mode"],row["best_temporal_complete_bytes"],
              "best_reference",row["best_reference_mode"],row["best_reference_complete_bytes"],
              "wins",row["temporal_smaller_than_best_reference"],flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":76,
            "preregistration":"research/dedup/BHUH_EXP76_PROTOCOL.md",
            "source_movies":list(SOURCES),
            "new_coders":list(NEW_MODES),
            "standard_coders":list(REFERENCE_MODES),
            "note":"only derived RGB8 is preserved exactly; original nickel is uint16",
            "software":{"python":platform.python_version(),"numpy":np.__version__,
                        "pillow":PIL_VERSION},
            "results":records,
        },indent=2),encoding="utf8")
    assert len(records)==len(SOURCES)
    print("EXP76 COMPLETE: real temporal microscopic movies, four lossless temporal predictors plus seven fair baselines",flush=True)


if __name__=="__main__":
    main()
