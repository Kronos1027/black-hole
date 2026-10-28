"""
BLKH Experiment 74 — exact 16px tile dedup vs true lossless animation.

See precommitted research/dedup/BHUH_EXP74_PROTOCOL.md.
REAL source photos, SIMULATED deterministic local edits, not natural videos.
20 correlated scenarios (5 sources * 4 edit intensities, 8 RGB frames).
Two custom exact dictionary variants vs WebP animation, APNG and packed
WebP/PNG 4-by-2 atlases, plus raw zlib control. All bytes and metadata counted.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import struct
import sys
import time
import zlib

import numpy as np
from PIL import Image, __version__ as PIL_VERSION, features

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"conditional"))
from experiment_73_cross_source_rans import SOURCES, get_photo, sha

SIDE=128
TILE=16
COUNT=8
DURATION_MS=100
EDIT_COUNTS=(0,2,8,32)
SEED=740074
HEADER=struct.Struct("<5sBI")
DICT_HEADER=struct.Struct("<HHI")
MAGIC=b"BL74F"
VERSION=1
MODES=("raw_zlib","webp_dictionary",
       "webp_animation","apng_animation",
       "webp_atlas","png_atlas","frame_zlib")
LIMIT=8_000_000


def frames_for(source_i:int,edits:int)->tuple[np.ndarray,dict]:
    if not (0<=source_i<len(SOURCES)) or edits not in EDIT_COUNTS:
        raise ValueError("Only preregistered source/edit conditions")
    name=SOURCES[source_i]
    donor_name=SOURCES[(source_i+1)%len(SOURCES)]
    original,original_info=get_photo(name)
    donor,donor_info=get_photo(donor_name)
    frame=original.copy()
    frames=[frame.copy()]
    for frame_idx in range(1,COUNT):
        rng=np.random.default_rng(SEED+source_i*100+edits*10+frame_idx)
        touched=rng.choice(64,size=edits,replace=False)
        sampled=rng.choice(64,size=edits,replace=True)
        for tile_idx,donor_idx in zip(touched,sampled):
            y,x=divmod(int(tile_idx),8)
            dy,dx=divmod(int(donor_idx),8)
            frame[y*TILE:(y+1)*TILE,x*TILE:(x+1)*TILE]=(
                donor[dy*TILE:(dy+1)*TILE,dx*TILE:(dx+1)*TILE])
        frames.append(frame.copy())
    arrays=np.stack(frames)
    hashes=[sha(image.tobytes()) for image in arrays]
    manifest={
        "source":name,"donor":donor_name,
        "source_photo_sha256":original_info["original_photo_sha256"],
        "source_crop_sha256":original_info["crop_sha256"],
        "donor_photo_sha256":donor_info["original_photo_sha256"],
        "donor_crop_sha256":donor_info["crop_sha256"],
        "frame_size_rgb":[SIDE,SIDE,3],
        "frame_count":COUNT,
        "frame_duration_ms":DURATION_MS,
        "tile_side":TILE,
        "edited_tiles_per_transition":edits,
        "construction":"real photographic tiles; deterministic synthetic replacement sequence",
        "seed_formula":"740074 + source_index*100 + edits*10 + frame_index",
        "frame_rgb_sha256":hashes,
    }
    return arrays,manifest


def pack(frames:np.ndarray)->np.ndarray:
    if frames.shape!=(COUNT,SIDE,SIDE,3) or frames.dtype!=np.uint8:
        raise ValueError("Invalid eight-frame sequence")
    return frames.reshape(2,4,SIDE,SIDE,3).transpose(
        0,2,1,3,4).reshape(2*SIDE,4*SIDE,3)


def unpack(atlas:np.ndarray)->np.ndarray:
    if atlas.shape!=(2*SIDE,4*SIDE,3) or atlas.dtype!=np.uint8:
        raise ValueError("Invalid decoded atlas")
    return atlas.reshape(2,SIDE,4,SIDE,3).transpose(
        0,2,1,3,4).reshape(COUNT,SIDE,SIDE,3)


def wrap(codec:str,manifest:dict,payload:bytes)->bytes:
    info={**manifest,"codec":codec}
    meta=json.dumps(info,sort_keys=True,separators=(",",":")).encode()
    if not 0<len(meta)<16384:
        raise ValueError("Invalid manifest length")
    return HEADER.pack(MAGIC,VERSION,len(meta))+meta+payload


def unwrap(blob:bytes)->tuple[str,dict,bytes]:
    if not (HEADER.size<=len(blob)<=LIMIT):
        raise ValueError("Invalid container length")
    magic,version,metalen=HEADER.unpack(blob[:HEADER.size])
    if magic!=MAGIC or version!=VERSION or not 0<metalen<16384:
        raise ValueError("Unsupported BL74 file")
    end=HEADER.size+metalen
    if end>len(blob):
        raise ValueError("Truncated metadata")
    manifest=json.loads(blob[HEADER.size:end])
    if (manifest.get("frame_size_rgb")!=[SIDE,SIDE,3] or
            manifest.get("frame_count")!=COUNT or
            manifest.get("frame_duration_ms")!=DURATION_MS or
            manifest.get("source") not in SOURCES or
            manifest.get("codec") not in MODES):
        raise ValueError("BL74 source/sequence metadata mismatch")
    return manifest["codec"],manifest,blob[end:]


def dictionary_for(frames:np.ndarray)->tuple[list[bytes],np.ndarray]:
    unique=[]
    ids=[]
    mapping={}
    for frame in frames:
        for row in range(0,SIDE,TILE):
            for col in range(0,SIDE,TILE):
                raw=frame[row:row+TILE,col:col+TILE].tobytes()
                key=hashlib.sha256(raw).digest()
                if key not in mapping:
                    mapping[key]=len(unique)
                    unique.append(raw)
                elif unique[mapping[key]]!=raw:
                    raise ValueError("SHA256 collision detected")
                ids.append(mapping[key])
    if not 1<=len(unique)<=COUNT*64:
        raise ValueError("Bad unique tile count")
    return unique,np.asarray(ids,dtype="<u2")


def dictionary_payload(frames:np.ndarray,codec:str)->tuple[bytes,dict]:
    unique,ids=dictionary_for(frames)
    n=len(unique)
    if codec=="raw_zlib":
        payload=zlib.compress(b"".join(unique),9)
        width=0
    elif codec=="webp_dictionary":
        cols=16
        rows=(n+cols-1)//cols
        canvas=np.zeros((rows*TILE,cols*TILE,3),dtype=np.uint8)
        for i,raw in enumerate(unique):
            y,x=divmod(i,cols)
            canvas[y*TILE:(y+1)*TILE,x*TILE:(x+1)*TILE]=np.frombuffer(
                raw,dtype=np.uint8).reshape(TILE,TILE,3)
        with io.BytesIO() as dest:
            Image.fromarray(canvas).save(dest,format="WEBP",lossless=True,method=6)
            payload=dest.getvalue()
        width=cols
    else:
        raise ValueError("Unsupported dictionary coding")
    prefix=DICT_HEADER.pack(n,width,len(payload))
    return prefix+ids.tobytes()+payload,{
        "unique_tiles":n,
        "total_tile_references":COUNT*64,
        "exact_reused_tile_references":COUNT*64-n,
        "dictionary_payload_bytes":len(payload),
        "tile_index_bytes":ids.nbytes,
    }


def reconstruct_dictionary(raw:bytes,codec:str)->np.ndarray:
    if len(raw)<DICT_HEADER.size+COUNT*64*2:
        raise ValueError("Incomplete dictionary file")
    n,cols,size=DICT_HEADER.unpack(raw[:DICT_HEADER.size])
    if not 1<=n<=COUNT*64:
        raise ValueError("Invalid tile count")
    start=DICT_HEADER.size
    ref_end=start+COUNT*64*2
    ids=np.frombuffer(raw[start:ref_end],dtype="<u2")
    if np.any(ids>=n) or ref_end+size!=len(raw):
        raise ValueError("Invalid tile references or trailing bytes")
    encoded=raw[ref_end:]
    if codec=="raw_zlib":
        if cols!=0:raise ValueError("Invalid raw layout")
        dec=zlib.decompressobj()
        decoded=dec.decompress(encoded,n*TILE*TILE*3+1)
        if (len(decoded)!=n*TILE*TILE*3 or not dec.eof
                or dec.unused_data or dec.unconsumed_tail):
            raise ValueError("Invalid dictionary zlib stream")
        tiles=np.frombuffer(decoded,dtype=np.uint8).reshape(n,TILE,TILE,3)
    elif codec=="webp_dictionary":
        if cols!=16:raise ValueError("Invalid WebP atlas geometry")
        with Image.open(io.BytesIO(encoded)) as im:
            im.load()
            canvas=np.asarray(im.convert("RGB"),dtype=np.uint8)
        rows=(n+cols-1)//cols
        if canvas.shape!=(rows*TILE,cols*TILE,3):
            raise ValueError("Incorrect WebP dictionary size")
        tiles=np.stack([
            canvas[(i//cols)*TILE:(i//cols+1)*TILE,
                   (i%cols)*TILE:(i%cols+1)*TILE]
            for i in range(n)
        ])
    else:
        raise ValueError("Invalid dictionary codec")
    out=np.empty((COUNT,SIDE,SIDE,3),dtype=np.uint8)
    for k,tile_id in enumerate(ids):
        frame_id=k//64
        y,x=divmod(k%64,8)
        out[frame_id,y*TILE:(y+1)*TILE,x*TILE:(x+1)*TILE]=tiles[tile_id]
    return out


def encode_baseline(frames:np.ndarray,codec:str)->bytes:
    if codec=="frame_zlib":
        return zlib.compress(frames.tobytes(),9)
    if codec in ("webp_atlas","png_atlas"):
        atlas=pack(frames)
        with io.BytesIO() as dest:
            if codec=="webp_atlas":
                Image.fromarray(atlas).save(dest,format="WEBP",lossless=True,method=6)
            else:
                Image.fromarray(atlas).save(dest,format="PNG",optimize=True,
                                            compress_level=9)
            return dest.getvalue()
    if codec in ("webp_animation","apng_animation"):
        images=[Image.fromarray(frame) for frame in frames]
        with io.BytesIO() as dest:
            if codec=="webp_animation":
                images[0].save(dest,format="WEBP",save_all=True,
                    append_images=images[1:],duration=DURATION_MS,
                    loop=0,lossless=True,method=6)
            else:
                images[0].save(dest,format="PNG",save_all=True,
                    append_images=images[1:],duration=DURATION_MS,
                    loop=0,disposal=0,blend=0,optimize=False)
            return dest.getvalue()
    raise ValueError("Unknown baseline")


def decode_baseline(payload:bytes,codec:str)->np.ndarray:
    if codec=="frame_zlib":
        dec=zlib.decompressobj()
        raw=dec.decompress(payload,COUNT*SIDE*SIDE*3+1)
        if (len(raw)!=COUNT*SIDE*SIDE*3 or not dec.eof
                or dec.unused_data or dec.unconsumed_tail):
            raise ValueError("Bad zlib frame stream")
        return np.frombuffer(raw,dtype=np.uint8).reshape(COUNT,SIDE,SIDE,3).copy()
    with Image.open(io.BytesIO(payload)) as im:
        if codec in ("webp_atlas","png_atlas"):
            im.load()
            return unpack(np.asarray(im.convert("RGB"),dtype=np.uint8))
        if codec not in ("webp_animation","apng_animation"):
            raise ValueError("Unsupported animation codec")
        expanded=[]
        for j in range(im.n_frames):
            im.seek(j)
            picture=np.asarray(im.convert("RGB"),dtype=np.uint8)
            duration=float(im.info.get("duration",DURATION_MS))
            count=round(duration/DURATION_MS)
            if (count<1 or abs(duration-count*DURATION_MS)>1.0
                    or len(expanded)+count>COUNT):
                raise ValueError(f"Animation frame timing invalid: {codec} frame={j} n_frames={im.n_frames} duration={duration} count={count} already={len(expanded)} info={dict(im.info)}")
            expanded.extend(picture.copy() for _ in range(count))
        if len(expanded)!=COUNT:
            raise ValueError(f"Expected {COUNT} timed RGB animation frames, got {len(expanded)}")
        return np.stack(expanded)


def encode_case(frames:np.ndarray,manifest:dict,codec:str)->tuple[bytes,dict]:
    if codec in ("raw_zlib","webp_dictionary"):
        raw,stats=dictionary_payload(frames,codec)
    else:
        raw,stats=encode_baseline(frames,codec),{}
    return wrap(codec,manifest,raw),stats


def decode_case(raw:bytes)->tuple[np.ndarray,dict]:
    codec,meta,stream=unwrap(raw)
    frames=(reconstruct_dictionary(stream,codec)
            if codec in ("raw_zlib","webp_dictionary")
            else decode_baseline(stream,codec))
    if frames.shape!=(COUNT,SIDE,SIDE,3) or frames.dtype!=np.uint8:
        raise ValueError("Incorrect RGB frame geometry")
    expected=meta.get("frame_rgb_sha256")
    if not isinstance(expected,list) or len(expected)!=COUNT:
        raise ValueError("Missing frame checksums")
    for i,frame in enumerate(frames):
        if sha(frame.tobytes())!=expected[i]:
            raise ValueError(f"Lossy or corrupted encoded RGB frame {i}")
    return frames,meta


def self_test():
    rng=np.random.default_rng(SEED)
    frame=rng.integers(0,256,(SIDE,SIDE,3),dtype=np.uint8)
    for edits in (0,3,32):
        sequence=np.stack([frame.copy() for _ in range(COUNT)])
        for i in range(1,COUNT):
            if edits:
                positions=rng.choice(64,size=edits,replace=False)
                for tid in positions:
                    y,x=divmod(int(tid),8)
                    sequence[i,y*TILE:(y+1)*TILE,x*TILE:(x+1)*TILE]=rng.integers(
                        0,256,(TILE,TILE,3),dtype=np.uint8)
        meta={"source":SOURCES[0],"frame_size_rgb":[SIDE,SIDE,3],
              "frame_count":COUNT,"frame_duration_ms":DURATION_MS,
              "frame_rgb_sha256":[sha(f.tobytes()) for f in sequence]}
        assert np.array_equal(unpack(pack(sequence)),sequence)
        for codec in MODES:
            b,stat=encode_case(sequence,meta,codec)
            actual,decoded_meta=decode_case(b)
            assert decoded_meta["source"]==SOURCES[0]
            assert np.array_equal(actual,sequence)
            if codec in ("raw_zlib","webp_dictionary"):
                assert stat["unique_tiles"]<=COUNT*64
                try:
                    decode_case(b[:-1])
                except (ValueError,zlib.error,OSError):
                    pass
                else:
                    raise AssertionError("Corrupted tile archive accepted")
    print("EXP74 SELFTEST PASS: all seven codecs fully exact in 3 simulated-edit regimes")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",default="/tmp/blkh_exp74")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if not features.check("webp"):
        raise RuntimeError("WebP codec support mandatory")
    # Pillow 13 removed the webp_anim feature flag; the lossless animated
    # WebP write/read self-test below is the reliable capability check.
    if a.self_test:
        self_test()
        return
    root=Path(a.out)
    root.mkdir(parents=True,exist_ok=True)
    results=[]
    for i,source in enumerate(SOURCES):
        for edit in EDIT_COUNTS:
            frames,meta=frames_for(i,edit)
            dir=root/source/f"edit_{edit:02d}"
            dir.mkdir(parents=True,exist_ok=True)
            codecs={}
            for codec in MODES:
                start=time.perf_counter_ns()
                blob,stat=encode_case(frames,meta,codec)
                duration_enc=(time.perf_counter_ns()-start)/1e9
                path=dir/(codec+".bl74")
                path.write_bytes(blob)
                t=time.perf_counter_ns()
                decoded,back=decode_case(path.read_bytes())
                duration_dec=(time.perf_counter_ns()-t)/1e9
                if not np.array_equal(decoded,frames) or back!={
                    **meta,"codec":codec
                }:
                    raise AssertionError("Serialized file failed frame-order equivalence")
                codecs[codec]={
                    "file":path.name,"bytes":len(blob),
                    "sha256":sha(blob),
                    "encode_seconds":duration_enc,
                    "decode_seconds":duration_dec,
                    **stat,
                }
            baseline_best,baseline_mode=min(
                (codecs[mode]["bytes"],mode)
                for mode in MODES if mode not in ("raw_zlib","webp_dictionary")
            )
            best_dedup,dedup_mode=min(
                (codecs[mode]["bytes"],mode) for mode in ("raw_zlib","webp_dictionary")
            )
            row={
                "source":source,"changed_tiles_per_transition":edit,
                "frame_sha256":meta["frame_rgb_sha256"],
                "manifest":meta,
                "codecs":codecs,
                "best_reference":baseline_mode,
                "best_reference_complete_bytes":baseline_best,
                "best_dedup":dedup_mode,
                "best_dedup_complete_bytes":best_dedup,
                "dedup_smaller_than_best_valid_standard":bool(best_dedup<baseline_best),
                "all_8_frames_exact":True,
            }
            results.append(row)
            print("EXP74",source,"edit",edit,
                  "unique",codecs["raw_zlib"]["unique_tiles"],
                  "best_dedup",dedup_mode,best_dedup,
                  "best_ref",baseline_mode,baseline_best,
                  "win",row["dedup_smaller_than_best_valid_standard"],flush=True)
            (root/"results.json").write_text(json.dumps({
                "experiment":74,
                "preregistration":"research/dedup/BHUH_EXP74_PROTOCOL.md",
                "sources":list(SOURCES),
                "edit_counts":list(EDIT_COUNTS),
                "frame_count":COUNT,
                "real_sources_with_simulated_edits":True,
                "runner_versions":{"python":platform.python_version(),
                                   "numpy":np.__version__,"pillow":PIL_VERSION},
                "status":"provisional until complete external file audit",
                "records":results,
            },indent=2),encoding="utf8")
    assert len(results)==len(SOURCES)*len(EDIT_COUNTS)
    print("EXP74 COMPLETE: 20 RGB FRAME SCENARIOS, ALL 7 LOSSLESS ARCHIVES PER CASE",flush=True)


if __name__=="__main__":
    main()
