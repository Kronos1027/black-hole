"""
BLKH Experiment 75 — genuine time-lapse acquisition transfer of Exp74 codec.

No artificial image-tile edits, shifts or interpolations. Real temporal
scientific sequences from skimage.data.protein_transport and
skimage.data.nickel_solidification, first 8 original time points.

Source images may be uint16. Entire 8-frame grayscale crop is normalized
once to uint8 RGB (same mono channel copied to RGB). This is LOSSY relative
to raw uint16 scientific values, although ALL CODECS MUST be exactly
lossless relative to the byte-defined uint8 RGB benchmark frames.
Precommitted method: research/dedup/BHUH_EXP75_PROTOCOL.md.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import skimage
from skimage import data
from PIL import __version__ as PIL_VERSION

from experiment_74_cross_frame_exact_blocks import (
    MODES,COUNT,SIDE,TILE,DURATION_MS,
    encode_case,decode_case,sha
)

SOURCES=("protein_transport","nickel_solidification")


def raw_time_sequence(name:str)->tuple[np.ndarray,dict]:
    if name not in SOURCES:
        raise ValueError("Source not preregistered")
    raw=np.asarray(getattr(data,name)())
    if name=="protein_transport":
        if raw.ndim!=4 or raw.shape[0]<COUNT or raw.shape[1]<1:
            raise ValueError("Not an actual four-axis protein transport video")
        movie=raw[:COUNT,0]
        chosen_channel=0
    else:
        if raw.ndim!=3 or raw.shape[0]<COUNT:
            raise ValueError("Not an actual three-axis nickel radiograph movie")
        movie=raw[:COUNT]
        chosen_channel=None
    if movie.ndim!=3 or movie.shape[1]<SIDE or movie.shape[2]<SIDE:
        raise ValueError("Temporal spatial geometry unsupported")
    if not np.issubdtype(movie.dtype,np.integer):
        raise ValueError("Only raw integer acquisition data supported")
    y=(movie.shape[1]-SIDE)//2
    x=(movie.shape[2]-SIDE)//2
    source_crop=np.ascontiguousarray(movie[:,y:y+SIDE,x:x+SIDE])
    if source_crop.shape!=(COUNT,SIDE,SIDE):
        raise ValueError("Source crop invalid")
    low=int(source_crop.min());high=int(source_crop.max())
    if high==low:
        raise ValueError("Degenerate source dataset with constant pixels")
    # Precommitted globally shared normalization across all eight frames,
    # never selected per frame or per codec to improve dictionary results.
    normalized=np.rint(
        (source_crop.astype(np.float64)-low)*255/(high-low)
    ).clip(0,255).astype(np.uint8)
    rgb=np.repeat(normalized[...,None],3,axis=3)
    if rgb.shape!=(COUNT,SIDE,SIDE,3):
        raise AssertionError("RGB conversion failed")
    info={
        "source":name,
        "donor":"none; actual captured time sequence",
        "source_type":"genuine temporal scientific acquisition",
        "source_movie_shape":list(raw.shape),
        "original_full_movie_sha256":sha(raw.tobytes()),
        "original_movie_dtype":str(raw.dtype),
        "temporal_frames":[0,1,2,3,4,5,6,7],
        "selected_channel":chosen_channel,
        "crop_yx":[y,x],
        "original_8_frame_crop_sha256":sha(source_crop.tobytes()),
        "original_crop_dtype":str(source_crop.dtype),
        "global_minimum":low,
        "global_maximum":high,
        "normalization":"uint8 round((value-global_min)*255/(global_max-global_min)), shared across all 8 frames; lossy for source uint16",
        "color_mapping":"one grayscale channel replicated to R,G,B",
        "frame_size_rgb":[SIDE,SIDE,3],
        "frame_count":COUNT,
        "frame_duration_ms":DURATION_MS,
        "display_timing_notice":"100ms playback time assigned for benchmark only, not original experiment temporal interval",
        "tile_side":TILE,
        "frame_rgb_sha256":[sha(frame.tobytes()) for frame in rgb],
        "image_sequence_byte_sha256":sha(rgb.tobytes()),
    }
    return rgb,info


def controlled_self_test()->None:
    rng=np.random.default_rng(75)
    image=np.empty((COUNT,SIDE,SIDE,3),dtype=np.uint8)
    for i in range(COUNT):
        base=rng.integers(0,256,(SIDE,SIDE),dtype=np.uint8)
        image[i]=np.repeat(base[...,None],3,axis=2)
    info={
        "source":"real_time_series_selftest",
        "frame_size_rgb":[SIDE,SIDE,3],
        "frame_count":COUNT,
        "frame_duration_ms":DURATION_MS,
        "frame_rgb_sha256":[sha(im.tobytes()) for im in image],
    }
    for codec in MODES:
        archive,_=encode_case(image,info,codec)
        reconstruction,meta=decode_case(archive)
        assert meta=={**info,"codec":codec}
        assert np.array_equal(reconstruction,image)
    print("EXP75 SELFTEST PASS: seven exact RGB codecs support independent temporal frame identifiers")


def run(name:str,root:Path)->dict:
    frames,meta=raw_time_sequence(name)
    case=root/name
    case.mkdir(parents=True,exist_ok=True)
    encoded={}
    for codec in MODES:
        start=time.perf_counter_ns()
        archive,stats=encode_case(frames,meta,codec)
        encode_s=(time.perf_counter_ns()-start)/1e9
        target=case/(codec+".bl74")
        target.write_bytes(archive)
        start=time.perf_counter_ns()
        result,manifest=decode_case(target.read_bytes())
        decode_s=(time.perf_counter_ns()-start)/1e9
        if manifest!={**meta,"codec":codec} or not np.array_equal(result,frames):
            raise AssertionError("Real temporal source codec was not exact")
        encoded[codec]={
            "file":target.name,
            "complete_bytes":len(archive),
            "sha256":sha(archive),
            "encode_seconds":encode_s,
            "decode_seconds":decode_s,
            **stats,
        }
    best,mode=min((encoded[m]["complete_bytes"],m)
                  for m in MODES if m not in ("raw_zlib","webp_dictionary"))
    our,our_mode=min((encoded[m]["complete_bytes"],m)
                  for m in ("raw_zlib","webp_dictionary"))
    return {
        "source":name,
        "manifest":meta,
        "rgb_sha256":sha(frames.tobytes()),
        "temporal_frames_real_not_synthetic":True,
        "codec_input_representation_is_8bit_RGB_not_16bit_native":True,
        "codecs":encoded,
        "best_standard":mode,"best_standard_complete_bytes":best,
        "best_dedup":our_mode,"best_dedup_complete_bytes":our,
        "custom_smaller_than_best_standard":bool(our<best),
        "all_8_derived_rgb_frames_exactly_reconstructed":True,
    }


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--out",default="/tmp/blkh_exp75")
    options=ap.parse_args()
    if options.self_test:
        controlled_self_test()
        return
    root=Path(options.out);root.mkdir(parents=True,exist_ok=True)
    rows=[];exclusions=[]
    for name in SOURCES:
        try:
            row=run(name,root)
        except Exception as err:
            exclusions.append({"source":name,"error":str(err)})
            print("EXP75 DECLARED SOURCE FAILURE",name,type(err).__name__,str(err),flush=True)
            continue
        rows.append(row)
        print("EXP75",name,
              "original_dtype",row["manifest"]["original_movie_dtype"],
              "unique_tiles",row["codecs"]["webp_dictionary"]["unique_tiles"],
              "exact_refs",row["codecs"]["webp_dictionary"]["exact_reused_tile_references"],
              "best_dedup",row["best_dedup"],row["best_dedup_complete_bytes"],
              "best_standard",row["best_standard"],row["best_standard_complete_bytes"],
              "win",row["custom_smaller_than_best_standard"],flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":75,
            "preregistration":"research/dedup/BHUH_EXP75_PROTOCOL.md",
            "sources":list(SOURCES),
            "source_count":len(rows),
            "exclusions":exclusions,
            "source_representation_warning":"source dtype may be uint16; all codecs are lossless only against globally normalized derived 8bit RGB, not native uint16 acquisition",
            "software":{"python":platform.python_version(),
                        "numpy":np.__version__,"skimage":skimage.__version__,
                        "pillow":PIL_VERSION},
            "results":rows,
        },indent=2),encoding="utf8")
    (root/"results.json").write_text(json.dumps({
        "experiment":75,
        "preregistration":"research/dedup/BHUH_EXP75_PROTOCOL.md",
        "sources":list(SOURCES),
        "source_count":len(rows),
        "exclusions":exclusions,
        "source_representation_warning":"source dtype may be uint16; all codecs are lossless only against globally normalized derived 8bit RGB, not native uint16 acquisition",
        "software":{"python":platform.python_version(),
                    "numpy":np.__version__,"skimage":skimage.__version__,
                    "pillow":PIL_VERSION},
        "results":rows,
    },indent=2),encoding="utf8")
    if exclusions or len(rows)!=len(SOURCES):
        raise RuntimeError("Unavailable declared real temporal source(s); no replacements; recorded in results.json")
    print("EXP75 DONE: both genuine temporal video sources, seven fully decoded RGB codec files each",flush=True)


if __name__=="__main__":
    main()
