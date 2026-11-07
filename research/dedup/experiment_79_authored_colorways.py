"""
Exp79: true CC0 Kenney author-produced color variations for identical UI shapes.

Reuse ALL seven exact unchanged Exp78 compressors. Only dataset grouping
is different: first 8,16,24 lexicographically sorted exact shape names,
with original Blue,Green,Red,Yellow variants in shape-major ordering.
Not real historical revisions, not independent source families.
See BHUH_EXP79_REAL_COLORWAYS_PROTOCOL.md for fixed data selection.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import platform
import zipfile
import time

import numpy as np
from PIL import Image, __version__ as PIL_VERSION

from experiment_78_real_cc0_rgba_sprites import (
    SOURCE_URL,SOURCE_LANDING,LICENSE,TILE,NESTED,CODECS,
    fetch_zip,sha,pack_assets,base_manifest,encode_case,decode_case
)

COLORS=("Blue","Green","Red","Yellow")
SHAPES=(8,16,24)
EXPECTED_SHA256="a8a14a234911eb648c062622915c93e79e94e97cb7f9f375a70f6617f1174318"


def authored_colorways(blob:bytes,source_hash:str,root:Path)->tuple[list[dict],dict]:
    if source_hash!=EXPECTED_SHA256:
        raise ValueError("Fixed source ZIP changed; no substitution or results")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        keys=set(z.namelist())
        per_color={}
        for color in COLORS:
            prefix=f"PNG/{color}/Default/"
            per_color[color]={name[len(prefix):] for name in keys
                              if name.startswith(prefix) and name.lower().endswith(".png")
                              and "/" not in name[len(prefix):]}
        intersection=sorted(set.intersection(*(per_color[c] for c in COLORS)))
        eligible=[]
        rejected=[]
        for suffix in intersection:
            choices=[]
            for color in COLORS:
                path=f"PNG/{color}/Default/{suffix}"
                raw=z.read(path)
                with Image.open(io.BytesIO(raw)) as f:
                    f.load()
                    w,h=f.size
                    if not (16<=w<=512 and 16<=h<=512):
                        rejected.append({"shape":suffix,"reason":"geometry outside frozen bounds"})
                        choices=[]
                        break
                    rgba=np.asarray(f.convert("RGBA"),dtype=np.uint8)
                if not np.any(rgba[:,:,3]<255):
                    rejected.append({"shape":suffix,"reason":"opaque"})
                    choices=[]
                    break
                choices.append({
                    "meta":{
                        "filename":path,"colorway":color,"shape_suffix":suffix,
                        "source_png_sha256":sha(raw),
                        "pixel_rgba_sha256":sha(rgba.tobytes()),
                        "width":w,"height":h,
                        "alpha_transparent_pixel_count":int(np.count_nonzero(rgba[:,:,3]==0)),
                        "alpha_partial_pixel_count":int(np.count_nonzero(
                            (rgba[:,:,3]>0)&(rgba[:,:,3]<255))),
                    },
                    "rgba":rgba,"raw":raw,
                })
            if len(choices)==len(COLORS):
                eligible.append((suffix,choices))
        manifest={
            "source_zip_sha256":source_hash,
            "source_zip_bytes":len(blob),
            "colors_fixed":list(COLORS),
            "matching_filename_suffixes_all_four_colors":len(intersection),
            "eligible_shape_suffixes":len(eligible),
            "excluded_shapes":rejected,
            "selected_first_24_suffixes":[a for a,_ in eligible[:24]],
        }
        (root/"source_colorway_inventory.json").write_text(json.dumps(manifest,indent=2))
        if len(eligible)<max(SHAPES):
            raise ValueError("Fewer than preregistered 24 eligible matched real shapes")
        return [item for _,items in eligible[:max(SHAPES)] for item in items],manifest


def run(root:Path,cache:Path)->dict:
    root.mkdir(parents=True,exist_ok=True)
    zipped,h=fetch_zip(cache)
    source,inventory=authored_colorways(zipped,h,root)
    rows=[]
    for shapes in SHAPES:
        original=source[:shapes*len(COLORS)]
        n=len(original)
        assert n in NESTED
        atlas,layout=pack_assets(original)
        manifest=base_manifest(h,len(zipped),original,atlas,layout)
        manifest["variant_semantics"]="4 genuine Kenney-authored palette/colorways of each selected UI shape, NOT chronological revision files"
        manifest["colors_fixed_order"]=list(COLORS)
        manifest["shape_suffixes"]=inventory["selected_first_24_suffixes"][:shapes]
        folder=root/f"colorways_{n:03d}"
        folder.mkdir(parents=True,exist_ok=True)
        with io.BytesIO() as b:
            Image.fromarray(atlas,mode="RGBA").save(b,format="PNG")
            original_atlas=b.getvalue()
        (folder/"source_atlas_rgba.png").write_bytes(original_atlas)
        modes={}
        for codec in CODECS:
            start=time.perf_counter_ns()
            stream,stats=encode_case(atlas,original,manifest,codec)
            encode_time=(time.perf_counter_ns()-start)/1e9
            file=folder/(codec+".bl78")
            file.write_bytes(stream)
            start=time.perf_counter_ns()
            decoded,meta=decode_case(file.read_bytes())
            decode_time=(time.perf_counter_ns()-start)/1e9
            if len(decoded)!=n or meta!={**manifest,"codec":codec}:
                raise AssertionError("Saved file or manifest is not exact")
            if not all(np.array_equal(left["rgba"],right)
                       for left,right in zip(original,decoded)):
                raise AssertionError("RGBA original pixel mismatch")
            modes[codec]={
                "complete_bytes":len(stream),
                "file_sha256":sha(stream),
                "file":file.name,
                "encode_seconds":encode_time,
                "decode_seconds":decode_time,
                **stats,
            }
        dsize,dcodec=min((modes[k]["complete_bytes"],k)
                         for k in CODECS if k.startswith("tile"))
        ssize,scodec=min((modes[k]["complete_bytes"],k)
                         for k in CODECS if not k.startswith("tile"))
        row={
            "real_shape_count":shapes,
            "source_asset_count":n,
            "colorways":list(COLORS),
            "source_zip_sha256":h,
            "manifest":manifest,
            "original_atlas_file":"source_atlas_rgba.png",
            "original_atlas_file_sha256":sha(original_atlas),
            "methods":modes,
            "exact_rgba_all_7_decoders":True,
            "best_indexed_method":dcodec,"best_indexed_bytes":dsize,
            "best_lossless_reference":scodec,"best_lossless_reference_bytes":ssize,
            "indexed_smaller_than_best_reference":dsize<ssize,
        }
        rows.append(row)
        print("EXP79",shapes,"shapes",n,"real colorway images",
              "best_indexed",dcodec,dsize,
              "best_standard",scodec,ssize,
              "unique_tile_count",modes["tile16_webp"]["unique_tiles"],
              "indexed_wins",row["indexed_smaller_than_best_reference"],flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":79,
            "protocol":"research/dedup/BHUH_EXP79_REAL_COLORWAYS_PROTOCOL.md",
            "source":"Kenney UI Pack CC0 mirror at OpenGameArt; ZIP SHA256 frozen",
            "source_zip_sha256":h,
            "selected_shape_suffixes":inventory["selected_first_24_suffixes"],
            "colorways":list(COLORS),
            "shapes":list(SHAPES),
            "limited_scope":"one known art pack, correlated real authored color variants, not genuine historical file revisions or external independent artists",
            "software":{"numpy":np.__version__,"pillow":PIL_VERSION,
                        "python":platform.python_version()},
            "records":rows,
        },indent=2))
    assert len(rows)==3
    return {"source_hash":h,"records":rows}


def self_test():
    if tuple(shapes*len(COLORS) for shapes in SHAPES)!=NESTED:
        raise AssertionError("Colorway groups violate frozen expected subset sizes")
    assert len(set(COLORS))==4
    print("EXP79 SELFTEST PASS: frozen four-color grouping maps to 32,64,96 same-license actual PNG assets")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="/tmp/blkh_exp79")
    p.add_argument("--cache",default="/tmp/blkh_exp78_cache")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:self_test();return
    run(Path(a.out),Path(a.cache))
    print("EXP79 DONE: all 21 full-file real-colorway RGBA archives independently decodable",flush=True)


if __name__=="__main__":
    main()
