"""
Exp85: original 24 historical Mindustry sprites across 3 real authored commits.

External game project source vs previously studied Pixelorama icons.
Freeze actual first-eight sorted Git-modified PNG source filenames per commit,
then apply 100% UNMODIFIED Exp84 sparse codec and eleven original alternatives
plus true bsdiff4 binary patches. 56 complete archive files. This is source
transfer of an existing method, NOT fresh post-hoc optimization.
Protocol: BHUH_EXP85_MINDUSTRY_GIT_HOLDOUT_PROTOCOL.md
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import platform
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, __version__ as PIL_VERSION

import experiment_83_pixelorama_git_history as history
from experiment_84_sparse_pixel_patches import (
    NEW,OLD,BINARY,evaluate,shared_manifest
)

REPO="Anuken/Mindustry"
REPO_URL="https://github.com/Anuken/Mindustry"
LICENSE="GPL-3.0 (Mindustry repository root LICENSE; original game art)"
SOURCE_GROUPS=(
    ("unit_redesign","b9ccc528e933e705d7f737c781e57176aead5cc3"),
    ("conduit_redesign","b2a4cb42cf9c3a435ec5915e5b8a722c9e0b36f4"),
    ("block_redesign","f0f7b5aaf7edce01c023f408d0e48a2c8f552e0e"),
)
FILE_PREFIX="core/assets-raw/sprites/"
PIXELS_PER_GROUP=8
PNG_LIMIT=5_000_000


def configure_provenance()->None:
    # Exp83p's complete-file manifest uses module constants. Only the
    # *source identity fields* change here. The Exp84 codec, settings,
    # bitstreams, reference arm ordering and comparison logic are untouched.
    history.REPOSITORY=REPO
    history.REPOSITORY_URL=REPO_URL
    history.LICENSE=LICENSE


def get_json(url:str)->dict:
    req=urllib.request.Request(url,headers={
        "Accept":"application/vnd.github+json",
        "User-Agent":"BLKH-Exp85-reproducible-Git-artwork-benchmark/1.0",
    })
    with urllib.request.urlopen(req,timeout=50) as response:
        data=response.read(1_000_001)
    if len(data)>1_000_000:raise ValueError("Unexpected giant Git metadata response")
    return json.loads(data)


def commit_sources(commit:str)->tuple[str,list[str]]:
    info=get_json(f"https://api.github.com/repos/{REPO}/commits/{commit}")
    if info.get("sha")!=commit or len(info.get("parents",[]))!=1:
        raise ValueError("Frozen actual commit/parent invalid")
    parent=info["parents"][0]["sha"]
    compare=get_json(f"https://api.github.com/repos/{REPO}/compare/{parent}...{commit}")
    changed=sorted(f["filename"] for f in compare.get("files",[])
                   if f.get("status")=="modified" and
                   f.get("filename","").startswith(FILE_PREFIX) and
                   f.get("filename","").lower().endswith(".png"))
    if len(changed)<PIXELS_PER_GROUP:
        raise ValueError("Pinned source has fewer than eight original modified sprites")
    return parent,changed[:PIXELS_PER_GROUP]


def original_png(path:str,ref:str)->tuple[bytes,np.ndarray]:
    url=f"https://raw.githubusercontent.com/{REPO}/{ref}/"+urllib.parse.quote(path,safe="/")
    req=urllib.request.Request(url,headers={
        "User-Agent":"BLKH-Exp85-authentic-Git-PNG-reproduction/1.0",
    })
    with urllib.request.urlopen(req,timeout=50) as response:
        raw=response.read(PNG_LIMIT+1)
    if (not 0<len(raw)<=PNG_LIMIT or not raw.startswith(b"\x89PNG\r\n\x1a\n")):
        raise ValueError("Source not a real complete original PNG")
    with Image.open(io.BytesIO(raw)) as pic:
        pic.load()
        if not 1<=pic.width<=1024 or not 1<=pic.height<=1024:
            raise ValueError("Frozen original sprite outside preregistered geometry")
        rgba=np.asarray(pic.convert("RGBA"),dtype=np.uint8)
    return raw,rgba


def get_sources(folder:Path)->tuple[list[dict],dict]:
    folder.mkdir(parents=True,exist_ok=True)
    cache=folder/"sources"
    cache.mkdir(parents=True,exist_ok=True)
    all_assets=[]
    groups={}
    for name,commit in SOURCE_GROUPS:
        parent,paths=commit_sources(commit)
        manifest_paths=[]
        changed_pixels=0
        for index,path in enumerate(paths):
            pair=[]
            for role,ref in (("old",parent),("new",commit)):
                raw,rgba=original_png(path,ref)
                (cache/f"{name}_{index:02d}_{role}.png").write_bytes(raw)
                meta={
                    "source_repository":REPO,
                    "commit_label":name,"commit_sha":commit,
                    "parent_sha":parent,"file_path":path,
                    "pair_index":index,"role":role,
                    "filename":f"{name}:{index}:{role}:{path}",
                    "source_png_sha256":history.sha(raw),
                    "pixel_rgba_sha256":history.sha(rgba.tobytes()),
                    "width":int(rgba.shape[1]),"height":int(rgba.shape[0]),
                    "alpha_zero_pixels":int(np.count_nonzero(rgba[:,:,3]==0)),
                }
                pair.append({"rgba":rgba,"raw":raw,"meta":meta})
            unchanged=pair[0]["rgba"].shape==pair[1]["rgba"].shape and np.array_equal(
                pair[0]["rgba"],pair[1]["rgba"])
            changed_pixels+=int(not unchanged)
            manifest_paths.append({"path":path,"same_exact_rgba_pixels":bool(unchanged)})
            all_assets.extend(pair)
        groups[name]={
            "commit":commit,
            "first_parent":parent,
            "png_pairs_modified_in_git":PIXELS_PER_GROUP,
            "png_pairs_really_changed_rgba":changed_pixels,
            "status":manifest_paths,
        }
        print("EXP85 SOURCE",name,"real Git modified",len(paths),
              "actual visually changed",changed_pixels,flush=True)
    inventory={
        "repository":REPO,"repository_url":REPO_URL,"repository_license":LICENSE,
        "revision_groups":groups,
        "source_png_count":len(all_assets),
        "real_file_pair_count":len(all_assets)//2,
        "pixel_changed_pairs":sum(g["png_pairs_really_changed_rgba"] for g in groups.values()),
    }
    (folder/"real_mindustry_git_inventory.json").write_text(json.dumps(inventory,indent=2))
    if len(all_assets)!=2*PIXELS_PER_GROUP*len(SOURCE_GROUPS):
        raise ValueError("Frozen source count unexpected")
    return all_assets,inventory


def self_test():
    configure_provenance()
    assert REPO=="Anuken/Mindustry"
    assert len(SOURCE_GROUPS)==3 and PIXELS_PER_GROUP==8
    assert len(NEW)==2 and len(OLD)==11 and BINARY=="bsdiff_png_zstd"
    # The substantive stream integrity including true RGBA and changed
    # geometry is tested with the unchanged Exp84 test function in CI.
    print("EXP85 SELFTEST PASS: independent source metadata, 3 frozen commits, 24 historical PNG pairs; codec implementation reused")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",default="/tmp/blkh_exp85")
    a=p.parse_args()
    self_test()
    if a.self_test:return
    root=Path(a.out)
    configure_provenance()
    items,inv=get_sources(root)
    cohorts=[("all_24_pairs",items)]
    for label,commit in SOURCE_GROUPS:
        case=[i for i in items if i["meta"]["commit_label"]==label]
        assert len(case)==16
        cohorts.append((label,case))
    summary=[]
    for label,cohort in cohorts:
        result=evaluate(cohort,inv,label,root)
        summary.append(result)
        print("EXP85 RESULT",label,"pixels changed",
              round(result["changed_rgba_pixel_statistics"]["changed_pixel_fraction"],5),
              "sparse",result["best_sparse_codec"],result["best_sparse_bytes"],
              "reference",result["best_of_12_strong_references"],
              result["best_reference_bytes"],
              "win",result["sparse_smaller_than_best_reference"],flush=True)
        (root/"results.json").write_text(json.dumps({
            "experiment":85,
            "protocol":"research/dedup/BHUH_EXP85_MINDUSTRY_GIT_HOLDOUT_PROTOCOL.md",
            "actual_source":REPO,"license":LICENSE,
            "independent_repository_vs_exp84":True,
            "source_inventory":inv,
            "pairs":24,"images":48,
            "new_candidates":list(NEW),
            "references":[BINARY,*OLD],
            "software":{"python":platform.python_version(),
                        "Pillow":PIL_VERSION,"numpy":np.__version__},
            "results":summary,
        },indent=2))
    success=summary[0]["sparse_smaller_than_best_reference"] and sum(
        x["sparse_smaller_than_best_reference"] for x in summary[1:])>=2
    print("EXP85 COMPLETE: 24 genuine independent-repository source history PNG pairs, 56 complete archive decodes; preregistered gate",success,flush=True)


if __name__=="__main__":
    main()
