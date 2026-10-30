"""Experiment 82 — source audit only, no compression results.
Real historical GPL-2.0 OpenGFX2 graphics PNG edits from immutable git history.
See preregistered research/dedup/BHUH_EXP82_OPENGFX2_SOURCE_AUDIT_PROTOCOL.md.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
from PIL import Image, __version__ as PIL_VERSION
import numpy as np

SOURCE_REPO="https://github.com/OpenTTD/OpenGFX2.git"
SOURCE_LINK="https://github.com/OpenTTD/OpenGFX2"
HEAD="18f21b51a19df0f36635917e03aa84eb6e10e5b7"
COMMIT_COUNT=120
MAX_SOURCE_BYTES=12_000_000
MAX_DIMENSION=1024
MAX_SELECTED=24
SOURCE_LICENSE="GPL-2.0 (OpenGFX2 / OpenTTD art credits)"


def sha(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def git(directory:Path,*commands:str)->bytes:
    p=subprocess.run(["git","-C",str(directory),*commands],
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     timeout=120)
    if p.returncode:
        raise RuntimeError("Pinned Git historical source failed: "+
                           p.stderr.decode(errors="replace")[:1000])
    return p.stdout


def clone(cache:Path)->Path:
    cache.mkdir(parents=True,exist_ok=True)
    local=cache/"OpenGFX2"
    if not (local/".git").is_dir():
        result=subprocess.run(["git","clone","--quiet","--no-checkout",
             "--filter=blob:none",SOURCE_REPO,str(local)],
             stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
        if result.returncode:
            raise RuntimeError("Could not clone original source: "+
                   result.stderr.decode(errors="replace")[:1500])
    git(local,"checkout","--quiet",HEAD)
    if git(local,"rev-parse","HEAD").decode().strip()!=HEAD:
        raise ValueError("Moving external source head rejected")
    return local


def decode_png(raw:bytes):
    if not (8<len(raw)<=MAX_SOURCE_BYTES and raw[:8]==b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not PNG or source member too large")
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgba=np.ascontiguousarray(np.asarray(image.convert("RGBA"),dtype=np.uint8))
    return rgba


def scan(local:Path)->dict:
    commits=git(local,"rev-list",f"--max-count={COMMIT_COUNT}",
                "--first-parent",HEAD).decode().strip().splitlines()
    if len(commits)!=COMMIT_COUNT:
        raise ValueError("Missing original historical first-parent commits")
    records=[]
    eligible=[]
    rejected={}
    count_paths=0
    for index,(child,parent) in enumerate(zip(commits,commits[1:])):
        changed=git(local,"diff","--name-only","--diff-filter=M",
                    parent,child,"--","*.png").decode().strip().splitlines()
        for path in sorted(changed):
            count_paths+=1
            result={"transition_index":index,"parent_commit":parent,
                    "child_commit":child,"original_path":path}
            if not path.startswith("graphics/") or any(
                    key in path.lower() for key in
                    ("preview","spritesheet","tilemap",".import")):
                result["rejected"]="outside fixed graphics paths / source preview"
            else:
                try:
                    b=git(local,"show",f"{parent}:{path}")
                    a=git(local,"show",f"{child}:{path}")
                    before=decode_png(b)
                    after=decode_png(a)
                    result.update({
                        "before_png_sha256":sha(b),"after_png_sha256":sha(a),
                        "before_rgba_sha256":sha(before.tobytes()),
                        "after_rgba_sha256":sha(after.tobytes()),
                        "before_hwc":list(before.shape),"after_hwc":list(after.shape),
                        "source_png_bytes":[len(b),len(a)],
                        "has_transparent_pixels":[bool(np.any(before[:,:,3]<255)),
                                                  bool(np.any(after[:,:,3]<255))],
                    })
                    if before.shape!=after.shape:
                        raise ValueError("modified source PNG dimensions changed")
                    h,w,_=before.shape
                    if min(h,w)<16 or max(h,w)>MAX_DIMENSION:
                        raise ValueError("modified source PNG outside 16..1024 dimensions")
                    mismatch=int(np.count_nonzero(before!=after))
                    result["changed_rgba_channel_values"]=mismatch
                    if not mismatch:
                        raise ValueError("PNG file changed but RGBA pixel values identical")
                    result["combined_native_rgba_bytes"]=int(before.nbytes+after.nbytes)
                    result["qualified"]=True
                    eligible.append(result)
                except (OSError,ValueError,RuntimeError) as err:
                    result["rejected"]=str(err)
            if "rejected" in result:
                rejected[result["rejected"]]=rejected.get(result["rejected"],0)+1
            records.append(result)
    return {
        "experiment":82,"kind":"source-qualification-audit-not-codec-benchmark",
        "protocol":"research/dedup/BHUH_EXP82_OPENGFX2_SOURCE_AUDIT_PROTOCOL.md",
        "repository":SOURCE_LINK,"source_license":SOURCE_LICENSE,
        "fixed_head_sha":HEAD,
        "fixed_commit_count":COMMIT_COUNT,
        "consecutive_transitions":COMMIT_COUNT-1,
        "modified_png_paths_seen_in_diff":count_paths,
        "qualified_edited_rgba_pairs":len(eligible),
        "selected_first_24":eligible[:MAX_SELECTED],
        "rejected_count_by_reason":rejected,
        "all_modified_png_records":records,
        "software":{"python":platform.python_version(),"numpy":np.__version__,
                    "pillow":PIL_VERSION},
        "scientific_limit":"only one GPL-2.0 source project; audit, not compression result; correlated Git edits",
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache",default="/tmp/blkh_exp82_git")
    ap.add_argument("--out",default="/tmp/blkh_exp82")
    a=ap.parse_args()
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
    result=scan(clone(Path(a.cache)))
    (folder/"source_audit.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print("EXP82 SOURCE AUDIT: 119 pinned Git transitions, modified PNG",result["modified_png_paths_seen_in_diff"],
          "exact same-shape changed RGBA pairs",result["qualified_edited_rgba_pairs"],flush=True)
    for item in result["selected_first_24"]:
        print("ELIGIBLE",item["parent_commit"][:12],item["child_commit"][:12],
              item["original_path"],item["before_hwc"],
              "changed_rgba_channel_values",item["changed_rgba_channel_values"],flush=True)
    if result["qualified_edited_rgba_pairs"]<4:
        print("EXP82 SOURCE LIMIT: fewer than four paired source PNG revisions. No compression hypothesis tested.",flush=True)
    else:
        print("EXP82 DATASET AVAILABLE: next independently frozen compression protocol may proceed.",flush=True)


if __name__=="__main__":
    main()
