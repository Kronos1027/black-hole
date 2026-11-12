"""Exp102: preregistered three new Git PNG source families (zero pixel preselection).

16 original parent/child modified PNG pairs from Luanti, Shattered Pixel Dungeon,
and Widelands. Use frozen Exp100 six reversible variants and all 14 Exp84
same-fidelity references. No release claim before audited independent results.
"""
from __future__ import annotations
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from experiment_78_real_cc0_rgba_sprites import encode_webp
import experiment_83_pixelorama_git_history as base
import experiment_84_sparse_pixel_patches as sparse
import experiment_100_residual_transforms as residual

PROTOCOL="research/dedup/BHUH_EXP102_THREE_NEW_GIT_SOURCES_PREREGISTRATION.md"
SOURCES=(
  dict(label="luanti_leaf_art",repository="luanti-org/minetest_game",
       commit="960aff2f6366c2de864bdd9d4fd4f5eafa956bf3",
       license_file="LICENSE.txt",license_marker="Attribution-ShareAlike",
       declared_license="Project root LICENSE.txt includes media CC BY-SA 3.0, per-asset attribution must be checked",
       paths=(
         "mods/default/textures/default_acacia_leaves.png",
         "mods/default/textures/default_acacia_leaves_simple.png",
         "mods/default/textures/default_aspen_leaves.png",
         "mods/default/textures/default_blueberry_bush_leaves.png",
         "mods/default/textures/default_blueberry_overlay.png",
         "mods/default/textures/default_jungleleaves.png",
         "mods/default/textures/default_jungleleaves_simple.png",
         "mods/default/textures/default_leaves.png",
         "mods/default/textures/default_leaves_simple.png",
         "mods/default/textures/default_pine_needles.png")),
  dict(label="shattered_item_art",repository="00-Evan/shattered-pixel-dungeon",
       commit="14b1a7f2b7b656ffab922c349bc42aee39c04cb2",
       license_file="LICENSE.txt",license_marker="Version 3",
       declared_license="Project root GPLv3, per-asset image authorship must be checked",
       paths=(
         "core/src/main/assets/interfaces/icons.png",
         "core/src/main/assets/sprites/items.png")),
  dict(label="widelands_frisians",repository="widelands/widelands",
       commit="30d09f1ba2510478f73de7126d851ceb84f79260",
       license_file="COPYING",license_marker="Version 2",
       declared_license="Project root GPLv2, per-asset image authorship must be checked",
       paths=(
         "data/tribes/buildings/militarysites/frisians/outpost/idle_1.png",
         "data/tribes/buildings/militarysites/frisians/outpost/idle_1_pc.png",
         "data/tribes/buildings/militarysites/frisians/outpost/idle_2.png",
         "data/tribes/buildings/militarysites/frisians/outpost/idle_2_pc.png")),
)
EXPECTED_PAIRS=16

def configure(source):
    base.REPOSITORY=source["repository"]
    base.REPOSITORY_URL=f"https://github.com/{source['repository']}"
    base.LICENSE=source["declared_license"]
    def manifest(items,inventory):
        result=base.manifest_for(items,inventory)
        result["paired_source_order"]=(
            "Exp102 preregistered original commit paths in fixed lexicographic order; "
            "first parent then child; three previously unseen source projects")
        return result
    sparse.manifest_for=manifest

def first_parent(source):
    endpoint=f"https://api.github.com/repos/{source['repository']}/commits/{source['commit']}"
    info=json.loads(base.request_bytes(endpoint,600_000))
    if info.get("sha")!=source["commit"] or len(info.get("parents",[]))!=1:
        raise ValueError("Predeclared source commit is missing or has more than one parent")
    parent=info["parents"][0]["sha"]
    if len(parent)!=40:raise ValueError("Malformed upstream parent SHA")
    return parent

def verify_modified_paths(source,parent):
    url=(f"https://api.github.com/repos/{source['repository']}/compare/"
         f"{parent}...{source['commit']}")
    info=json.loads(base.request_bytes(url,5_000_000))
    if info.get("ahead_by")!=1 or info.get("status") not in ("ahead","identical"):
        raise ValueError("Original historical source revision not an adjacent commit")
    changed={f["filename"] for f in info.get("files",[]) if f["status"]=="modified"}
    if not set(source["paths"]).issubset(changed):
        raise ValueError("One or more frozen source paths not genuinely modified")
    return len(changed)

def source_license(out,source):
    url=(f"https://raw.githubusercontent.com/{source['repository']}/"
         f"{source['commit']}/{source['license_file']}")
    raw=base.request_bytes(url,150_000)
    if source["license_marker"].encode() not in raw:
        raise ValueError("Frozen source project's root license not verified")
    (out/"UPSTREAM_ROOT_LICENSE.txt").write_bytes(raw)
    (out/"SOURCE_ATTRIBUTION.txt").write_text(
       f"Original upstream Git images: {source['repository']}\n"
       f"Commit: {source['commit']}\n"
       f"Root license: {source['license_file']}\n"
       f"Rights caution: {source['declared_license']}\n"
       "Full file paths, parent SHA and original PNG byte SHA256 in source_inventory.json.\n"
       "Source images are third-party content; verify author credits before any public redistribution.\n")

def discover(out,source):
    configure(source)
    out.mkdir(parents=True,exist_ok=True)
    source_license(out,source)
    parent=first_parent(source)
    modified_count=verify_modified_paths(source,parent)
    folder=out/"sources";folder.mkdir(parents=True,exist_ok=True)
    items=[]
    changed=0
    for i,path in enumerate(source["paths"]):
        pair=[]
        for role,sha in (("old",parent),("new",source["commit"])):
            raw,rgba=base.source_png(sha,path)
            (folder/f"{i:02d}_{role}.png").write_bytes(raw)
            metadata=dict(source_repository=source["repository"],
                source_original_rights=source["declared_license"],
                commit_label=source["label"],commit_sha=source["commit"],
                parent_sha=parent,file_path=path,pair_index=i,role=role,
                filename=f"{source['label']}:{i}:{role}:{path}",
                source_png_sha256=base.sha(raw),
                pixel_rgba_sha256=base.sha(rgba.tobytes()),
                width=int(rgba.shape[1]),height=int(rgba.shape[0]),
                alpha_zero_pixels=int(np.count_nonzero(rgba[:,:,3]==0)))
            pair.append(dict(rgba=rgba,raw=raw,meta=metadata))
        changed+=int(pair[0]["rgba"].shape!=pair[1]["rgba"].shape or
                     not np.array_equal(pair[0]["rgba"],pair[1]["rgba"]))
        items.extend(pair)
    inventory=dict(source_repository=source["repository"],
        license=source["declared_license"],revision_groups={
            source["label"]:dict(commit=source["commit"],first_parent=parent,
                png_pairs_modified_in_git=len(source["paths"]),
                png_pairs_really_changed_rgba=changed,
                paths=list(source["paths"]))},
        source_png_count=len(items),real_file_pair_count=len(items)//2,
        pixel_changed_pairs=changed,
        frozen_modified_path_count=modified_count,
        previously_used_in_BLKH=False,
        protocol=PROTOCOL)
    if len(items)!=2*len(source["paths"]):
        raise AssertionError("Exp102 frozen source pairs incomplete")
    (out/"source_inventory.json").write_text(json.dumps(inventory,indent=2)+"\n")
    return items,inventory

def self_test():
    residual.self_test()
    assert [len(s["paths"]) for s in SOURCES]==[10,2,4]
    assert sum(len(s["paths"]) for s in SOURCES)==EXPECTED_PAIRS
    assert len({s["repository"] for s in SOURCES})==3
    assert len({s["commit"] for s in SOURCES})==3
    assert all(len(s["commit"])==40 for s in SOURCES)
    assert all(tuple(sorted(s["paths"]))==s["paths"] for s in SOURCES)
    print("EXP102 SELFTEST PASS: 16 fixed real modified PNG paths across 3 new projects")

def execute(out):
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for source in SOURCES:
        group=source["label"]
        folder=out/group
        items,inventory=discover(folder,source)
        references=sparse.evaluate(items,inventory,group,folder)
        meta=references["shared_source_manifest"]
        old_atlas,positions=base.pack(items[::2])
        assert list(old_atlas.shape)==meta["old_atlas_shape"] and positions==meta["old_atlas_positions"]
        old_webp=encode_webp(old_atlas)
        t0=time.perf_counter()
        plans=residual.plan_shifts(items)
        select_time=time.perf_counter()-t0
        methods={}
        for selector,shifts in plans.items():
            for mode in residual.TRANSFORMS:
                t0=time.perf_counter()
                archive,stats=residual.encode(items,meta,shifts,mode,old_webp)
                seconds=time.perf_counter()-t0
                dst=folder/group;dst.mkdir(parents=True,exist_ok=True)
                filename=f"residual_{selector}_{mode}.bl100"
                (dst/filename).write_bytes(archive)
                rebuilt,decoded_meta=residual.decode(archive)
                assert decoded_meta=={**meta,"codec":residual.MODE}
                assert all(np.array_equal(x["rgba"],y) for x,y in zip(items,rebuilt))
                methods[f"{selector}_{mode}"]=dict(filename=filename,
                    bytes=len(archive),sha256=base.sha(archive),
                    encode_seconds=round(seconds,6),**stats)
        best=min(methods,key=lambda x:(methods[x]["bytes"],x))
        prior=min(references["best_reference_bytes"],references["best_sparse_bytes"])
        row=dict(source=group,source_repository=source["repository"],
            child_commit=source["commit"],real_pairs=len(items)//2,
            pixel_changed_pairs=inventory["pixel_changed_pairs"],
            manifest=meta,selection_seconds=round(select_time,6),
            best_candidate=best,best_candidate_bytes=methods[best]["bytes"],
            best_reference_bytes=prior,
            candidate_minus_reference_bytes=methods[best]["bytes"]-prior,
            candidate_methods=methods,reference_results=references,
            exact_rgba=True,release_eligible=False)
        rows.append(row)
        print("EXP102",group,"changed",inventory["pixel_changed_pairs"],
              "best candidate",best,methods[best]["bytes"],
              "best reference",prior,"diff",methods[best]["bytes"]-prior,flush=True)
    assert len(rows)==3
    data=dict(experiment=102,protocol=PROTOCOL,
        new_source_repositories=True,independent_source_families=3,
        source_pairs=EXPECTED_PAIRS,release_eligible=False,
        total_persisted_archives=60,
        peak_process_maxrss_kib_linux=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        results=rows)
    (out/"results102.json").write_text(json.dumps(data,indent=2)+"\n")
    print("EXP102 complete: 60 self-contained archives, public release remains HOLD",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="/tmp/blkh_exp102")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:self_test()
    else:execute(Path(args.out))
