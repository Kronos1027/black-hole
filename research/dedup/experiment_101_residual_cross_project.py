"""Exp101: unchanged Exp100 exact residual codec transfer to Godot/SuperTux.

Frozen historical real source PNGs from Exp86 and Exp88. Not a newly
blinded independent holdout or public superiority claim.
"""
from __future__ import annotations
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from experiment_78_real_cc0_rgba_sprites import encode_webp
import experiment_83_pixelorama_git_history as base
import experiment_84_sparse_pixel_patches as sparse
import experiment_86_godot_png_holdout as godot
import experiment_88_supertux_png_history as supertux
import experiment_100_residual_transforms as residual

SOURCES=(("supertux",supertux,29,"all_29_pairs"),
         ("godot",godot,14,"all_14_pairs"))

def self_test():
    residual.self_test()
    supertux.self_test()
    godot.self_test()
    assert [s[2] for s in SOURCES]==[29,14]
    print("EXP101 SELFTEST PASS: frozen source membership and six reversible variants",flush=True)

def execute(out):
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label,source,expected,aggregate in SOURCES:
        family=out/label
        family.mkdir(parents=True,exist_ok=True)
        source.configure_sources()
        source.source_license(family)
        originals,inventory=source.discover(family)
        assert len(originals)==2*expected and inventory["real_file_pair_count"]==expected
        groups=[(aggregate,originals)]
        for group,_,paths in source.SOURCE_GROUPS:
            subset=[x for x in originals if x["meta"]["commit_label"]==group]
            assert len(subset)==2*len(paths)
            groups.append((group,subset))
        for group,subset in groups:
            references=sparse.evaluate(subset,inventory,group,family)
            manifest=references["shared_source_manifest"]
            old_atlas,old_positions=base.pack(subset[::2])
            assert list(old_atlas.shape)==manifest["old_atlas_shape"]
            assert old_positions==manifest["old_atlas_positions"]
            old_stream=encode_webp(old_atlas)
            start=time.perf_counter()
            plans=residual.plan_shifts(subset)
            select_seconds=time.perf_counter()-start
            methods={}
            for selector,shifts in plans.items():
                for mode in residual.TRANSFORMS:
                    began=time.perf_counter()
                    archive,stats=residual.encode(subset,manifest,shifts,mode,old_stream)
                    elapsed=time.perf_counter()-began
                    folder=family/group;folder.mkdir(parents=True,exist_ok=True)
                    filename=f"residual_{selector}_{mode}.bl100"
                    (folder/filename).write_bytes(archive)
                    rebuilt,decoded_meta=residual.decode(archive)
                    assert decoded_meta=={**manifest,"codec":residual.MODE}
                    assert all(np.array_equal(x["rgba"],y) for x,y in zip(subset,rebuilt))
                    methods[f"{selector}_{mode}"]=dict(filename=filename,
                        bytes=len(archive),sha256=base.sha(archive),
                        encode_seconds=round(elapsed,6),**stats)
            best=min(methods,key=lambda x:(methods[x]["bytes"],x))
            refsize=min(references["best_sparse_bytes"],references["best_reference_bytes"])
            row=dict(source=label,source_repository=source.SOURCE_REPO,
                group=group,real_pairs=len(subset)//2,
                manifest=manifest,selection_seconds=round(select_seconds,6),
                best_candidate=best,best_candidate_bytes=methods[best]["bytes"],
                best_reference_bytes=refsize,
                candidate_minus_reference_bytes=methods[best]["bytes"]-refsize,
                candidate_methods=methods,reference_results=references,exact_rgba=True)
            rows.append(row)
            print("EXP101",label,group,"best",best,methods[best]["bytes"],
                  "reference",refsize,"delta",methods[best]["bytes"]-refsize,flush=True)
    assert len(rows)==7
    data=dict(experiment=101,retrospective_cross_project=True,
              independent_blinded_holdout=False,release_eligible=False,
              source_repositories=[s[1].SOURCE_REPO for s in SOURCES],
              unique_source_pairs=43,total_persisted_archives=140,
              peak_process_maxrss_kib_linux=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              results=rows)
    (out/"results101.json").write_text(json.dumps(data,indent=2)+"\n")
    print("EXP101 complete; 140 persisted archives, retrospective only",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="/tmp/blkh_exp101")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:self_test()
    else:execute(Path(args.out))
