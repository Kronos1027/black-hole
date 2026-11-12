"""Exp96: fixed Exp95 selector transferred to frozen SuperTux real Git PNG edits.

This is a cross-project retrospective test (source already used in Exp88),
not a newly blinded independent holdout or a public release claim.
All candidate archives and all 14 historical reference archives are persisted.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import experiment_83_pixelorama_git_history as base
import experiment_84_sparse_pixel_patches as sparse
import experiment_88_supertux_png_history as source
import experiment_92_motion_shift_patch as motion
from experiment_95_compressed_shift_retrospective import choose_by_compressed_patch

def execute(out):
    out.mkdir(parents=True,exist_ok=True)
    source.configure_sources()
    source.source_license(out)
    items,inv=source.discover(out)
    assert len(items)==58 and inv["real_file_pair_count"]==29
    groups=[("all_29_pairs",items)]
    for label,_,paths in source.SOURCE_GROUPS:
        subset=[x for x in items if x["meta"]["commit_label"]==label]
        assert len(subset)==2*len(paths)
        groups.append((label,subset))
    rows=[]
    original=motion.select_shift
    try:
        for name,subset in groups:
            references=sparse.evaluate(subset,inv,name,out)
            manifest=references["shared_source_manifest"]
            motion.select_shift=original
            mismatch,ms=motion.encode(subset,manifest)
            motion.select_shift=choose_by_compressed_patch
            compressed,cs=motion.encode(subset,manifest)
            motion.select_shift=original
            folder=out/name
            for filename,stream in (("motion_mismatch.bl92m",mismatch),
                                    ("motion_compressed.bl92m",compressed)):
                (folder/filename).write_bytes(stream)
                reconstructed,metadata=motion.decode(stream)
                assert metadata=={**manifest,"codec":motion.MODE}
                assert all(np.array_equal(x["rgba"],y) for x,y in zip(subset,reconstructed))
            ref=references["best_reference_bytes"]
            prior=references["best_sparse_bytes"]
            best=min(ref,prior)
            rows.append(dict(group=name,pairs=len(subset)//2,
                mismatch_bytes=len(mismatch),compressed_bytes=len(compressed),
                best_reference_bytes=ref,best_sparse_bytes=prior,
                best_existing_bytes=best,
                gain_vs_best_existing_fraction=(best-len(compressed))/best,
                mismatch_sha256=base.sha(mismatch),compressed_sha256=base.sha(compressed),
                manifest=manifest,exact_rgba=True,baseline_results=references,
                mismatch_stats=ms,compressed_stats=cs))
            print("EXP96",name,"motion",len(mismatch),"compressed-choice",len(compressed),
                  "best prior",best,"gain fraction",(best-len(compressed))/best,flush=True)
    finally:
        motion.select_shift=original
    (out/"results96.json").write_text(json.dumps(dict(experiment=96,
        source_repository=source.SOURCE_REPO,real_pairs=29,
        retrospective_cross_project=True,independent_blinded_holdout=False,
        release_eligible=False,results=rows),indent=2)+"\n")
    print("EXP96 complete; exact archives, transfer retrospective only",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="/tmp/blkh_exp96")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        source.self_test()
        a=np.zeros((8,8,4),dtype=np.uint8)
        assert len(choose_by_compressed_patch(a,a))==3
        print("EXP96 self-test pass")
    else:execute(Path(args.out))
