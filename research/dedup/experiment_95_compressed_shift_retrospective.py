"""Exp95: retrospective *actual compressed-byte* shift choice, Wesnoth.

Exploratory and correlated with Exp93. No independent release claim.
The new selector minimizes each pair's serialized zstd9 patch length, not
pixel mismatches. Final whole-archive bytes are measured, not estimated.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import experiment_83_pixelorama_git_history as base
import experiment_84_sparse_pixel_patches as sparse
import experiment_87_wesnoth_real_maintenance as source
import experiment_92_motion_shift_patch as motion

def choose_by_compressed_patch(old,new):
    best=None
    for dx,dy in motion.SHIFTS:
        pred=motion.predicted(old,new.shape,dx,dy)
        mask=np.any(pred!=new,axis=2).reshape(-1)
        n=int(mask.sum())
        raw=(motion.RECORD.pack(dx,dy,n)+
             np.packbits(mask,bitorder="little").tobytes()+
             np.ascontiguousarray(new.reshape(-1,4)[mask]).tobytes())
        score=len(sparse.zcompress(raw))
        key=(score,len(raw),abs(dx)+abs(dy),abs(dy),abs(dx),dy,dx)
        if best is None or key<best[0]:
            best=(key,(dx,dy,n))
    return best[1]

def self_test():
    motion.self_test()
    a=np.zeros((9,11,4),dtype=np.uint8)
    b=a.copy();b[1,2]=[255,4,2,0]
    x,y,n=choose_by_compressed_patch(a,b)
    assert n==int(np.count_nonzero(np.any(motion.predicted(a,b.shape,x,y)!=b,axis=2)))
    print("EXP95 selector selftest PASS; retrospective only",flush=True)

def execute(out):
    out.mkdir(parents=True,exist_ok=True)
    source.configure_sources()
    source.source_license(out)
    originals,inv=source.discover(out)
    assert len(originals)==42 and inv["real_file_pair_count"]==21
    groups=[("all_21_pairs",originals)]
    for label,_,paths in source.SOURCE_GROUPS:
        selected=[x for x in originals if x["meta"]["commit_label"]==label]
        assert len(selected)==2*len(paths)
        groups.append((label,selected))
    rows=[]
    original_selector=motion.select_shift
    try:
        for name,items in groups:
            meta=sparse.shared_manifest(items,inv)
            motion.select_shift=original_selector
            old_stream,old_stats=motion.encode(items,meta)
            motion.select_shift=choose_by_compressed_patch
            new_stream,new_stats=motion.encode(items,meta)
            motion.select_shift=original_selector
            decoded,decoded_meta=motion.decode(new_stream)
            assert decoded_meta=={**meta,"codec":motion.MODE}
            assert all(np.array_equal(x["rgba"],y) for x,y in zip(items,decoded))
            folder=out/name;folder.mkdir(parents=True,exist_ok=True)
            (folder/"mismatch_min.bl92m").write_bytes(old_stream)
            (folder/"compressed_patch_min.bl92m").write_bytes(new_stream)
            rows.append(dict(group=name,old_bytes=len(old_stream),new_bytes=len(new_stream),
                difference_bytes=len(new_stream)-len(old_stream),
                best_prior_reference_bytes={"all_21_pairs":64802,
                    "thea_shadow_fix":43783,"dune_firetrooper_cleanup":13119,
                    "ant_sprite_distinction":9916}[name],
                old_sha256=base.sha(old_stream),new_sha256=base.sha(new_stream),
                exact_rgba=True,old_stats=old_stats,new_stats=new_stats))
            print("EXP95",name,"mismatch",len(old_stream),
                "compressed-patch selector",len(new_stream),flush=True)
    finally:
        motion.select_shift=original_selector
    data=dict(experiment=95,retrospective=True,independent=False,
        release_eligible=False,method="per-pair compressed zstd patch bytes",
        results=rows)
    (out/"results.json").write_text(json.dumps(data,indent=2)+"\n")
    print("EXP95 COMPLETE: retrospective exploration, NOT independent validation",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",default="/tmp/blkh_exp95")
    args=p.parse_args()
    self_test()
    if not args.self_test:execute(Path(args.out))
