"""Exp98: global compressed-stream motion shift ablation on frozen Wesnoth.

Unlike Exp95's independent per-pair compressed patch choice, optimize the
*whole joint Zstd patch stream* using deterministic coordinate descent over
three frozen candidate shift sets: mismatch-min, per-pair zstd-min, and zero.
This is retrospective optimization, NOT an unseen holdout or release result.
"""
from __future__ import annotations
import argparse,json,time,resource
from pathlib import Path
import numpy as np
import experiment_83_pixelorama_git_history as base
import experiment_84_sparse_pixel_patches as sparse
import experiment_87_wesnoth_real_maintenance as source
import experiment_92_motion_shift_patch as motion
from experiment_95_compressed_shift_retrospective import choose_by_compressed_patch

def record(old,new,shift):
    dx,dy,_=shift
    pred=motion.predicted(old,new.shape,dx,dy)
    mask=np.any(pred!=new,axis=2).reshape(-1)
    count=int(mask.sum())
    return (motion.RECORD.pack(dx,dy,count)+np.packbits(mask,bitorder="little").tobytes()+
            np.ascontiguousarray(new.reshape(-1,4)[mask]).tobytes())

def optimize(items):
    pairs=list(zip(items[::2],items[1::2]))
    choices=[]
    for a,b in pairs:
        old,new=a["rgba"],b["rgba"]
        shiftlist=[motion.select_shift(old,new),choose_by_compressed_patch(old,new),(0,0,0)]
        unique={}
        mapping=[]
        for shift in shiftlist:
            candidate=record(old,new,shift)
            if candidate not in unique:
                unique[candidate]=len(unique)
            mapping.append(unique[candidate])
        choices.append(([(shiftlist[next(i for i,s in enumerate(shiftlist) if record(old,new,s)==key)],key) for key in unique],mapping))
    states={}
    for strategy in ("mismatch","local_zstd","zero"):
        index={"mismatch":0,"local_zstd":1,"zero":2}[strategy]
        selected=[]
        for opts,mapping in choices:
            selected.append(mapping[index])
        states[strategy]=selected
    def size(indices):
        return len(sparse.zcompress(b"".join(choices[i][0][j][1] for i,j in enumerate(indices))))
    best_name,minstate=min(states.items(),key=lambda kv:(size(kv[1]),kv[0]))
    best=minstate.copy()
    start=size(best)
    passes=0
    for _ in range(3):
        improved=False
        for i,(opts,_) in enumerate(choices):
            current=best[i]
            local=min(range(len(opts)),key=lambda j:(size(best[:i]+[j]+best[i+1:]),j))
            if size(best[:i]+[local]+best[i+1:])<size(best):
                best[i]=local
                improved=True
        passes+=1
        if not improved:break
    final=size(best)
    assert final<=start
    return [opts[j][0] for (opts,_),j in zip(choices,best)],dict(
        initial_joint_patch_bytes=start,optimized_joint_patch_bytes=final,
        baseline_start=best_name,passes=passes,nonzero_shift_pairs=sum(
            bool(choices[i][0][j][0][0] or choices[i][0][j][0][1]) for i,j in enumerate(best)))

def self_test():
    a=np.zeros((8,9,4),dtype=np.uint8);b=a.copy();b[2,3]=[10,20,30,0]
    assert record(a,b,(0,0,0))==record(a,b,(0,0,1))
    fixtures=[]
    for i,(old,new) in enumerate(((a,b),(b,a))):
        fixtures.extend(({"rgba":old,"meta":{"pixel_rgba_sha256":base.sha(old.tobytes())}},
                         {"rgba":new,"meta":{"pixel_rgba_sha256":base.sha(new.tobytes())}}))
    selected,diagnostics=optimize(fixtures)
    assert len(selected)==2
    assert diagnostics["optimized_joint_patch_bytes"]<=diagnostics["initial_joint_patch_bytes"]
    print("EXP98 selftest PASS",flush=True)

def execute(out):
    out.mkdir(parents=True,exist_ok=True)
    source.configure_sources()
    source.source_license(out)
    items,inv=source.discover(out)
    assert len(items)==42 and inv["real_file_pair_count"]==21
    groups=[("all_21_pairs",items)]
    for label,_,paths in source.SOURCE_GROUPS:
        selected=[x for x in items if x["meta"]["commit_label"]==label]
        assert len(selected)==2*len(paths)
        groups.append((label,selected))
    rows=[]
    original=motion.select_shift
    try:
        for name,subset in groups:
            manifest=sparse.shared_manifest(subset,inv)
            motion.select_shift=original
            old,oldstats=motion.encode(subset,manifest)
            motion.select_shift=choose_by_compressed_patch
            local,localstats=motion.encode(subset,manifest)
            motion.select_shift=original
            start_wall=time.perf_counter()
            selected_shifts,diagnostics=optimize(subset)
            diagnostics["optimizer_wall_seconds"]=round(time.perf_counter()-start_wall,6)
            diagnostics["peak_process_maxrss_kib_linux"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            shift_iter=iter(selected_shifts)
            def chosen(a,b):
                return next(shift_iter)
            motion.select_shift=chosen
            global_archive,globalstats=motion.encode(subset,manifest)
            assert next(shift_iter,None) is None
            motion.select_shift=original
            folder=out/name;folder.mkdir(parents=True,exist_ok=True)
            for filename,archive in (("mismatch.bl92m",old),("local.bl92m",local),
                                     ("global.bl92m",global_archive)):
                (folder/filename).write_bytes(archive)
                decoded,meta=motion.decode(archive)
                assert meta=={**manifest,"codec":motion.MODE}
                assert all(np.array_equal(x["rgba"],y) for x,y in zip(subset,decoded))
            rows.append(dict(group=name,real_pairs=len(subset)//2,
                mismatch_bytes=len(old),local_bytes=len(local),global_bytes=len(global_archive),
                best_historical_bytes={"all_21_pairs":64802,"thea_shadow_fix":43783,
                    "dune_firetrooper_cleanup":13119,"ant_sprite_distinction":9916}[name],
                mismatch_sha256=base.sha(old),local_sha256=base.sha(local),
                global_sha256=base.sha(global_archive),manifest=manifest,
                diagnostics=diagnostics,exact_rgba=True))
            print("EXP98",name,"mismatch",len(old),"local",len(local),
                  "global",len(global_archive),flush=True)
    finally:
        motion.select_shift=original
    (out/"results98.json").write_text(json.dumps(dict(experiment=98,retrospective=True,
        independent_holdout=False,release_eligible=False,results=rows),indent=2)+"\n")
    print("EXP98 complete; no public release claim",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="/tmp/blkh_exp98")
    p.add_argument("--self-test",action="store_true");args=p.parse_args()
    self_test()
    if not args.self_test:execute(Path(args.out))
