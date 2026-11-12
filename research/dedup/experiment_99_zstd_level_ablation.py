"""Exp99: Zstandard level ablation on frozen real Wesnoth exact motion archives.

Retrospective: test compression level 3,9,19 for both mismatch-min and
Exp95 compressed-byte motion selectors. Full archive bytes, time and exact
decode are measured; no new independent holdout.
"""
from __future__ import annotations
import argparse,json,struct,time,resource
from pathlib import Path
import numpy as np
import zstandard as zstd
import experiment_83_pixelorama_git_history as base
import experiment_84_sparse_pixel_patches as sparse
import experiment_87_wesnoth_real_maintenance as source
import experiment_92_motion_shift_patch as motion
from experiment_95_compressed_shift_retrospective import choose_by_compressed_patch

LEVELS=(3,9,19)
def repack(archive,level):
    magic,version,n=motion.HEAD.unpack_from(archive)
    assert magic==motion.MAGIC and version==1
    at=motion.HEAD.size+n
    nold,npatch=motion.LENGTHS.unpack_from(archive,at)
    start=at+motion.LENGTHS.size
    assert start+nold+npatch==len(archive)
    old=archive[start:start+nold]
    patch=archive[start+nold:]
    raw=zstd.ZstdDecompressor().decompress(patch,max_output_size=sparse.MAX_ARCHIVE)
    begin=time.perf_counter()
    compressed=zstd.ZstdCompressor(level=level).compress(raw)
    seconds=time.perf_counter()-begin
    packed=archive[:at]+motion.LENGTHS.pack(nold,len(compressed))+old+compressed
    return packed,seconds,len(raw)

def execute(out):
    out.mkdir(parents=True,exist_ok=True)
    source.configure_sources();source.source_license(out)
    items,inv=source.discover(out)
    assert len(items)==42 and inv["real_file_pair_count"]==21
    groups=[("all_21_pairs",items)]
    for label,_,paths in source.SOURCE_GROUPS:
        group=[x for x in items if x["meta"]["commit_label"]==label]
        assert len(group)==2*len(paths)
        groups.append((label,group))
    rows=[];original=motion.select_shift
    try:
        for name,subset in groups:
            manifest=sparse.shared_manifest(subset,inv)
            methods={}
            for selector,func in (("mismatch",original),("compressed",choose_by_compressed_patch)):
                motion.select_shift=func
                initial,_=motion.encode(subset,manifest)
                motion.select_shift=original
                for level in LEVELS:
                    stream,elapsed,rawlen=repack(initial,level)
                    folder=out/name;folder.mkdir(parents=True,exist_ok=True)
                    filename=f"{selector}_zstd{level}.bl92m"
                    (folder/filename).write_bytes(stream)
                    decoded,meta=motion.decode(stream)
                    assert meta=={**manifest,"codec":motion.MODE}
                    assert all(np.array_equal(x["rgba"],y) for x,y in zip(subset,decoded))
                    methods[f"{selector}_zstd{level}"]=dict(filename=filename,bytes=len(stream),
                        sha256=base.sha(stream),patch_uncompressed_bytes=rawlen,
                        zstd_compress_seconds=round(elapsed,6))
            prior={"all_21_pairs":64802,"thea_shadow_fix":43783,
                   "dune_firetrooper_cleanup":13119,"ant_sprite_distinction":9916}[name]
            best=min(methods,key=lambda x:methods[x]["bytes"])
            rows.append(dict(group=name,real_pairs=len(subset)//2,manifest=manifest,
                methods=methods,best_method=best,best_bytes=methods[best]["bytes"],
                best_historical_bytes=prior,exact_rgba=True))
            print("EXP99",name,"best",best,methods[best]["bytes"],
                  "prior",prior,flush=True)
    finally:motion.select_shift=original
    (out/"results99.json").write_text(json.dumps(dict(experiment=99,
        retrospective=True,independent_holdout=False,release_eligible=False,
        zstd_levels=list(LEVELS),results=rows,
        peak_process_maxrss_kib_linux=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+"\n")
    print("EXP99 complete; retrospective only",flush=True)

def self_test():
    a=np.zeros((8,8,4),dtype=np.uint8);b=a.copy();b[2,3]=[1,2,3,0]
    motion.self_test()
    assert len(LEVELS)==3
    print("EXP99 selftest PASS")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="/tmp/blkh_exp99")
    p.add_argument("--self-test",action="store_true");args=p.parse_args()
    self_test() if args.self_test else execute(Path(args.out))
