"""Exp93: exact motion-compensated sprite-history codec on already-known
Wesnoth Git PNG corpus. Retrospective engineering only; NOT a fresh holdout.
Protocol BHUH_EXP93_WESNOTH_MOTION_RETROSPECTIVE_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import __version__ as PIL_VERSION
import zstandard as zstd

import experiment_83_pixelorama_git_history as old
import experiment_84_sparse_pixel_patches as sparse
import experiment_87_wesnoth_real_maintenance as source
import experiment_92_motion_shift_patch as motion

PROTOCOL="research/dedup/BHUH_EXP93_WESNOTH_MOTION_RETROSPECTIVE_PROTOCOL.md"


def self_test() -> None:
    source.self_test()
    motion.self_test()
    assert len(sparse.ALL)==14 and len(sparse.NEW)==2
    assert [len(g[2]) for g in source.SOURCE_GROUPS]==[14,4,3]
    print("EXP93 SELFTEST PASS: frozen 21 original PNG pairs, 14 static methods + 1 exact motion codec",flush=True)


def execute(out: Path) -> None:
    out.mkdir(parents=True,exist_ok=True)
    source.configure_sources()
    source.source_license(out)
    originals, inv=source.discover(out)
    if len(originals)!=42 or inv["real_file_pair_count"]!=21:
        raise ValueError("Frozen source qualification failure")
    cases=[("all_21_pairs",originals)]
    for label,_,paths in source.SOURCE_GROUPS:
        selected=[x for x in originals if x["meta"]["commit_label"]==label]
        if len(selected)!=2*len(paths):
            raise ValueError("Missing source from original frozen Wesnoth commit")
        cases.append((label,selected))
    results=[]
    for label,items in cases:
        row=sparse.evaluate(items,inv,label,out)
        begin=time.perf_counter_ns()
        encoded,stats=motion.encode(items,row["shared_source_manifest"])
        encode_seconds=(time.perf_counter_ns()-begin)/1e9
        dest=out/label/(motion.MODE+".bl92m")
        dest.write_bytes(encoded)
        begin=time.perf_counter_ns()
        raw=dest.read_bytes()
        decoded,meta=motion.decode(raw)
        decode_seconds=(time.perf_counter_ns()-begin)/1e9
        assert meta=={**row["shared_source_manifest"],"codec":motion.MODE}
        assert len(decoded)==len(items)
        assert all(np.array_equal(v["rgba"],new) for v,new in zip(items,decoded))
        row["methods"][motion.MODE]={
            "filename":dest.name,"bytes":len(raw),"sha256":old.sha(raw),
            "encode_seconds":encode_seconds,"decode_seconds":decode_seconds,
            **stats
        }
        row["new_motion_bytes"]=len(raw)
        row["new_motion_smaller_than_best_prior_sparse"]=len(raw)<row["best_sparse_bytes"]
        row["new_motion_smaller_than_best_reference"]=len(raw)<row["best_reference_bytes"]
        row["new_motion_smallest_all_15"]=len(raw)<min(row["methods"][m]["bytes"] for m in sparse.ALL)
        results.append(row)
        print("EXP93",label,"motion",len(raw),
              "shifts",stats["nonzero_motion_pairs"],
              "motion changed fraction",stats["motion_after_pixel_difference_fraction"],
              "no-motion sparse",row["best_sparse_bytes"],
              "best of 12 refs",row["best_reference_bytes"],
              "all-15-best",row["new_motion_smallest_all_15"],flush=True)
    qualified=inv["pixel_changed_pairs"]>=2
    groups_won=sum(int(row["new_motion_smaller_than_best_reference"]) for row in results[1:])
    observation=bool(qualified and results[0]["new_motion_smallest_all_15"] and groups_won>=2)
    payload={
      "experiment":93,"protocol":PROTOCOL,"source_repo":source.SOURCE_REPO,
      "retrospective_known_source":True,"not_independent_replication":True,
      "source_revisions":inv,
      "methods":list(sparse.ALL)+(motion.MODE,),
      "candidate_methods":list(sparse.NEW)+(motion.MODE,),
      "reference_methods":list((sparse.BINARY,)+sparse.OLD),
      "software":{"python":platform.python_version(),"numpy":np.__version__,
                  "pillow":PIL_VERSION,"zstandard":zstd.__version__},
      "qualified":qualified,"motion_group_wins_vs_reference":groups_won,
      "motion_retrospective_gate_observation":observation,
      "results":results,
    }
    (out/"results.json").write_text(json.dumps(payload,indent=2))
    assert len(results)==4 and len(payload["methods"])==15
    print("EXP93 FINISHED: 60 full exact archives; retrospective observation",observation,
          "; independent source verification remains separate",flush=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="/tmp/blkh_exp93")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    self_test() if args.self_test else execute(Path(args.out))


if __name__=="__main__":
    main()
