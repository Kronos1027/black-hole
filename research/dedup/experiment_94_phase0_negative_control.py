"""Exp94 phase-0: deterministic retrospective control and independent-release gate.

No fresh corpus is consumed here. This test deliberately rejects oracle selection
and overlapping cohorts as independent replication.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

GROUPS = {
    "all_21_pairs": (68051, 64802, 65259),
    "thea_shadow_fix": (47012, 43783, 43964),
    "dune_firetrooper_cleanup": (13305, 13292, 13119),
    "ant_sprite_distinction": (9949, 9939, 9916),
}
MIN_FAMILIES = 3
MIN_SAVING = 0.10

def analyze(groups=GROUPS):
    rows = []
    for name, (motion, sparse, reference) in groups.items():
        winner = min(sparse, reference)
        rows.append({
            "group": name,
            "motion_bytes": motion,
            "best_prior_bytes": winner,
            "motion_penalty_bytes": motion-winner,
            "motion_penalty_pct": round(100*(motion-winner)/winner, 6),
            "motion_wins": motion < winner,
        })
    return rows

def release_gate(families):
    """Families must be disjoint, unseen, audited, equal-fidelity, CI-reproduced."""
    if len(families) < MIN_FAMILIES:
        return False
    names = [f["source_family"] for f in families]
    if len(set(names)) != len(names):
        return False
    for f in families:
        if not (f["unseen"] and f["independent_source"] and
                f["exact_decode_verified"] and f["same_fidelity_baseline"] and
                f["full_archive_counted"] and f["ci_reproduced"] and
                f["license_verified"] and f["latency_measured"] and
                f["peak_rss_measured"]):
            return False
        if f["candidate_bytes"] > (1-MIN_SAVING)*f["best_baseline_bytes"]:
            return False
        if f["decode_latency_ratio"] > 2 or f["peak_rss_ratio"] > 2:
            return False
    return True

def self_test():
    rows = analyze()
    assert len(rows)==4 and not any(x["motion_wins"] for x in rows)
    assert [x["motion_penalty_bytes"] for x in rows] == [3249,3229,186,33]
    assert sum(x["motion_penalty_bytes"] for x in rows[1:]) != rows[0]["motion_penalty_bytes"]
    # The full 21-pair aggregate overlaps all three subgroups, not four samples.
    assert not release_gate([])
    qualified = dict(source_family="a", unseen=True, independent_source=True,
        exact_decode_verified=True, same_fidelity_baseline=True,
        full_archive_counted=True, ci_reproduced=True, license_verified=True,
        latency_measured=True, peak_rss_measured=True, candidate_bytes=85,
        best_baseline_bytes=100, decode_latency_ratio=1.5, peak_rss_ratio=1.5)
    assert not release_gate([qualified]*3)  # duplicated family
    assert not release_gate([qualified, {**qualified,"source_family":"b"}])
    assert release_gate([{**qualified,"source_family":n} for n in "abc"])
    assert not release_gate([{**qualified,"source_family":n,
        "candidate_bytes":95} for n in "abc"])
    assert not release_gate([{**qualified,"source_family":n,
        "same_fidelity_baseline":False} for n in "abc"])
    assert not release_gate([{**qualified,"source_family":n,
        "decode_latency_ratio":2.1} for n in "abc"])
    print("EXP94 PHASE-0 SELFTEST PASS: 4 retrospective rows, overlap and release gate guards")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",default="/tmp/blkh_exp94_phase0")
    args=p.parse_args()
    self_test()
    if args.self_test:
        return
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=analyze()
    payload=dict(experiment=94,phase="retrospective_negative_control",
        independent_holdout=False,novel_compression_result=False,
        eligible_for_public_demo=False,release_gate_passed=release_gate([]),
        overlapping_aggregate_group="all_21_pairs",
        independent_subgroups=0,
        method="exact original RGBA",rows=rows)
    (out/"results.json").write_text(json.dumps(payload,indent=2)+"\n")
    print("EXP94 PHASE-0 FINISHED: retrospective negative control; RELEASE HOLD")

if __name__=="__main__":
    main()
