"""
Experiment 55 — SINGLE globally coordinated SIREN on continuous REAL DEM.

Unlike Exp54's six small tiles and per-image conditioning, this experiment
fits ONE neural function f(x,y) to one 256x256 geographic patch and evaluates
a fixed coverage area at 64,128,256 px. Model bytes include reconstruction
min/max; WebP and physical-elevation predictors are real decoded baselines.

Source: public Mapzen/AWS Skadi, SRTM-derived HGT.
Example:
    python research/universe/experiment_55_contiguous_dem.py --self-test
    python research/universe/experiment_55_contiguous_dem.py --size-list 64,128,256 --seeds 42,123 --epochs 450
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

from experiment_51_fair_sharing import one_run, reconstruct
from experiment_54_srtm_real_meters import (
    read_hgt, evaluate_meters, baseline_webp, baseline_png16,
    baseline_predictive, baseline_raw_zlib, attach_artifact,
)

ORIGIN = (1200, 1300)
SOURCE_SIDE = 256


def contiguous_patch(raw: np.ndarray, pixels: int) -> np.ndarray:
    if pixels < 4 or SOURCE_SIDE % pixels != 0:
        raise ValueError("Size must divide 256 (at least 4 pixels)")
    r,c = ORIGIN
    data = np.asarray(raw[r:r+SOURCE_SIDE, c:c+SOURCE_SIDE])
    if data.shape != (SOURCE_SIDE, SOURCE_SIDE):
        raise ValueError("Invalid DEM crop")
    if (data == -32768).any():
        raise ValueError("NoData in original DEM source")
    size = SOURCE_SIDE // pixels
    # One spatial footprint at each resolution; area-mean aggregation,
    # then round elevation to integer meters to keep integer DEM baselines fair.
    averaged = data.astype(np.float64).reshape(
        pixels, size, pixels, size
    ).mean(axis=(1, 3))
    return np.rint(averaged).astype(np.float32)[None, :, :]


def run_one(
    original: np.ndarray, seed: int, width: int, depth: int,
    epochs: int, target: Path,
) -> dict:
    lo, hi = float(original.min()), float(original.max())
    if hi <= lo:
        raise ValueError("Constant elevation region")
    normalized = (original - lo) / (hi - lo)
    measurement = one_run(
        normalized, "individual", 1, width, depth,
        30.0, 4, epochs, 1e-3, seed, target,
    )
    payload = (target / measurement["encoded_file"]).read_bytes()
    decoded = reconstruct(payload) * (hi-lo) + lo
    saved = struct.pack("<ff", lo, hi) + payload
    return attach_artifact(
        target, f"exp55_single_h{width}_depth{depth}_seed{seed}.blkh55",
        original, saved, decoded,
        "global_siren", width=width, depth=depth, seed=seed,
        epochs=epochs, train_seconds=measurement["train_seconds"],
    )


def baselines(original: np.ndarray, out: Path) -> list[dict]:
    records=[]
    for q in (1,10,25,40,60,80,95):
        p,d=baseline_webp(original,q)
        records.append(attach_artifact(
            out, f"exp55_webp_q{q}.bin", original,p,d,"webp_8bit",quality=q,
        ))
    for step in (.5,1.,2.,5.,10.,20.,50.):
        p,d=baseline_predictive(original,step)
        records.append(attach_artifact(
            out,f"exp55_delta_{str(step).replace('.','p')}m.bin",
            original,p,d,"delta_zlib",step_m=step,
        ))
    for name,fn in (("png16_lossless",baseline_png16),
                    ("int16_zlib_lossless",baseline_raw_zlib)):
        p,d=fn(original)
        records.append(attach_artifact(
            out,f"exp55_{name}.bin",original,p,d,name,
        ))
    return records


def compare(reference: np.ndarray, neural: list[dict], classic: list[dict]) -> None:
    for model in neural:
        dominated=[
            {"method":p["method"],"bytes":p["bytes"],
             "rmse_m":p["rmse_m"],
             "quality":p.get("quality"),"step_m":p.get("step_m")}
            for p in classic
            if p["bytes"]<=model["bytes"]
            and p["rmse_m"]<=model["rmse_m"]
            and (p["bytes"]<model["bytes"] or p["rmse_m"]<model["rmse_m"])
        ]
        model["dominated_by_classical"]=dominated
        print(
            f"GLOBAL SIREN pixels={reference.shape[-1]} seed={model['seed']} "
            f"{model['bytes']} B rmse={model['rmse_m']:.3f}m "
            f"dominated={bool(dominated)}",
            flush=True,
        )
    for p in classic:
        if p["method"]=="webp_8bit":
            print(
                f"WEBP size={reference.shape[-1]} q={p['quality']} "
                f"{p['bytes']} B rmse={p['rmse_m']:.3f}m",flush=True
            )


def self_test() -> None:
    source=np.arange(SOURCE_SIDE**2,dtype=np.int16).reshape(
        SOURCE_SIDE,SOURCE_SIDE,
    )
    full=np.zeros((1600,1600),dtype=np.int16)
    r,c=ORIGIN
    full[r:r+SOURCE_SIDE,c:c+SOURCE_SIDE]=source
    for n in (64,128,256):
        ref=contiguous_patch(full,n)
        assert ref.shape==(1,n,n) and np.isfinite(ref).all()
        b,d=baseline_predictive(ref,1.)
        assert b and np.array_equal(d,ref)
        b,d=baseline_png16(ref)
        assert b and np.array_equal(d,ref)
    print("EXP55 SELF-TEST PASS: same footprint, decoded meter accuracy")


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--regions",default="N37W122,N37W119")
    p.add_argument("--size-list",default="64,128,256")
    p.add_argument("--epochs",type=int,default=350)
    p.add_argument("--seeds",default="42,123")
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--depth",type=int,default=1)
    p.add_argument("--cache",default="/tmp/blkh_exp54_srtm_cache")
    p.add_argument("--out",default="research/universe/_exp55_out")
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.epochs<1 or args.width<1 or args.depth<0:
        p.error("Invalid hyperparameters")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    sizes=[int(s) for s in args.size_list.split(",")]
    seeds=[int(s) for s in args.seeds.split(",")]
    report={"experiment":55,"status":"exploratory",
            "source":"Mapzen/AWS Skadi SRTM-derived",
            "objective":"single global-coordinate neural representation vs matched-error classic codecs",
            "geographic_origin_row_col":ORIGIN,
            "source_footprint_pixels":SOURCE_SIDE,
            "notes":[
                "Every resolution uses the same geographic footprint but aggregates samples.",
                "Quantization to integer meters after area-mean downsampling is explicit.",
                "This is in-sample encoding, not held-out generalization.",
                "All evaluated errors follow actual serialization and decoding.",
                "Coordinates/CRS are external common metadata and excluded equally.",
            ],
            "regions":[]}
    for tile in [r.strip() for r in args.regions.split(",")]:
        dem,provenance=read_hgt(tile,Path(args.cache))
        region={"provenance":provenance,"sizes":[]}
        for pixels in sizes:
            original=contiguous_patch(dem,pixels)
            dst=out/tile/str(pixels)
            dst.mkdir(parents=True,exist_ok=True)
            classics=baselines(original,dst)
            nets=[run_one(original,s,args.width,args.depth,args.epochs,dst) for s in seeds]
            compare(original,nets,classics)
            region["sizes"].append({
                "pixels":pixels,
                "source_sha256":hashlib.sha256(original.tobytes()).hexdigest(),
                "min_height_m":float(original.min()),
                "max_height_m":float(original.max()),
                "neural":nets,"classical":classics,
            })
            report["regions"][-1:]=report["regions"][-1:]
            (out/"results.json").write_text(json.dumps(
                {**report,"regions":report["regions"]+[region]},indent=2
            ),encoding="utf-8")
        report["regions"].append(region)
        (out/"results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"EXP55 DONE {out/'results.json'}")


if __name__=="__main__":
    main()
