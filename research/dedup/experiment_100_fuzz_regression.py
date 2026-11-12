"""Fuzz regression for Exp100's actual full lossless residual encoder/decoder.

120 synthetic adversarial *correctness* fixtures, 360 real full archives.
NOT real-world compression evidence; use Exp102 for source-independent data.
"""
from __future__ import annotations
import argparse
import numpy as np
from experiment_78_real_cc0_rgba_sprites import encode_webp
import experiment_83_pixelorama_git_history as base
import experiment_92_motion_shift_patch as motion
import experiment_100_residual_transforms as codec

def self_test(trials=120):
    rng=np.random.default_rng(100100)
    tested=0
    for trial in range(trials):
        oh=int(rng.integers(1,44));ow=int(rng.integers(1,44))
        h=int(rng.integers(1,44));w=int(rng.integers(1,44))
        dx=int(rng.integers(-8,9));dy=int(rng.integers(-8,9))
        original=rng.integers(0,256,(oh,ow,4),dtype=np.uint8)
        changed=motion.predicted(original,(h,w,4),dx,dy)
        if trial%3==0:
            changed[:]=rng.integers(0,256,(h,w,4),dtype=np.uint8)
        else:
            for _ in range(int(rng.integers(1,12))):
                y=int(rng.integers(0,h));x=int(rng.integers(0,w))
                changed[y,x]=rng.integers(0,256,(4,),dtype=np.uint8)
        changed[0,0]=[255,37,59,0]
        items=[]
        for role,rgba in (("old",original),("new",changed)):
            items.append(dict(rgba=rgba,raw=b"synthetic-only",meta=dict(
                width=rgba.shape[1],height=rgba.shape[0],
                role=role,filename=f"fuzz:{trial}:{role}",
                source_png_sha256="f"*64,
                pixel_rgba_sha256=base.sha(rgba.tobytes()))))
        atlas,positions=base.pack(items)
        old_atlas,oldpos=base.pack(items[::2])
        manifest=dict(atlas_shape=list(atlas.shape),
            atlas_sha256=base.sha(atlas.tobytes()),source_sprites=positions,
            historical_pair_count=1,old_atlas_shape=list(old_atlas.shape),
            old_atlas_positions=oldpos,
            old_atlas_rgba_sha256=base.sha(old_atlas.tobytes()))
        pred=motion.predicted(original,changed.shape,dx,dy)
        count=int(np.count_nonzero(np.any(pred!=changed,axis=2)))
        shift=[(dx,dy,count)]
        old_encoded=encode_webp(old_atlas)
        for mode in codec.TRANSFORMS:
            archive,_=codec.encode(items,manifest,shift,mode,old_encoded)
            restored,meta=codec.decode(archive)
            assert meta=={**manifest,"codec":codec.MODE}
            assert len(restored)==2
            assert np.array_equal(restored[0],original)
            assert np.array_equal(restored[1],changed)
            try:codec.decode(archive[:-1])
            except Exception:pass
            else:raise AssertionError("Truncated full archive not rejected")
            tested+=1
    assert tested==3*trials
    print("EXP100 FUZZ PASS:",trials,"synthetic correctness fixtures;",
          tested,"full archive exact RGBA round trips including hidden alpha RGB",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--trials",type=int,default=120)
    args=p.parse_args()
    if not 1<=args.trials<=500:
        p.error("trials must be between 1 and 500")
    self_test(args.trials)
