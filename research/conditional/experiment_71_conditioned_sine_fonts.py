"""
Experiment 71: genuinely nonlinear sine-coordinate conditioned shared image model.

Four previously seen font faces, three FIXED network sizes, all glyph latent
vectors counted. Training uses PyTorch CPU; serialization/decoding uses
NumPy ONLY with quantized float16 parameters. Reconstruct actual saved byte
streams before comparing PSNR+mean glyph SSIM against the same jointly packed
PNG/WebP/JPEG2000 atlas baselines used in Experiment 70.

This is not canonical SIREN initialization and not a new compression theorem.
Protocol: research/conditional/BHUH_EXP71_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import time
import zlib

import numpy as np
import torch
from torch import nn
from PIL import __version__ as PIL_VERSION

from experiment_70_shared_linear_font_atlas import (
    FONT_NAMES,WEBP_QUALITIES,GLYPHS,
    CODEC_PNG,CODEC_WEBP_LOSSLESS,CODEC_JPEG2000_LOSSLESS,
    WRAP,render_face,font_path,pack_atlas,encode_atlas,decode_atlas,
    sha,fidelity,baseline_quality_match,
)

MODEL_MAGIC=b"BL71N"
MODEL_CODEC=71
CONFIGS=((16,4),(32,8),(64,8))
STEPS=1600
BATCH=2048
LR=0.003
SEED=710071
OUTER_LIMIT=500_000
MAX_META=16384


class SineDecoder(nn.Module):
    def __init__(self,hidden:int,latent:int):
        super().__init__()
        self.embed=nn.Embedding(48,latent)
        self.l1=nn.Linear(latent+2,hidden)
        self.l2=nn.Linear(hidden,hidden)
        self.l3=nn.Linear(hidden,1)

    def forward(self,ids:torch.Tensor,xy:torch.Tensor)->torch.Tensor:
        x=torch.cat((xy,self.embed(ids)),dim=1)
        x=torch.sin(10*self.l1(x))
        x=torch.sin(10*self.l2(x))
        return torch.sigmoid(self.l3(x)).squeeze(-1)


def input_coords()->tuple[torch.Tensor,torch.Tensor]:
    i=torch.arange(48*32*32)
    pos=i % (32*32)
    ids=i//(32*32)
    xy=torch.stack(((pos%32).to(torch.float32)/15.5-1,
                   (pos//32).to(torch.float32)/15.5-1),axis=1)
    return ids,xy


def train(masks:np.ndarray,hidden:int,latent:int)->tuple[SineDecoder,float,float]:
    if masks.shape!=(48,32,32) or masks.dtype!=np.uint8:
        raise ValueError("Source pixels must be exactly 48 uint8 32px masks")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    net=SineDecoder(hidden,latent)
    opt=torch.optim.Adam(net.parameters(),lr=LR)
    ids,xy=input_coords()
    reference=torch.from_numpy(masks.flatten().copy()).to(torch.float32)/255
    begun=time.perf_counter()
    for _ in range(STEPS):
        batch=torch.randint(0,len(reference),(BATCH,))
        output=net(ids[batch],xy[batch])
        loss=(output-reference[batch]).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    duration=time.perf_counter()-begun
    return net,duration,float(loss.detach().item())


def model_arrays(net:SineDecoder)->list[np.ndarray]:
    return [
        net.embed.weight.detach().cpu().numpy(),
        net.l1.weight.detach().cpu().numpy(),
        net.l1.bias.detach().cpu().numpy(),
        net.l2.weight.detach().cpu().numpy(),
        net.l2.bias.detach().cpu().numpy(),
        net.l3.weight.detach().cpu().numpy(),
        net.l3.bias.detach().cpu().numpy(),
    ]


def serialize(net:SineDecoder,hidden:int,latent:int,manifest:dict)->bytes:
    if (hidden,latent) not in CONFIGS:
        raise ValueError("Model architecture is not preregistered")
    meta=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
    if not 0<len(meta)<MAX_META:
        raise ValueError("Bad source manifest")
    weights=b"".join(np.asarray(w,dtype="<f2").tobytes()
                     for w in model_arrays(net))
    body=struct.pack("<HH",hidden,latent)+zlib.compress(weights,9)
    return WRAP.pack(MODEL_MAGIC,MODEL_CODEC,len(meta))+meta+body


def decode_saved(stream:bytes)->tuple[np.ndarray,dict,tuple[int,int]]:
    if not WRAP.size<=len(stream)<=OUTER_LIMIT:
        raise ValueError("Untrusted model byte count")
    magic,codec,metadata_length=WRAP.unpack(stream[:WRAP.size])
    if (magic!=MODEL_MAGIC or codec!=MODEL_CODEC or
        not 0<metadata_length<MAX_META or
        len(stream)<WRAP.size+metadata_length+4):
        raise ValueError("Bad conditioned BL71 container")
    manifest=json.loads(stream[WRAP.size:WRAP.size+metadata_length])
    if (not isinstance(manifest,dict) or manifest.get("shape")!=[48,32,32]
            or manifest.get("glyphs")!=GLYPHS or
            manifest.get("atlas_size")!=[256,192]):
        raise ValueError("Invalid glyph metadata")
    body=stream[WRAP.size+metadata_length:]
    hidden,latent=struct.unpack("<HH",body[:4])
    if (hidden,latent) not in CONFIGS:
        raise ValueError("Unexpected non-linear decoder dimensions")
    shapes=[
        (48,latent),(hidden,latent+2),(hidden,),
        (hidden,hidden),(hidden,),(1,hidden),(1,),
    ]
    entries=sum(int(np.prod(x)) for x in shapes)
    dec=zlib.decompressobj()
    raw=dec.decompress(body[4:],entries*2+1)
    if len(raw)!=entries*2 or not dec.eof or dec.unused_data:
        raise ValueError("Corrupted/overlong float16 conditioned model")
    arrays=[]
    offset=0
    for shape in shapes:
        count=int(np.prod(shape))
        tensor=np.frombuffer(raw,dtype="<f2",count=count,offset=offset).reshape(shape)
        arrays.append(tensor.astype(np.float32))
        offset+=2*count
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Non-finite neural weights")
    emb,w1,b1,w2,b2,w3,b3=arrays
    image_rows=[]
    yy,xx=np.indices((32,32))
    xy=np.stack(((xx.ravel().astype(np.float32)/15.5-1),
                 (yy.ravel().astype(np.float32)/15.5-1)),axis=1)
    for i in range(48):
        conditioned=np.concatenate((
            xy,np.broadcast_to(emb[i],(1024,latent))
        ),axis=1)
        h1=np.sin(10*(conditioned@w1.T+b1))
        h2=np.sin(10*(h1@w2.T+b2))
        logit=np.clip(h2@w3.T+b3,-80,80)
        grayscale=1/(1+np.exp(-logit))
        image_rows.append(np.rint(grayscale.reshape(32,32)*255).clip(0,255).astype(np.uint8))
    return np.stack(image_rows),manifest,(hidden,latent)


def baselines_for(out:Path,masks:np.ndarray,manifest:dict)->dict:
    atlas=pack_atlas(masks)
    cfg=[
        ("png_lossless",CODEC_PNG,None),
        ("webp_lossless",CODEC_WEBP_LOSSLESS,None),
        ("jpeg2000_lossless",CODEC_JPEG2000_LOSSLESS,None),
    ]+[(f"webp_q{q:03}",10+q,q) for q in WEBP_QUALITIES]
    result={}
    for name,codec,q in cfg:
        coded=encode_atlas(codec,atlas,manifest,quality=q)
        path=out/(name+".bl70")
        path.write_bytes(coded)
        actual,restored_meta=decode_atlas(path.read_bytes())
        assert restored_meta==manifest
        scores=fidelity(masks,actual)
        if q is None and not scores["exact"]:
            raise AssertionError("Reference atlas should be lossless")
        result[name]={
            "file":path.name,"bytes":len(coded),"sha256":sha(coded),
            "quality":scores,
        }
    return result


def run_face(name:str,out:Path)->dict:
    font=font_path(name)
    masks=render_face(font)
    from PIL import Image
    out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(pack_atlas(masks)).save(out/"source_atlas.png")
    manifest={
        "font":name,"font_sha256":sha(font.read_bytes()),
        "glyphs":GLYPHS,"shape":[48,32,32],"atlas_size":[256,192],
        "pixel_interpretation":"8-bit grayscale antialiased glyph coverage",
        "source_atlas_sha256":sha(pack_atlas(masks).tobytes()),
    }
    baselines=baselines_for(out,masks,manifest)
    models={}
    for hidden,latent in CONFIGS:
        net,seconds,training_mse=train(masks,hidden,latent)
        stream=serialize(net,hidden,latent,manifest)
        path=out/f"nonlinear_h{hidden}_z{latent}.bl71"
        path.write_bytes(stream)
        actual,meta,arch=decode_saved(path.read_bytes())
        assert meta==manifest and arch==(hidden,latent)
        score=fidelity(masks,actual)
        matches=[(item["bytes"],key) for key,item in baselines.items()
                 if baseline_quality_match(score,item["quality"])]
        assert matches
        best,key=min(matches)
        useful=score["psnr_db"] is not None and score["psnr_db"]>=30.0 and score["mean_glyph_ssim"]>=0.90
        config=f"h{hidden}_z{latent}"
        models[config]={
            "file":path.name,"bytes":len(stream),"sha256":sha(stream),
            "hidden":hidden,"latent":latent,"parameter_count":sum(p.numel() for p in net.parameters()),
            "training_seconds":seconds,"last_training_minibatch_mse_0to1":training_mse,
            "quality_after_float16_decode":score,
            "smallest_equal_or_higher_quality_baseline":key,
            "baseline_matched_bytes":best,
            "size_win_at_matched_quality":len(stream)<best,
            "passes_useful_quality_gate":bool(useful),
            "release_candidate":bool(useful and len(stream)<best),
        }
        print(name,config,"model_bytes",len(stream),"matched_atlas",key,best,
              "PSNR",score["psnr_db"],"SSIM",round(score["mean_glyph_ssim"],5),
              "usable",useful,flush=True)
    return {
        "font_face":name,"manifest":manifest,
        "rasterized_count":len(masks),
        "source_atlas_file":"source_atlas.png",
        "baselines":baselines,"models":models,
    }


def self_test(out:Path)->None:
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    manifest={"font":"selftest","font_sha256":"0"*64,
              "glyphs":GLYPHS,"shape":[48,32,32],"atlas_size":[256,192]}
    for hidden,latent in CONFIGS:
        net=SineDecoder(hidden,latent)
        b=serialize(net,hidden,latent,manifest)
        restored,restored_meta,size=decode_saved(b)
        assert restored_meta==manifest and size==(hidden,latent)
        assert restored.shape==(48,32,32) and restored.dtype==np.uint8
        try:
            decode_saved(b[:-1])
        except (ValueError,zlib.error):
            pass
        else:
            raise AssertionError("Incomplete model accepted")
    print("EXP71 SELF-TEST PASS: quantized nonlinear sine network, latent decode and corruption rejection")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="/tmp/blkh_exp71")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    root=Path(a.out)
    root.mkdir(parents=True,exist_ok=True)
    if a.self_test:
        self_test(root);return
    results=[]
    for name in FONT_NAMES:
        r=run_face(name,root/name.replace(".ttf",""))
        results.append(r)
        (root/"results.json").write_text(json.dumps({
            "experiment":71,
            "status":"provisional until independent CI decoding",
            "protocol":"research/conditional/BHUH_EXP71_PROTOCOL.md",
            "model":"two sine-activated shared layers, per-glyph latent, sigmoid output",
            "configs":[list(v) for v in CONFIGS],
            "steps":STEPS,"batch":BATCH,"learning_rate":LR,"seed":SEED,
            "runtime":{"python":platform.python_version(),"torch":torch.__version__,
                       "numpy":np.__version__,"pillow":PIL_VERSION},
            "font_faces":list(FONT_NAMES),
            "limitations":"4 familiar font faces; fitted to same corpus, not held out; CPU single host, not canonical SIREN initialization",
            "results":results,
        },indent=2,allow_nan=False),encoding="utf-8")
    print("EXP71 DONE: 4 related source faces ×3 genuine non-linear conditioned sine models",flush=True)


if __name__=="__main__":
    main()
