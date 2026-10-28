"""
BLKH Experiment 72 — independently sourced RGB photo families, no data tuning.

See BHUH_EXP72_PROTOCOL.md. A real nonlinear sine-coordinate decoder shared
across 16 RGB tiles of each independently sourced public photograph, with
one learned latent vector per tile. All network parameters + latents are
serialized float16 + zlib and decoded using NumPy only. Compare actual
post-file RGB PSNR AND patch-mean SSIM to a single full-size RGB source atlas
saved as WebP, PNG, JPEG, and lossless JPEG2000.

One photo = ONE independent source; adjacent tiles are not independent.
This is a benchmark, NOT a new neural compression result.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import struct
import time
import zlib

import numpy as np
import skimage
from skimage import data
from skimage.metrics import structural_similarity
import torch
from torch import nn
from PIL import Image, __version__ as PIL_VERSION, features

SOURCES=("astronaut","coffee","chelsea","rocket")
HOLDOUT="rocket"
CONFIGS=((32,8),(64,8),(96,12))
STEP_COUNT=1500
BATCH_SIZE=2048
LR=0.003
SEED=720072
TILE=32
SIDE=128
MAGIC=b"BL72N"
BASE_MAGIC=b"BL72A"
HEADER=struct.Struct("<5sBI")
MODEL_CODEC=1
WEBP_Q=(1,5,10,20,30,40,50,60,70,80,90,95,100)
JPEG_Q=(5,20,40,60,80,95)
META_LIMIT=16384
STREAM_LIMIT=2_000_000

def digest(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()

def source_photo(name:str)->tuple[np.ndarray,np.ndarray,dict]:
    if name not in SOURCES:
        raise ValueError("Photo source outside fixed corpus")
    photo=np.asarray(getattr(data,name)())
    if photo.ndim!=3 or photo.shape[2]!=3 or photo.dtype!=np.uint8:
        raise ValueError("Source must be RGB uint8")
    h,w,_=photo.shape
    if h<SIDE or w<SIDE:
        raise ValueError("Source image smaller than declared crop")
    y=(h-SIDE)//2
    x=(w-SIDE)//2
    crop=np.ascontiguousarray(photo[y:y+SIDE,x:x+SIDE])
    blocks=np.stack([crop[yy:yy+TILE,xx:xx+TILE]
                     for yy in range(0,SIDE,TILE)
                     for xx in range(0,SIDE,TILE)])
    assert blocks.shape==(16,32,32,3)
    return crop,blocks,{
        "photo":name,"original_shape":list(photo.shape),
        "original_sha256":digest(photo.tobytes()),
        "crop_yx":[y,x],
        "atlas_shape":[128,128,3],
        "atlas_sha256":digest(crop.tobytes()),
        "block_shape":[16,32,32,3],
        "tiling":"16 nonoverlapping patches in original 4x4 spatial order",
        "color":"RGB uint8 sRGB interpreted",
        "holdout":name==HOLDOUT,
    }

def crop_to_blocks(arr:np.ndarray)->np.ndarray:
    if arr.shape!=(128,128,3) or arr.dtype!=np.uint8:
        raise ValueError("Bad decoded RGB atlas")
    return np.stack([arr[y:y+32,x:x+32]
                     for y in range(0,128,32) for x in range(0,128,32)])

def wrap(magic:bytes,codec:int,manifest:dict,payload:bytes)->bytes:
    meta=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
    if not (0<len(meta)<META_LIMIT):
        raise ValueError("Bad manifest length")
    return HEADER.pack(magic,codec,len(meta))+meta+payload

def unwrap(blob:bytes,magic_expected:bytes)->tuple[int,dict,bytes]:
    if not (HEADER.size<len(blob)<=STREAM_LIMIT):
        raise ValueError("Invalid file length")
    magic,codec,n=HEADER.unpack(blob[:HEADER.size])
    if magic!=magic_expected or not 0<n<META_LIMIT:
        raise ValueError("Unsupported file schema")
    if len(blob)<HEADER.size+n:
        raise ValueError("Truncated photo manifest")
    meta=json.loads(blob[HEADER.size:HEADER.size+n])
    if (not isinstance(meta,dict)
        or meta.get("block_shape")!=[16,32,32,3]
        or meta.get("atlas_shape")!=[128,128,3]
        or meta.get("photo") not in SOURCES):
        raise ValueError("Invalid source photo metadata")
    return codec,meta,blob[HEADER.size+n:]

class SineRGB(nn.Module):
    def __init__(self,hidden:int,latent:int):
        super().__init__()
        self.latent=nn.Embedding(16,latent)
        self.a=nn.Linear(2+latent,hidden)
        self.b=nn.Linear(hidden,hidden)
        self.c=nn.Linear(hidden,3)
    def forward(self,ids:torch.Tensor,xy:torch.Tensor)->torch.Tensor:
        z=torch.cat((xy,self.latent(ids)),dim=1)
        z=torch.sin(10*self.a(z))
        z=torch.sin(10*self.b(z))
        return torch.sigmoid(self.c(z))

def sample_grid():
    n=16*32*32
    idx=torch.arange(n)
    spatial=idx%(32*32)
    xy=torch.stack((spatial%32,spatial//32),dim=1).float()/15.5-1
    return idx//(32*32),xy

def train(blocks:np.ndarray,h:int,z:int):
    if (h,z) not in CONFIGS or blocks.shape!=(16,32,32,3):
        raise ValueError("Model/data not fixed")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    net=SineRGB(h,z)
    opt=torch.optim.Adam(net.parameters(),lr=LR)
    ids,xy=sample_grid()
    truth=torch.from_numpy(blocks.reshape(-1,3).copy()).float()/255
    start=time.perf_counter()
    for _ in range(STEP_COUNT):
        choice=torch.randint(0,len(truth),(BATCH_SIZE,))
        pred=net(ids[choice],xy[choice])
        loss=(pred-truth[choice]).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return net,time.perf_counter()-start,float(loss.detach())

def model_weights(net:SineRGB):
    return [
        net.latent.weight.detach().cpu().numpy(),
        net.a.weight.detach().cpu().numpy(),
        net.a.bias.detach().cpu().numpy(),
        net.b.weight.detach().cpu().numpy(),
        net.b.bias.detach().cpu().numpy(),
        net.c.weight.detach().cpu().numpy(),
        net.c.bias.detach().cpu().numpy(),
    ]

def encode_model(net:SineRGB,h:int,z:int,manifest:dict)->bytes:
    if (h,z) not in CONFIGS:
        raise ValueError("Unregistered network shape")
    raw=b"".join(np.asarray(w,dtype="<f2").tobytes() for w in model_weights(net))
    return wrap(MAGIC,MODEL_CODEC,manifest,struct.pack("<HH",h,z)+zlib.compress(raw,9))

def decode_model(blob:bytes)->tuple[np.ndarray,dict,tuple[int,int]]:
    codec,manifest,body=unwrap(blob,MAGIC)
    if codec!=MODEL_CODEC or len(body)<4:
        raise ValueError("Not a shared RGB model")
    h,z=struct.unpack("<HH",body[:4])
    if (h,z) not in CONFIGS:
        raise ValueError("Unregistered architecture in stream")
    shapes=[(16,z),(h,2+z),(h,),(h,h),(h,),(3,h),(3,)]
    elems=sum(int(np.prod(d)) for d in shapes)
    dec=zlib.decompressobj()
    raw=dec.decompress(body[4:],elems*2+1)
    if len(raw)!=elems*2 or not dec.eof or dec.unused_data:
        raise ValueError("Incomplete float16 model or trailing stream")
    tensors=[]
    off=0
    for shape in shapes:
        count=int(np.prod(shape))
        t=np.frombuffer(raw,dtype="<f2",count=count,offset=off).reshape(shape)
        tensors.append(t.astype(np.float32))
        off+=count*2
    if not all(np.isfinite(t).all() for t in tensors):
        raise ValueError("Nonfinite model weights")
    emb,w1,b1,w2,b2,w3,b3=tensors
    yy,xx=np.indices((32,32))
    xy=np.stack((xx.reshape(-1)/15.5-1,yy.reshape(-1)/15.5-1),
                axis=1).astype(np.float32)
    recon=[]
    for i in range(16):
        features=np.concatenate(
            (xy,np.broadcast_to(emb[i],(1024,z))),axis=1)
        a=np.sin(10*(features@w1.T+b1))
        b=np.sin(10*(a@w2.T+b2))
        logit=np.clip(b@w3.T+b3,-80,80)
        colors=np.rint(255/(1+np.exp(-logit))).clip(0,255).astype(np.uint8)
        recon.append(colors.reshape(32,32,3))
    return np.stack(recon),manifest,(h,z)

def encode_baseline(kind:str,atlas:np.ndarray,manifest:dict,q:int|None)->bytes:
    with io.BytesIO() as f:
        im=Image.fromarray(atlas,"RGB")
        if kind=="png":
            im.save(f,format="PNG",optimize=True,compress_level=9)
        elif kind=="webp_lossless":
            im.save(f,format="WEBP",lossless=True,method=6)
        elif kind=="j2k_lossless":
            im.save(f,format="JPEG2000",irreversible=False)
        elif kind=="webp" and q in WEBP_Q:
            im.save(f,format="WEBP",quality=q,method=6,lossless=False)
        elif kind=="jpeg" and q in JPEG_Q:
            im.save(f,format="JPEG",quality=q,subsampling=0,optimize=True)
        else:
            raise ValueError("Unknown baseline method")
        code={"png":2,"webp_lossless":3,"j2k_lossless":4,
              "webp":5,"jpeg":6}[kind]
        return wrap(BASE_MAGIC,code,manifest,f.getvalue())

def decode_baseline(raw:bytes)->tuple[np.ndarray,dict]:
    codec,manifest,blob=unwrap(raw,BASE_MAGIC)
    if codec not in (2,3,4,5,6):
        raise ValueError("Unknown atlas method")
    with Image.open(io.BytesIO(blob)) as im:
        im.load()
        rgb=np.asarray(im.convert("RGB"),dtype=np.uint8)
    return crop_to_blocks(rgb),manifest

def fidelity(truth:np.ndarray,decoded:np.ndarray)->dict:
    if truth.shape!=decoded.shape or decoded.dtype!=np.uint8:
        raise ValueError("Incorrect output image shape")
    diff=truth.astype(np.float64)-decoded.astype(np.float64)
    mse=float(np.mean(diff*diff))
    ssim=float(np.mean([
        structural_similarity(truth[i],decoded[i],channel_axis=2,data_range=255)
        for i in range(16)
    ]))
    return {
        "mse":mse,"psnr_db":float(10*np.log10(255**2/mse)) if mse>0 else None,
        "ssim_mean_rgb_tiles":ssim,"lossless":bool(np.array_equal(truth,decoded)),
        "maximum_channel_error":int(np.max(np.abs(diff))),
    }

def quality_meets(baseline:dict,net:dict)->bool:
    return (baseline["mse"]<=net["mse"]+1e-10 and
            baseline["ssim_mean_rgb_tiles"]+1e-10>=net["ssim_mean_rgb_tiles"])

def run_photo(name:str,root:Path)->dict:
    atlas,blocks,meta=source_photo(name)
    directory=root/name
    directory.mkdir(parents=True,exist_ok=True)
    Image.fromarray(atlas,"RGB").save(directory/"original_central_crop.png")
    baselines={}
    codecs=[("png",None),("webp_lossless",None),("j2k_lossless",None)]
    codecs += [("webp",q) for q in WEBP_Q]
    codecs += [("jpeg",q) for q in JPEG_Q]
    for kind,q in codecs:
        label=kind+(f"_q{q:03}" if q is not None else "")
        b=encode_baseline(kind,atlas,meta,q)
        p=directory/(label+".bl72")
        p.write_bytes(b)
        restored,manifest=decode_baseline(p.read_bytes())
        if manifest!=meta:
            raise AssertionError("Atlas metadata mismatch")
        score=fidelity(blocks,restored)
        if q is None and not score["lossless"]:
            raise AssertionError(f"{label} not lossless")
        baselines[label]={
            "bytes":len(b),"sha256":digest(b),"file":p.name,
            "decoded_quality":score,
        }
    models={}
    for h,z in CONFIGS:
        network,duration,final_minibatch=train(blocks,h,z)
        output=encode_model(network,h,z,meta)
        name_model=f"sine_h{h}_z{z}"
        p=directory/(name_model+".bl72")
        p.write_bytes(output)
        result,manifest,shape=decode_model(p.read_bytes())
        assert manifest==meta and shape==(h,z)
        q=fidelity(blocks,result)
        eligible=[(b["bytes"],key) for key,b in baselines.items()
                  if quality_meets(b["decoded_quality"],q)]
        best,format_name=min(eligible)
        passes=(q["psnr_db"] is not None and q["psnr_db"]>=30
                and q["ssim_mean_rgb_tiles"]>=0.90)
        models[name_model]={
            "file":p.name,"sha256":digest(output),"complete_bytes":len(output),
            "hidden":h,"latent":z,"all_parameters":sum(p.numel() for p in network.parameters()),
            "training_seconds":duration,"training_last_minibatch_mse":final_minibatch,
            "decoded_quality":q,
            "best_matched_baseline":format_name,
            "best_matched_baseline_bytes":best,
            "smaller_than_matched_baseline":bool(len(output)<best),
            "passes_usable_gate":bool(passes),
            "release_candidate":bool(passes and len(output)<best),
        }
        print(name,name_model,"bytes",len(output),"baseline",format_name,best,
              "psnr",round(q["psnr_db"],3),"ssim",
              round(q["ssim_mean_rgb_tiles"],4),"usable",passes,flush=True)
    return {
        "source":name,"source_manifest":meta,"independent_original_photo":True,
        "source_image_file":"original_central_crop.png",
        "baseline_atlases":baselines,"conditional_networks":models,
    }

def self_test(path:Path):
    path.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    placeholder={
        "photo":"astronaut","original_shape":[512,512,3],
        "original_sha256":"0"*64,"crop_yx":[0,0],
        "atlas_shape":[128,128,3],"atlas_sha256":"0"*64,
        "block_shape":[16,32,32,3]}
    source=np.arange(128*128*3,dtype=np.uint32).astype(np.uint8).reshape(128,128,3)
    assert np.array_equal(crop_to_blocks(source)[15],source[-32:,-32:])
    for h,z in CONFIGS:
        model=SineRGB(h,z)
        encoded=encode_model(model,h,z,placeholder)
        reconstructed,restored,configuration=decode_model(encoded)
        assert reconstructed.shape==(16,32,32,3) and configuration==(h,z)
        assert restored==placeholder
        try:
            decode_model(encoded[:-1])
        except (ValueError,zlib.error):
            pass
        else:
            raise AssertionError("Truncated neural archive accepted")
    for kind,q in (("png",None),("webp_lossless",None),("j2k_lossless",None)):
        packed=encode_baseline(kind,source,placeholder,q)
        decoded,manifest=decode_baseline(packed)
        assert manifest==placeholder and np.array_equal(decoded,crop_to_blocks(source))
    print("EXP72 SELF TEST PASS: independent NumPy neural decoder and lossless RGB atlas")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="/tmp/blkh_exp72")
    p.add_argument("--self-test",action="store_true")
    options=p.parse_args()
    root=Path(options.out)
    root.mkdir(parents=True,exist_ok=True)
    if options.self_test:
        self_test(root)
        return
    if not features.check("webp") or not features.check("jpg_2000"):
        raise RuntimeError("WebP and reversible J2K mandatory comparators")
    observations=[]
    for photo in SOURCES:
        row=run_photo(photo,root)
        observations.append(row)
        (root/"results.json").write_text(json.dumps({
            "experiment":72,"status":"provisional until CI audit",
            "protocol":"research/conditional/BHUH_EXP72_PROTOCOL.md",
            "dataset":"4 independently sourced natural-color public photos, central 128px RGB crop per photo",
            "sources":list(SOURCES),"holdout_source":HOLDOUT,
            "models":[list(x) for x in CONFIGS],
            "step_count":STEP_COUNT,"batch_size":BATCH_SIZE,
            "learning_rate":LR,"seed":SEED,
            "software":{"python":platform.python_version(),
                        "numpy":np.__version__,"torch":torch.__version__,
                        "pillow":PIL_VERSION,"scikit_image":skimage.__version__},
            "limits":"4 originals, 16 spatially adjacent tiles per photo are correlated; matched RGB PSNR and patch mean SSIM",
            "results":observations,
        },indent=2,allow_nan=False),encoding="utf-8")
    assert len(observations)==4
    print("EXP72 DONE: 4 real source photographs, 12 serialized shared RGB models, 22 packed RGB baseline families per source",flush=True)

if __name__=="__main__":
    main()
