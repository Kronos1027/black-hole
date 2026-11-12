"""
BLKH Experiment 70 — REAL shared conditional image decoder vs packed atlases.

A controlled, deliberately simple shared *linear* decoder, not SIREN:
one learned coordinate basis per font face, one int8 latent per glyph.
Entire decoder basis/mean/latents/scales + manifest is stored, and the actual
saved byte stream is decoded before PSNR and SSIM calculations. Compare to
PNG, WebP lossless, JPEG2000 lossless and lossy WebP quality ladder packed
into ONE atlas per face. This is a reproducible negative/positive control
before exploring nonlinear neural shared models.

See research/conditional/BHUH_EXP70_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import string
import struct
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features, __version__ as PIL_VERSION
from skimage.metrics import structural_similarity

FONT_NAMES = (
    "DejaVuSans.ttf",
    "DejaVuSerif.ttf",
    "Lato-Regular.ttf",
    "LiberationSans-Regular.ttf",
)
GLYPHS = string.ascii_uppercase + string.digits + "!?#%&()@+-_."
assert len(GLYPHS)==48
RANKS = (4, 8, 16, 32)
WEBP_QUALITIES = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
DIM = 32
GRID_COLUMNS = 8
GRID_ROWS = 6
SOURCE_SIZE = 48 * 32 * 32
WRAP = struct.Struct("<5sBI")
MAGIC = b"BL70C"
CODEC_MODEL = 1
CODEC_PNG = 2
CODEC_WEBP_LOSSLESS = 3
CODEC_JPEG2000_LOSSLESS = 4
MAX_META = 16384
MAX_WRAPPED = 1_000_000


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def font_path(name: str) -> Path:
    candidates = sorted(Path("/usr/share/fonts/truetype").rglob(name))
    if not candidates:
        raise FileNotFoundError(f"Required preregistered installed font missing: {name}")
    return candidates[0]


def render_face(path: Path) -> np.ndarray:
    font = ImageFont.truetype(str(path), 26)
    masks = []
    for letter in GLYPHS:
        image = Image.new("L", (DIM, DIM), 0)
        brush = ImageDraw.Draw(image)
        bounds = brush.textbbox((0,0),letter,font=font)
        x = (DIM - (bounds[2]-bounds[0]))//2-bounds[0]
        y = (DIM - (bounds[3]-bounds[1]))//2-bounds[1]
        brush.text((x,y),letter,font=font,fill=255)
        masks.append(np.asarray(image,dtype=np.uint8))
    return np.stack(masks,axis=0)


def pack_atlas(masks: np.ndarray) -> np.ndarray:
    if masks.shape!=(48,32,32) or masks.dtype!=np.uint8:
        raise ValueError("Invalid glyph corpus")
    atlas = masks.reshape(GRID_ROWS,GRID_COLUMNS,DIM,DIM)
    return atlas.transpose(0,2,1,3).reshape(GRID_ROWS*DIM,GRID_COLUMNS*DIM)


def unpack_atlas(atlas: np.ndarray) -> np.ndarray:
    if atlas.shape!=(GRID_ROWS*DIM,GRID_COLUMNS*DIM) or atlas.dtype!=np.uint8:
        raise ValueError("Invalid decompressed atlas")
    return atlas.reshape(GRID_ROWS,DIM,GRID_COLUMNS,DIM).transpose(
        0,2,1,3).reshape(48,DIM,DIM)


def wrapped(codec:int,manifest:dict,payload:bytes)->bytes:
    meta=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode("utf8")
    if not 0<len(meta)<MAX_META:
        raise ValueError("Invalid model/atlas metadata size")
    return WRAP.pack(MAGIC,codec,len(meta))+meta+payload


def unwrap(blob:bytes)->tuple[int,dict,bytes]:
    if len(blob)>MAX_WRAPPED or len(blob)<WRAP.size:
        raise ValueError("Incomplete/oversized BL70 file")
    magic,codec,meta_len=WRAP.unpack(blob[:WRAP.size])
    if magic!=MAGIC or not 0<meta_len<MAX_META:
        raise ValueError("Unsupported BL70 wrapper")
    if len(blob)<WRAP.size+meta_len:
        raise ValueError("Truncated manifest")
    manifest=json.loads(blob[WRAP.size:WRAP.size+meta_len])
    if (not isinstance(manifest,dict) or
        manifest.get("glyphs")!=GLYPHS or
        manifest.get("shape")!=[48,32,32] or
        manifest.get("atlas_size")!=[GRID_COLUMNS*DIM,GRID_ROWS*DIM]):
        raise ValueError("Invalid source atlas manifest")
    return codec,manifest,blob[WRAP.size+meta_len:]


def shared_linear_model(masks:np.ndarray,rank:int,manifest:dict)->bytes:
    if rank not in RANKS or masks.shape!=(48,32,32) or masks.dtype!=np.uint8:
        raise ValueError("Rank/data invalid")
    arr=masks.reshape(48,-1).astype(np.float32)
    mean=arr.mean(axis=0)
    centered=arr-mean
    u,s,vt=np.linalg.svd(centered,full_matrices=False)
    basis=vt[:rank].astype(np.float32)
    codes=(u[:,:rank]*s[:rank]).astype(np.float32)
    bs=np.maximum(np.max(np.abs(basis),axis=1),1e-10)/127
    cs=np.maximum(np.max(np.abs(codes),axis=0),1e-10)/127
    q_basis=np.rint(basis/bs[:,None]).clip(-127,127).astype(np.int8)
    q_codes=np.rint(codes/cs[None,:]).clip(-127,127).astype(np.int8)
    mean8=np.rint(mean).clip(0,255).astype(np.uint8)
    raw=(mean8.tobytes() + bs.astype("<f4").tobytes() +
         cs.astype("<f4").tobytes() + q_basis.tobytes() + q_codes.tobytes())
    coded=struct.pack("<H",rank)+zlib.compress(raw,level=9)
    return wrapped(CODEC_MODEL,manifest,coded)


def decode_model(blob:bytes)->tuple[np.ndarray,dict,int]:
    codec,manifest,body=unwrap(blob)
    if codec!=CODEC_MODEL or len(body)<3:
        raise ValueError("Not a conditioned BL70 linear model")
    rank=struct.unpack("<H",body[:2])[0]
    if rank not in RANKS:
        raise ValueError("Invalid shared basis rank")
    encoded=body[2:]
    # The experiment's serialized model is bounded and source-known.
    expected=1024+rank*4*2+rank*1024+48*rank
    dec=zlib.decompressobj()
    raw=dec.decompress(encoded,expected+1)
    if len(raw)!=expected or not dec.eof or dec.unused_data:
        raise ValueError("Incorrect decompressed linear basis bytes")
    pos=0
    avg=np.frombuffer(raw,dtype=np.uint8,count=1024,offset=pos).astype(np.float32)
    pos+=1024
    bs=np.frombuffer(raw,dtype="<f4",count=rank,offset=pos)
    pos+=4*rank
    cs=np.frombuffer(raw,dtype="<f4",count=rank,offset=pos)
    pos+=4*rank
    if not np.isfinite(bs).all() or not np.isfinite(cs).all():
        raise ValueError("Invalid quantization scale")
    qb=np.frombuffer(raw,dtype=np.int8,count=rank*1024,offset=pos).reshape(rank,1024)
    pos+=rank*1024
    qc=np.frombuffer(raw,dtype=np.int8,count=48*rank,offset=pos).reshape(48,rank)
    pos+=48*rank
    if pos!=expected:
        raise ValueError("Unexpected trailing model array bytes")
    restored=(qc.astype(np.float32)*cs.reshape(1,rank)) @ (
        qb.astype(np.float32)*bs.reshape(rank,1))
    restored=np.rint(restored+avg[None]).clip(0,255)
    return restored.astype(np.uint8).reshape(48,32,32),manifest,rank


def encode_atlas(codec:int,atlas:np.ndarray,manifest:dict,quality:int|None=None)->bytes:
    with io.BytesIO() as f:
        if codec==CODEC_PNG:
            Image.fromarray(atlas).save(f,format="PNG",optimize=True,compress_level=9)
        elif codec==CODEC_JPEG2000_LOSSLESS:
            Image.fromarray(atlas).save(f,format="JPEG2000",irreversible=False)
        elif codec==CODEC_WEBP_LOSSLESS:
            Image.fromarray(atlas).convert("RGB").save(
                f,format="WEBP",lossless=True,method=6)
        elif quality is not None and codec==10+quality:
            Image.fromarray(atlas).convert("RGB").save(
                f,format="WEBP",quality=quality,lossless=False,method=6)
        else:
            raise ValueError("Unknown atlas codec")
        return wrapped(codec,manifest,f.getvalue())


def decode_atlas(blob:bytes)->tuple[np.ndarray,dict]:
    codec,manifest,payload=unwrap(blob)
    if codec in (CODEC_MODEL,) or codec not in (
        CODEC_PNG,CODEC_JPEG2000_LOSSLESS,CODEC_WEBP_LOSSLESS
    ) and not 11<=codec<=110:
        raise ValueError("Not a baseline atlas")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        arr=np.asarray(image.convert("L"),dtype=np.uint8)
    return unpack_atlas(arr),manifest


def fidelity(source:np.ndarray,decoded:np.ndarray)->dict:
    if source.shape!=decoded.shape:
        raise ValueError("Incorrect reconstructed corpus geometry")
    delta=source.astype(np.float64)-decoded.astype(np.float64)
    mse=float(np.mean(delta**2))
    return {
        "mse":mse,
        "psnr_db":float(10*math.log10(255*255/mse)) if mse>0 else None,
        "exact":bool(np.array_equal(source,decoded)),
        "mean_glyph_ssim":float(np.mean([
            structural_similarity(source[i],decoded[i],data_range=255)
            for i in range(len(source))
        ])),
        "max_absolute_pixel_error":int(np.max(np.abs(delta))),
    }


def baseline_quality_match(reference:dict,probe:dict)->bool:
    return bool(probe["mse"]<=reference["mse"]+1e-9 and
                probe["mean_glyph_ssim"]+1e-9>=reference["mean_glyph_ssim"])


def exercise_font(name:str,out:Path)->dict:
    font=font_path(name)
    out.mkdir(parents=True,exist_ok=True)
    masks=render_face(font)
    atlas=pack_atlas(masks)
    Image.fromarray(atlas).save(out/"source_atlas.png")
    manifest={
        "font":name,"font_sha256":sha(font.read_bytes()),"glyphs":GLYPHS,
        "shape":[48,32,32],"atlas_size":[256,192],
        "pixel_interpretation":"8-bit grayscale antialiased glyph coverage",
        "source_atlas_sha256":sha(atlas.tobytes()),
    }
    baseline_configs=[
        ("png_lossless",CODEC_PNG,None),
        ("webp_lossless",CODEC_WEBP_LOSSLESS,None),
        ("jpeg2000_lossless",CODEC_JPEG2000_LOSSLESS,None),
    ]+[(f"webp_q{q:03}",10+q,q) for q in WEBP_QUALITIES]
    baselines={}
    for key,codec,q in baseline_configs:
        stream=encode_atlas(codec,atlas,manifest,quality=q)
        path=out/(key+".bl70")
        path.write_bytes(stream)
        dec,meta=decode_atlas(path.read_bytes())
        if meta!=manifest:
            raise AssertionError("Atlas manifest mismatch")
        scores=fidelity(masks,dec)
        if q is None and not scores["exact"]:
            raise AssertionError("Lossless atlas codec was not exact")
        baselines[key]={
            "file":path.name,"complete_file_bytes":path.stat().st_size,
            "file_sha256":sha(stream),"quality":scores,
        }
    ranks={}
    for rank in RANKS:
        stream=shared_linear_model(masks,rank,manifest)
        path=out/f"shared_linear_rank{rank}.bl70"
        path.write_bytes(stream)
        dec,meta,decoded_rank=decode_model(path.read_bytes())
        assert decoded_rank==rank and meta==manifest
        scores=fidelity(masks,dec)
        matched=[(b["complete_file_bytes"],key) for key,b in baselines.items()
                 if baseline_quality_match(scores,b["quality"])]
        if not matched:
            raise AssertionError("Lossless atlas must always match quality")
        size,key=min(matched)
        ranks[str(rank)]={
            "file":path.name,
            "complete_file_bytes":path.stat().st_size,
            "file_sha256":sha(stream),
            "quality":scores,
            "best_quality_matched_atlas":key,
            "best_quality_matched_atlas_bytes":size,
            "model_bytes_relative_to_best_atlas":path.stat().st_size/size,
            "wins_matched_baseline":bool(path.stat().st_size<size),
        }
        print(name,"rank",rank,"model_bytes",path.stat().st_size,
              "post_decode_psnr",scores["psnr_db"],
              "glyph_ssim",round(scores["mean_glyph_ssim"],5),
              "best_atlas",key,size,flush=True)
    return {
        "font_face":name,
        "font_sha256":manifest["font_sha256"],
        "source_manifest":manifest,
        "source_atlas_file":"source_atlas.png",
        "source_sha256":manifest["source_atlas_sha256"],
        "rasterized_images":48,
        "baseline_atlases":baselines,
        "shared_conditioned_linear_models":ranks,
    }


def self_test(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(70)
    glyph=rng.integers(0,256,size=(48,32,32),dtype=np.uint8)
    manifest={"font":"selftest","font_sha256":"0"*64,"glyphs":GLYPHS,
              "shape":[48,32,32],"atlas_size":[256,192],
              "source_atlas_sha256":sha(pack_atlas(glyph).tobytes())}
    assert np.array_equal(unpack_atlas(pack_atlas(glyph)),glyph)
    for rank in RANKS:
        b=shared_linear_model(glyph,rank,manifest)
        actual,m,k=decode_model(b)
        assert m==manifest and k==rank and actual.shape==glyph.shape
        try:
            decode_model(b[:-1])
        except (ValueError,zlib.error):
            pass
        else:
            raise AssertionError("Truncated conditioned model accepted")
    for c in (CODEC_PNG,CODEC_JPEG2000_LOSSLESS,CODEC_WEBP_LOSSLESS):
        coded=encode_atlas(c,pack_atlas(glyph),manifest)
        outdata,meta=decode_atlas(coded)
        assert meta==manifest and np.array_equal(outdata,glyph)
    print("EXP70 SELF-TEST PASS: serialized int8 shared model, fixed manifest and lossless packed atlas")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",default="/tmp/blkh_exp70")
    args=p.parse_args()
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    if args.self_test:
        self_test(root)
        return
    if not features.check("webp") or not features.check("jpg_2000"):
        raise RuntimeError("Need Pillow WebP and JPEG2000, no silent baseline exclusions")
    results=[]
    for font in FONT_NAMES:
        result=exercise_font(font,root/font.replace(".ttf",""))
        results.append(result)
        (root/"results.json").write_text(json.dumps({
            "experiment":70,
            "status":"provisional until CI independently decodes all bytes",
            "protocol":"research/conditional/BHUH_EXP70_PROTOCOL.md",
            "purpose":"shared conditional LINEAR representation vs quality-matched packed atlases",
            "ranks":list(RANKS),"font_faces":list(FONT_NAMES),
            "glyphs":GLYPHS,
            "runtime":{"python":platform.python_version(),
                       "numpy":np.__version__,"pillow":PIL_VERSION},
            "limitations":"same-corpus SVD is fitted, not holdout; related font glyphs; no SIREN/MLP; no independent art family",
            "results":results,
        },indent=2,allow_nan=False),encoding="utf8")
    print("EXP70 DONE: 4 font faces × 4 shared linear models, post-byte-decoded and atlas-matched",flush=True)


if __name__=="__main__":
    main()
