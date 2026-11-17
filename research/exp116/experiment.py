"""Exp116: deterministic palette spatial XOR ablation; Python 3, numpy, Pillow."""
import io, json, lzma, struct, time, zlib, platform, resource
import numpy as np
from PIL import Image, features

def sample(family, seed, n=128):
 r=np.random.default_rng(seed)
 if family=="noise": return r.integers(0,256,(n,n,3),dtype=np.uint8)
 k=16 if family=="palette" else 8
 p=r.integers(0,256,(k,3),dtype=np.uint8)
 if family=="palette": idx=r.integers(0,k,(n,n))
 elif family=="blocks": idx=np.repeat(np.repeat(r.integers(0,k,(n//8,n//8)),8,0),8,1)
 else: idx=np.tile((np.arange(n)//8)%k,(n,1))
 return p[idx]

def encode(a, mode, ent):
 h,w,_=a.shape
 pal,ids=np.unique(a.reshape(-1,3),axis=0,return_inverse=True)
 if len(pal)>256:return None
 ids=ids.astype(np.uint8).reshape(h,w)
 if mode=="left": ids=np.bitwise_xor(ids,np.concatenate([np.zeros((h,1),dtype=np.uint8),ids[:,:-1]],axis=1))
 if mode=="up": ids=np.bitwise_xor(ids,np.concatenate([np.zeros((1,w),dtype=np.uint8),ids[:-1,:]],axis=0))
 body=zlib.compress(ids.tobytes(),9) if ent=="zlib" else lzma.compress(ids.tobytes(),preset=6)
 return b"B116"+struct.pack("<HHHBB",w,h,len(pal),["raw","left","up"].index(mode),["zlib","lzma"].index(ent))+pal.tobytes()+body

def decode(b):
 if len(b)<12 or b[:4]!=b"B116":raise ValueError("header")
 w,h,n,m,e=struct.unpack("<HHHBB",b[4:12])
 if not(w and h and 1<=n<=256 and m<3 and e<2):raise ValueError("fields")
 start=12+3*n
 if len(b)<start:raise ValueError("palette")
 pal=np.frombuffer(b[12:start],dtype=np.uint8).reshape(n,3)
 raw=zlib.decompress(b[start:]) if e==0 else lzma.decompress(b[start:])
 if len(raw)!=w*h:raise ValueError("length")
 idx=np.frombuffer(raw,dtype=np.uint8).reshape(h,w).copy()
 if m==1:
  for x in range(1,w):idx[:,x]^=idx[:,x-1]
 if m==2:
  for y in range(1,h):idx[y,:]^=idx[y-1,:]
 if np.any(idx>=n):raise ValueError("index")
 return pal[idx]

def run():
 rows=[]
 for split,seeds in [("dev",[101,102,103]),("test",[9101,9102,9103])]:
  for family in ["blocks","palette","stripes","noise"]:
   for seed in seeds:
    a=sample(family,seed)
    buf=io.BytesIO();Image.fromarray(a).save(buf,format="WEBP",lossless=True,method=6)
    assert np.array_equal(a,np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
    options=[]
    for mode in ["raw","left","up"]:
     for ent in ["zlib","lzma"]:
      t=time.perf_counter(); b=encode(a,mode,ent); enc_ms=(time.perf_counter()-t)*1000
      if b is None:continue
      t=time.perf_counter();recon=decode(b);dec_ms=(time.perf_counter()-t)*1000
      assert np.array_equal(a,recon)
      options.append(dict(mode=mode,entropy=ent,bytes=len(b),encode_ms=round(enc_ms,3),decode_ms=round(dec_ms,3)))
    rows.append(dict(split=split,family=family,seed=seed,webp_bytes=len(buf.getvalue()),oracle=min(options,key=lambda x:x["bytes"]) if options else None,options=options))
 for split in ["dev","test"]:
  for family in ["blocks","palette","stripes","noise"]:
   rr=[r for r in rows if r["split"]==split and r["family"]==family]
   print(split,family,"BLKH",round(np.mean([r["oracle"]["bytes"] for r in rr])) if all(r["oracle"] for r in rr) else "N/A","WebP",round(np.mean([r["webp_bytes"] for r in rr])),"wins",sum(bool(r["oracle"]) and r["oracle"]["bytes"]<r["webp_bytes"] for r in rr))
 print("roundtrips",sum(len(r["options"]) for r in rows),"rss_kib",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 print(json.dumps(dict(environment=dict(python=platform.python_version(),numpy=np.__version__,pillow=Image.__version__,libwebp=features.version("webp")),rows=rows),indent=2))
if __name__=="__main__":run()
