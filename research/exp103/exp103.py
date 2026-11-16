"""Exp103 research-only lossless grayscale predictor. Python stdlib."""
import argparse, hashlib, json, struct, time, zlib
from pathlib import Path
MAGIC=b'B103'
NAMES=('none','left','up','paeth')
def paeth(a,b,c):
 p=a+b-c
 return min((a,b,c),key=lambda v:(abs(p-v),(a,b,c).index(v)))
def transform(data,w,h,mode,inverse=False):
 out=bytearray(len(data))
 for y in range(h):
  for x in range(w):
   i=y*w+x
   source=out if inverse else data
   a=source[i-1] if x else 0
   b=source[i-w] if y else 0
   c=source[i-w-1] if x and y else 0
   pred=(0,a,b,paeth(a,b,c))[mode]
   out[i]=(data[i]+pred if inverse else data[i]-pred)&255
 return bytes(out)
def encode(raw,w,h):
 if w<1 or h<1 or w*h!=len(raw): raise ValueError('invalid dimensions')
 options=[(zlib.compress(transform(raw,w,h,m),9),m) for m in range(4)]
 payload,mode=min(options,key=lambda v:(len(v[0]),v[1]))
 return MAGIC+struct.pack('>IIB',w,h,mode)+payload
def decode(blob):
 if len(blob)<13 or blob[:4]!=MAGIC: raise ValueError('invalid header')
 w,h,mode=struct.unpack('>IIB',blob[4:13])
 if not (w and h and w*h<=100_000_000 and mode<4): raise ValueError('invalid metadata')
 obj=zlib.decompressobj()
 data=obj.decompress(blob[13:],w*h+1)
 if len(data)!=w*h or not obj.eof or obj.unused_data or obj.unconsumed_tail: raise ValueError('invalid payload')
 return transform(data,w,h,mode,True),w,h
def benchmark(raw,w,h):
 t=time.perf_counter(); blob=encode(raw,w,h); enc=time.perf_counter()-t
 t=time.perf_counter(); restored,rw,rh=decode(blob); dec=time.perf_counter()-t
 assert (restored,rw,rh)==(raw,w,h)
 return dict(raw_bytes=len(raw),prototype_bytes=len(blob),zlib_raw_bytes=len(zlib.compress(raw,9)),mode=NAMES[blob[12]],encode_s=enc,decode_s=dec,sha256=hashlib.sha256(raw).hexdigest(),roundtrip=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('raw',type=Path);p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True)
 a=p.parse_args();print(json.dumps(benchmark(a.raw.read_bytes(),a.width,a.height),indent=2))
