"""Exp109 lossless RGB8 Paeth + selectable entropy backend; research prototype."""
import struct,zlib,lzma,brotli
MAGIC=b'B109'
def paeth(a,b,c):
 p=a+b-c;ds=(abs(p-a),abs(p-b),abs(p-c));return (a,b,c)[ds.index(min(ds))]
def color(raw,mode,inv=False):
 if not mode:return raw
 out=bytearray(len(raw))
 for i in range(0,len(raw),3):
  r,g,b=raw[i:i+3]
  if inv:out[i:i+3]=((r+g)&255,r,(r+b)&255)
  else:out[i:i+3]=(g,(r-g)&255,(b-g)&255)
 return bytes(out)
def filt(raw,w,h,inv=False):
 out=bytearray(len(raw));stride=w*3
 for y in range(h):
  for x in range(stride):
   i=y*stride+x;src=out if inv else raw
   a=src[i-3] if x>=3 else 0;b=src[i-stride] if y else 0;c=src[i-stride-3] if y and x>=3 else 0
   p=paeth(a,b,c);out[i]=((raw[i]+p) if inv else (raw[i]-p))&255
 return bytes(out)
def pack(data,backend):
 return (zlib.compress(data,9),brotli.compress(data,quality=9),lzma.compress(data,preset=6))[backend]
def unpack(data,backend):
 return (lambda:zlib.decompress(data),lambda:brotli.decompress(data),lambda:lzma.decompress(data))[backend]()
def encode(raw,w,h,backends=(0,1,2)):
 if w<1 or h<1 or w*h>100_000_000 or len(raw)!=w*h*3:raise ValueError('dimensions')
 candidates=[]
 for cm in (0,1):
  residual=filt(color(raw,cm),w,h)
  for backend in backends:
   payload=pack(residual,backend);candidates.append((len(payload),cm,backend,payload))
 _,cm,backend,payload=min(candidates)
 return MAGIC+struct.pack('>IIBB',w,h,cm,backend)+payload
def decode(blob):
 if len(blob)<14 or blob[:4]!=MAGIC:raise ValueError('header')
 w,h,cm,backend=struct.unpack('>IIBB',blob[4:14])
 if not(w and h and w*h<=100_000_000 and cm<=1 and backend<=2):raise ValueError('metadata')
 raw=unpack(blob[14:],backend)
 if len(raw)!=w*h*3:raise ValueError('length')
 return color(filt(raw,w,h,True),cm,True),w,h
