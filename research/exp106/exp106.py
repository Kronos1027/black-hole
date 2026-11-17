"""Exp106: reversible RGB decorrelation and optional Paeth per plane. Research only."""
import struct,zlib
MAGIC=b'B106'
def paeth(a,b,c):
 p=a+b-c;d=(abs(p-a),abs(p-b),abs(p-c));return (a,b,c)[d.index(min(d))]
def planes(raw,w,h,mode):
 out=[bytearray(w*h) for _ in range(3)]
 for i in range(w*h):
  r,g,b=raw[3*i:3*i+3]
  v=(r,g,b) if mode==0 else ((g,(r-g)&255,(b-g)&255) if mode==1 else (r,(g-r)&255,(b-g)&255))
  for k in range(3):out[k][i]=v[k]
 return out
def unplanes(ps,mode):
 out=bytearray(3*len(ps[0]))
 for i in range(len(ps[0])):
  a,b,c=(p[i] for p in ps)
  r,g,bl=(a,b,c) if mode==0 else (((a+b)&255,a,(a+c)&255) if mode==1 else (a,(a+b)&255,(a+b+c)&255))
  out[3*i:3*i+3]=bytes((r,g,bl))
 return bytes(out)
def filt(raw,w):
 o=bytearray(len(raw))
 for i,v in enumerate(raw):
  x=i%w;y=i//w;a=raw[i-1] if x else 0;b=raw[i-w] if y else 0;c=raw[i-w-1] if x and y else 0
  o[i]=(v-paeth(a,b,c))&255
 return bytes(o)
def unfilt(raw,w):
 o=bytearray(len(raw))
 for i,v in enumerate(raw):
  x=i%w;y=i//w;a=o[i-1] if x else 0;b=o[i-w] if y else 0;c=o[i-w-1] if x and y else 0
  o[i]=(v+paeth(a,b,c))&255
 return bytes(o)
def encode(raw,w,h):
 if not (0<w and 0<h and w*h<=100000000 and len(raw)==3*w*h):raise ValueError('dimensions')
 opts=[]
 for mode in range(3):
  ps=planes(raw,w,h,mode)
  for pred in range(2):
   payload=zlib.compress(b''.join(filt(p,w) if pred else p for p in ps),9)
   opts.append((len(payload),mode,pred,payload))
 _,mode,pred,payload=min(opts)
 return MAGIC+struct.pack('>IIBB',w,h,mode,pred)+payload
def decode(blob):
 if len(blob)<14 or blob[:4]!=MAGIC:raise ValueError('header')
 w,h,mode,pred=struct.unpack('>IIBB',blob[4:14]);n=w*h
 if not (w and h and n<=100000000 and mode<3 and pred<2):raise ValueError('metadata')
 obj=zlib.decompressobj();raw=obj.decompress(blob[14:],3*n+1)
 if len(raw)!=3*n or not obj.eof or obj.unused_data or obj.unconsumed_tail:raise ValueError('payload')
 ps=[raw[k*n:(k+1)*n] for k in range(3)]
 if pred:ps=[unfilt(p,w) for p in ps]
 return unplanes(ps,mode),w,h
