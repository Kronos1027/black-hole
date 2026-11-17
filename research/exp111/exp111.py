"""Exp111 reversible RGB residual streams; research prototype."""
import struct,zlib,lzma
MAGIC=b'B111'
def paeth(a,b,c):
 p=a+b-c;ds=(abs(p-a),abs(p-b),abs(p-c));return (a,b,c)[ds.index(min(ds))]
def residual(raw,w,h,inverse=False):
 out=bytearray(len(raw));stride=w*3
 for y in range(h):
  for x in range(stride):
   i=y*stride+x;src=out if inverse else raw
   a=src[i-3] if x>=3 else 0;b=src[i-stride] if y else 0;c=src[i-stride-3] if y and x>=3 else 0
   v=paeth(a,b,c);out[i]=(raw[i]+v if inverse else raw[i]-v)&255
 return bytes(out)
def color(raw,inverse=False):
 out=bytearray(len(raw))
 for i in range(0,len(raw),3):
  r,g,b=raw[i:i+3]
  out[i:i+3]=(((r+g) if inverse else (r-g))&255,g,((b+g) if inverse else (b-g))&255)
 return bytes(out)
def encode_mode(raw,w,h,mode,backend):
 if w<1 or h<1 or w*h>10000000 or len(raw)!=3*w*h:raise ValueError('dimensions')
 data=residual(color(raw),w,h)
 streams=[data] if mode==0 else [data[i::3] for i in range(3)]
 compressor=lambda x:zlib.compress(x,9) if backend==0 else lzma.compress(x,preset=6)
 compressed=[compressor(s) for s in streams]
 return MAGIC+struct.pack('>IIBB',w,h,mode,backend)+b''.join(struct.pack('>I',len(s)) for s in compressed)+b''.join(compressed)
def encode(raw,w,h):
 return min((encode_mode(raw,w,h,m,b) for m in range(2) for b in range(2)),key=lambda x:(len(x),x[12:14]))
def decode(blob):
 if len(blob)<18 or blob[:4]!=MAGIC:raise ValueError('header')
 w,h,mode,backend=struct.unpack('>IIBB',blob[4:14]);n=1 if mode==0 else 3
 if not(w and h and w*h<=10000000 and mode<2 and backend<2 and len(blob)>=14+4*n):raise ValueError('metadata')
 sizes=struct.unpack('>'+('I'*n),blob[14:14+4*n]);off=14+4*n
 if sum(sizes)!=len(blob)-off:raise ValueError('length')
 streams=[]
 for size in sizes:
  payload=blob[off:off+size];off+=size
  if backend==0:
   d=zlib.decompressobj();v=d.decompress(payload,w*h*3+1)
   if not d.eof or d.unused_data or d.unconsumed_tail:raise ValueError('zlib invalid')
  else:
   d=lzma.LZMADecompressor();v=d.decompress(payload,max_length=w*h*3+1)
   if not d.eof or d.unused_data:raise ValueError('lzma invalid')
  if len(v)!=(3*w*h if mode==0 else w*h):raise ValueError('stream length')
  streams.append(v)
 raw=streams[0] if mode==0 else bytes(v for triple in zip(*streams) for v in triple)
 return color(residual(raw,w,h,True),True),w,h
