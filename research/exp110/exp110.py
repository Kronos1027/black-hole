"""Exp110: reversible byte layouts of RGB Paeth residual, stdlib+lzma."""
import struct,zlib,lzma
MAGIC=b'B110'
def paeth(a,b,c):
 p=a+b-c; ds=(abs(p-a),abs(p-b),abs(p-c));return (a,b,c)[ds.index(min(ds))]
def color(raw, inverse=False):
 out=bytearray(len(raw))
 for i in range(0,len(raw),3):
  r,g,b=raw[i:i+3]
  if inverse:out[i:i+3]=((g+r)&255,g,(g+b)&255)
  else:out[i:i+3]=((r-g)&255,g,(b-g)&255)
 return bytes(out)
def predict(raw,w,h,inverse=False):
 out=bytearray(len(raw));stride=w*3
 for y in range(h):
  for x in range(stride):
   i=y*stride+x; src=out if inverse else raw
   a=src[i-3] if x>=3 else 0;b=src[i-stride] if y else 0;c=src[i-stride-3] if y and x>=3 else 0
   v=paeth(a,b,c);out[i]=((raw[i]+v) if inverse else (raw[i]-v))&255
 return bytes(out)
def layout(raw,mode,inverse=False):
 if mode==0:return raw
 n=len(raw)//3
 if mode==1:
  if inverse:return bytes(v for triple in zip(raw[:n],raw[n:2*n],raw[2*n:]) for v in triple)
  return raw[0::3]+raw[1::3]+raw[2::3]
 if mode==2:
  if inverse:
   out=bytearray(len(raw))
   for bit in range(8):
    for i in range(len(raw)):
     j=bit*len(raw)+i;out[i]|=((raw[j//8]>>(7-j%8))&1)<<(7-bit)
   return bytes(out)
  out=bytearray(len(raw))
  for bit in range(8):
   for i,v in enumerate(raw):
    j=bit*len(raw)+i;out[j//8]|=((v>>(7-bit))&1)<<(7-j%8)
  return bytes(out)
 raise ValueError('layout')
def encode(raw,w,h):
 if w<1 or h<1 or w*h>100000000 or len(raw)!=3*w*h:raise ValueError('dimensions')
 residual=predict(color(raw),w,h)
 options=[]
 for mode in range(3):
  stream=layout(residual,mode)
  for backend in range(2):
   payload=zlib.compress(stream,9) if backend==0 else lzma.compress(stream,preset=6)
   options.append((len(payload),mode,backend,payload))
 _,mode,backend,payload=min(options)
 return MAGIC+struct.pack('>IIBB',w,h,mode,backend)+payload
def decode(blob):
 if len(blob)<14 or blob[:4]!=MAGIC:raise ValueError('header')
 w,h,mode,backend=struct.unpack('>IIBB',blob[4:14])
 if not(w and h and w*h<=100000000 and mode<3 and backend<2):raise ValueError('metadata')
 payload=blob[14:]
 if backend==0:
  d=zlib.decompressobj();data=d.decompress(payload,w*h*3+1)
  if not d.eof or d.unused_data or d.unconsumed_tail:raise ValueError('trailing or truncated')
 else:
  d=lzma.LZMADecompressor();data=d.decompress(payload,max_length=w*h*3+1)
  if not d.eof or d.unused_data:raise ValueError('trailing or truncated')
 if len(data)!=w*h*3:raise ValueError('length')
 return color(predict(layout(data,mode,True),w,h,True),True),w,h
