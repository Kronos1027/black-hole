"""Experimental lossless RGB context split; stdlib only."""
import struct,zlib,lzma
MAGIC=b'B112'
def paeth(a,b,c):
 p=a+b-c
 return min((a,b,c),key=lambda v:abs(p-v))
def encode_mode(raw,w,h,contexts=2,backend='zlib'):
 if w<=0 or h<=0 or w*h>10000000 or len(raw)!=3*w*h or contexts not in (1,2,4):raise ValueError('metadata')
 streams=[bytearray() for _ in range(contexts)];stride=w*3
 for y in range(h):
  for x in range(w):
   for ch in range(3):
    i=y*stride+x*3+ch
    a=raw[i-3] if x else 0;b=raw[i-stride] if y else 0;c=raw[i-stride-3] if x and y else 0
    ctx=min(contexts-1,abs(a-b)*contexts//256)
    streams[ctx].append((raw[i]-paeth(a,b,c))&255)
 compressor=(lambda s:zlib.compress(s,9)) if backend=='zlib' else (lambda s:lzma.compress(s,preset=6))
 parts=[compressor(s) for s in streams];flag=0 if backend=='zlib' else 1
 return MAGIC+struct.pack('>IIBB',w,h,contexts,flag)+b''.join(struct.pack('>I',len(p)) for p in parts)+b''.join(parts)
def decode(blob):
 if len(blob)<18 or blob[:4]!=MAGIC:raise ValueError('header')
 w,h,contexts,flag=struct.unpack('>IIBB',blob[4:14])
 if w<=0 or h<=0 or w*h>10000000 or contexts not in (1,2,4) or flag>1 or len(blob)<14+4*contexts:raise ValueError('metadata')
 sizes=struct.unpack('>'+('I'*contexts),blob[14:14+4*contexts]);off=14+4*contexts
 if sum(sizes)!=len(blob)-off:raise ValueError('payload length')
 streams=[]
 for size in sizes:
  payload=blob[off:off+size];off+=size
  if flag==0:
   d=zlib.decompressobj();v=d.decompress(payload,3*w*h+1)
   if not d.eof or d.unused_data or d.unconsumed_tail:raise ValueError('zlib')
  else:
   d=lzma.LZMADecompressor();v=d.decompress(payload,max_length=3*w*h+1)
   if not d.eof or d.unused_data:raise ValueError('lzma')
  streams.append(v)
 pos=[0]*contexts;raw=bytearray(3*w*h);stride=w*3
 for y in range(h):
  for x in range(w):
   for ch in range(3):
    i=y*stride+x*3+ch
    a=raw[i-3] if x else 0;b=raw[i-stride] if y else 0;c=raw[i-stride-3] if x and y else 0
    ctx=min(contexts-1,abs(a-b)*contexts//256)
    if pos[ctx]>=len(streams[ctx]):raise ValueError('short stream')
    raw[i]=(streams[ctx][pos[ctx]]+paeth(a,b,c))&255;pos[ctx]+=1
 if any(pos[i]!=len(streams[i]) for i in range(contexts)):raise ValueError('trailing residuals')
 return bytes(raw),w,h
