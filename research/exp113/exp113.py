"""Exp113: four reversible spatial predictors, zlib/LZMA. RGB8 only."""
import struct,zlib,lzma
MAGIC=b'B113'
def pred(raw,i,x,y,w,mode):
 a=raw[i-3] if x else 0;b=raw[i-3*w] if y else 0;c=raw[i-3*w-3] if x and y else 0
 if mode==0:return a
 if mode==1:return b
 if mode==2:
  p=a+b-c;return min((a,b,c),key=lambda v:abs(p-v))
 return (a+b)//2
def encode(raw,w,h,mode=2,backend=0):
 if not 0<w*h<=10000000 or len(raw)!=w*h*3 or mode not in range(4) or backend not in (0,1):raise ValueError('metadata')
 r=bytearray(len(raw))
 for i in range(0,len(raw),3):
  x=(i//3)%w;y=(i//3)//w
  for c in range(3):r[i+c]=(raw[i+c]-pred(raw,i+c,x,y,w,mode))&255
 payload=zlib.compress(r,9) if backend==0 else lzma.compress(r,preset=6)
 return MAGIC+struct.pack('>IIBBI',w,h,mode,backend,len(payload))+payload
def decode(blob):
 if len(blob)<18 or blob[:4]!=MAGIC:raise ValueError('header')
 w,h,mode,backend,n=struct.unpack('>IIBBI',blob[4:18])
 if not 0<w*h<=10000000 or mode not in range(4) or backend not in (0,1) or len(blob)!=18+n:raise ValueError('metadata')
 if backend==0:
  d=zlib.decompressobj();r=d.decompress(blob[18:],w*h*3+1)
  if not d.eof or d.unused_data or d.unconsumed_tail:raise ValueError('zlib')
 else:
  d=lzma.LZMADecompressor();r=d.decompress(blob[18:],max_length=w*h*3+1)
  if not d.eof or d.unused_data:raise ValueError('lzma')
 if len(r)!=w*h*3:raise ValueError('length')
 raw=bytearray(len(r))
 for i in range(0,len(r),3):
  x=(i//3)%w;y=(i//3)//w
  for c in range(3):raw[i+c]=(r[i+c]+pred(raw,i+c,x,y,w,mode))&255
 return bytes(raw),w,h
