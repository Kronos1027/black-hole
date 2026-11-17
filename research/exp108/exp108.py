"""Exp108: lossless RGB8 row filters; research prototype (stdlib)."""
import struct,zlib
MAGIC=b'B108'
def paeth(a,b,c):
 p=a+b-c;x=abs(p-a);y=abs(p-b);z=abs(p-c)
 return a if x<=y and x<=z else b if y<=z else c
def color(raw,mode,inverse=False):
 if mode==0:return raw
 out=bytearray(len(raw))
 for i in range(0,len(raw),3):
  r,g,b=raw[i:i+3]
  if not inverse:out[i:i+3]=(g,(r-g)&255,(b-g)&255)
  else:out[i:i+3]=((r+g)&255,r,(r+b)&255)
 return bytes(out)
def rowfilter(raw,w,h,adaptive):
 stride=w*3;prev=bytes(stride);out=bytearray()
 for y in range(h):
  row=raw[y*stride:(y+1)*stride];candidates=[]
  for mode in ((0,1,2,3,4) if adaptive else (4,)):
   residual=bytearray(stride)
   for i,v in enumerate(row):
    a=row[i-3] if i>=3 else 0;b=prev[i];c=prev[i-3] if i>=3 else 0
    p=(0,a,b,(a+b)//2,paeth(a,b,c))[mode]
    residual[i]=(v-p)&255
   score=sum(min(x,256-x) for x in residual)
   candidates.append((score,mode,bytes(residual)))
  _,mode,part=min(candidates);out.append(mode);out.extend(part);prev=row
 return bytes(out)
def unfilter(stream,w,h):
 stride=w*3;prev=bytes(stride);raw=bytearray();pos=0
 for _ in range(h):
  mode=stream[pos];pos+=1
  if mode>4:raise ValueError('invalid filter')
  res=stream[pos:pos+stride];pos+=stride;row=bytearray(stride)
  for i,v in enumerate(res):
   a=row[i-3] if i>=3 else 0;b=prev[i];c=prev[i-3] if i>=3 else 0
   p=(0,a,b,(a+b)//2,paeth(a,b,c))[mode]
   row[i]=(v+p)&255
  raw.extend(row);prev=row
 return bytes(raw)
def encode(raw,w,h,adaptive=True,decorrelate=True):
 if w<1 or h<1 or len(raw)!=w*h*3:raise ValueError('dimensions')
 candidates=[]
 for cm in ((0,1) if decorrelate else (0,)):
  filtered=rowfilter(color(raw,cm),w,h,adaptive);payload=zlib.compress(filtered,9)
  candidates.append((len(payload),cm,payload))
 _,cm,payload=min(candidates)
 return MAGIC+struct.pack('>IIB',w,h,cm)+payload
def decode(blob):
 if len(blob)<13 or blob[:4]!=MAGIC:raise ValueError('header')
 w,h,cm=struct.unpack('>IIB',blob[4:13])
 if not (0<w and 0<h and w*h<=100_000_000 and cm<=1):raise ValueError('metadata')
 expected=h*(1+3*w);obj=zlib.decompressobj()
 data=obj.decompress(blob[13:],expected+1)
 if len(data)!=expected or not obj.eof or obj.unused_data or obj.unconsumed_tail:raise ValueError('payload')
 return color(unfilter(data,w,h),cm,True),w,h
