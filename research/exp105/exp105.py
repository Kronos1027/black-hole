"""Exp105 independent compressed stripes, research-only grayscale lossless."""
import struct,zlib
MAGIC=b'B105'
def paeth(a,b,c):
 p=a+b-c
 return min((a,b,c),key=lambda v:(abs(p-v),(a,b,c).index(v)))
def stripe(raw,w,start,end,mode):
 out=bytearray()
 for y in range(start,end):
  for x in range(w):
   i=y*w+x;a=raw[i-1] if x else 0;b=raw[i-w] if y else 0;c=raw[i-w-1] if y and x else 0
   out.append((raw[i]-(0,a,b,paeth(a,b,c))[mode])&255)
 return bytes(out)
def encode(raw,w,h,rows=16):
 if not (0<w and 0<h and w*h==len(raw) and w*h<=100000000 and 0<rows<=65535):raise ValueError('dimensions')
 result=bytearray(MAGIC+struct.pack('>IIH',w,h,rows))
 for start in range(0,h,rows):
  end=min(h,start+rows)
  payload,mode=min(((zlib.compress(stripe(raw,w,start,end,m),9),m) for m in range(4)),key=lambda a:(len(a[0]),a[1]))
  result.extend(struct.pack('>BI',mode,len(payload)));result.extend(payload)
 return bytes(result)
def decode(blob):
 if len(blob)<14 or blob[:4]!=MAGIC:raise ValueError('header')
 w,h,rows=struct.unpack('>IIH',blob[4:14]);n=w*h
 if not (w and h and rows and n<=100000000):raise ValueError('dimensions')
 out=bytearray(n);pos=14
 for start in range(0,h,rows):
  end=min(h,start+rows)
  if pos+5>len(blob):raise ValueError('truncated')
  mode,length=struct.unpack('>BI',blob[pos:pos+5]);pos+=5
  if mode>3 or length>len(blob)-pos:raise ValueError('metadata')
  obj=zlib.decompressobj();res=obj.decompress(blob[pos:pos+length],w*(end-start)+1);pos+=length
  if not obj.eof or obj.unused_data or obj.unconsumed_tail or len(res)!=w*(end-start):raise ValueError('payload')
  j=0
  for y in range(start,end):
   for x in range(w):
    i=y*w+x;a=out[i-1] if x else 0;b=out[i-w] if y else 0;c=out[i-w-1] if y and x else 0
    out[i]=(res[j]+(0,a,b,paeth(a,b,c))[mode])&255;j+=1
 if pos!=len(blob):raise ValueError('trailing data')
 return bytes(out),w,h
