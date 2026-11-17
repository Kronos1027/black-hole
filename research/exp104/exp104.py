"""Exp104: reversible row-adaptive predictor for 8-bit grayscale, research only."""
import struct,zlib
MAGIC=b'B104'
def paeth(a,b,c):
 p=a+b-c;d=(abs(p-a),abs(p-b),abs(p-c));return (a,b,c)[d.index(min(d))]
def residual_row(raw,w,y,mode):
 o=bytearray(w);offset=y*w
 for x in range(w):
  i=offset+x;a=raw[i-1] if x else 0;b=raw[i-w] if y else 0;c=raw[i-w-1] if x and y else 0
  pred=(0,a,b,paeth(a,b,c))[mode];o[x]=(raw[i]-pred)&255
 return bytes(o)
def encode(raw,w,h):
 if not (w>0 and h>0 and w*h==len(raw) and w*h<=100000000):raise ValueError('dimensions')
 modes=bytearray((h+3)//4);res=bytearray()
 for y in range(h):
  options=[residual_row(raw,w,y,m) for m in range(4)]
  mode=min(range(4),key=lambda m:(sum(min(v,256-v) for v in options[m]),m))
  modes[y//4]|=mode<<(2*(y%4));res.extend(options[mode])
 return MAGIC+struct.pack('>II',w,h)+bytes(modes)+zlib.compress(res,9)
def decode(blob):
 if len(blob)<12 or blob[:4]!=MAGIC:raise ValueError('header')
 w,h=struct.unpack('>II',blob[4:12]);n=w*h
 if not (w and h and n<=100000000):raise ValueError('dimensions')
 k=(h+3)//4
 if len(blob)<12+k:raise ValueError('mode table')
 modes=blob[12:12+k]
 if h%4 and modes[-1]>>(2*(h%4)):raise ValueError('padding bits')
 obj=zlib.decompressobj();data=obj.decompress(blob[12+k:],n+1)
 if len(data)!=n or not obj.eof or obj.unused_data or obj.unconsumed_tail:raise ValueError('payload')
 out=bytearray(n)
 for y in range(h):
  mode=(modes[y//4]>>(2*(y%4)))&3
  for x in range(w):
   i=y*w+x;a=out[i-1] if x else 0;b=out[i-w] if y else 0;c=out[i-w-1] if x and y else 0
   out[i]=(data[i]+(0,a,b,paeth(a,b,c))[mode])&255
 return bytes(out),w,h
