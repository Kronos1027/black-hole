"""Exp119 standalone operational selector benchmark. python experiment.py > results.json"""
import io,json,time,zlib,lzma,platform,resource
import numpy as np
from PIL import Image
FAMILIES=('random','stripes','diagonal','blocks','runs')
MODES=('raw','xorleft','subleft','subup','paeth')
CONFIGS=[(m,b) for m in MODES for b in ('zlib','lzma')]
def image(seed,family,n=128):
 r=np.random.default_rng(seed);pal=r.integers(0,256,(16,3),dtype=np.uint8);y,x=np.indices((n,n))
 if family=='random':a=r.integers(0,16,(n,n),dtype=np.uint8)
 elif family=='stripes':a=((x//8+seed)%16).astype('uint8')
 elif family=='diagonal':a=((x+y+seed)//8%16).astype('uint8')
 elif family=='blocks':a=r.integers(0,16,(n//8,n//8),dtype=np.uint8).repeat(8,0).repeat(8,1)
 else:a=np.broadcast_to(r.integers(0,16,n,dtype=np.uint8),(n,n)).copy()
 return a,pal
def encode(a,pal,mode,backend):
 left=np.zeros_like(a);left[:,1:]=a[:,:-1]
 up=np.zeros_like(a);up[1:]=a[:-1]
 ul=np.zeros_like(a);ul[1:,1:]=a[:-1,:-1]
 if mode=='raw':res=a
 elif mode=='xorleft':res=a^left
 else:
  if mode=='subleft':p=left
  elif mode=='subup':p=up
  else:
   q=left.astype('int16')+up.astype('int16')-ul
   dl=abs(q-left);du=abs(q-up);dul=abs(q-ul)
   p=np.where((dl<=du)&(dl<=dul),left,np.where(du<=dul,up,ul))
  res=(a.astype('int16')-p)%256
 raw=res.astype('uint8').tobytes()
 payload=zlib.compress(raw,9) if backend=='zlib' else lzma.compress(raw,preset=6)
 return b'B119'+bytes([MODES.index(mode),backend=='lzma',16])+a.shape[0].to_bytes(2,'big')+a.shape[1].to_bytes(2,'big')+pal.tobytes()+payload
def decode(blob):
 assert blob[:4]==b'B119'
 mode=MODES[blob[4]];backend=blob[5];n=int.from_bytes(blob[7:9],'big');m=int.from_bytes(blob[9:11],'big')
 pal=np.frombuffer(blob[11:59],dtype='uint8').reshape(16,3)
 raw=lzma.decompress(blob[59:]) if backend else zlib.decompress(blob[59:])
 r=np.frombuffer(raw,dtype='uint8').reshape(n,m);a=np.zeros_like(r)
 for y in range(n):
  for x in range(m):
   l=int(a[y,x-1]) if x else 0;u=int(a[y-1,x]) if y else 0;ul=int(a[y-1,x-1]) if x and y else 0
   if mode=='raw':v=int(r[y,x])
   elif mode=='xorleft':v=int(r[y,x])^l
   else:
    p=l if mode=='subleft' else u
    if mode=='paeth':
     q=l+u-ul;p=l if abs(q-l)<=abs(q-u) and abs(q-l)<=abs(q-ul) else u if abs(q-u)<=abs(q-ul) else ul
    v=int(r[y,x])+p
   a[y,x]=v%256
 return a,pal
def webp(rgb):
 b=io.BytesIO();Image.fromarray(rgb,'RGB').save(b,format='WEBP',lossless=True,method=6)
 raw=b.getvalue();assert np.array_equal(np.asarray(Image.open(io.BytesIO(raw)).convert('RGB')),rgb)
 return len(raw)
def main():
 dev=[]
 for f in FAMILIES:
  for s in (101,102,103):
   a,p=image(s,f);dev.append([len(encode(a,p,*c)) for c in CONFIGS])
 global_i=min(range(10),key=lambda i:(sum(d[i] for d in dev),i))
 results=[]
 for f in FAMILIES:
  for s in (9101,9102,9103):
   a,p=image(s,f);w=webp(p[a])
   t=time.perf_counter();global_blob=encode(a,p,*CONFIGS[global_i]);global_ms=(time.perf_counter()-t)*1000
   t=time.perf_counter();blobs=[encode(a,p,*c) for c in CONFIGS];selection_ms=(time.perf_counter()-t)*1000
   i=min(range(10),key=lambda j:(len(blobs[j]),j));selected=blobs[i]
   t=time.perf_counter()
   for blob in (global_blob,selected):
    aa,pp=decode(blob);assert np.array_equal(aa,a) and np.array_equal(pp,p)
   dec_ms=(time.perf_counter()-t)*1000
   results.append(dict(family=f,seed=s,webp=w,global_bytes=len(global_blob),selected_bytes=len(selected),selected_mode='/'.join(CONFIGS[i]),global_encode_ms=round(global_ms,3),selection_encode_ms=round(selection_ms,3),decode_both_ms=round(dec_ms,3)))
 return dict(environment=dict(python=platform.python_version(),numpy=np.__version__,pillow=Image.__version__,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),global_mode='/'.join(CONFIGS[global_i]),dev_samples=len(dev),test_samples=len(results),results=results)
if __name__=='__main__':print(json.dumps(main(),indent=2))
