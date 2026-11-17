import json,random,time,io,hashlib,resource,platform
from PIL import Image
from exp113 import encode,decode
def image(kind,seed,w=96,h=96):
 rng=random.Random(seed);b=bytearray()
 for y in range(h):
  for x in range(w):
   if kind=='gradient':p=((x*2+y+seed)%256,(x+y*2)%256,(x+y)%256)
   elif kind=='blocks':p=((x//12*23+y//12*7)%256,(x//16*11)%256,(y//16*17)%256)
   elif kind=='noise':p=(rng.randrange(256),rng.randrange(256),rng.randrange(256))
   else:p=tuple((x*3+y*2+rng.randrange(13))%256 for _ in range(3))
   b.extend(p)
 return bytes(b)
rows=[]
for split,seeds in [('development',[1,2]),('holdout',[991,992])]:
 for kind in ['gradient','blocks','noise','texture']:
  for seed in seeds:
   raw=image(kind,seed);img=Image.frombytes('RGB',(96,96),raw);base={}
   for fmt,opts in [('PNG',dict(optimize=True)),('WEBP',dict(lossless=True,quality=100,method=6))]:
    out=io.BytesIO();img.save(out,format=fmt,**opts);data=out.getvalue()
    assert Image.open(io.BytesIO(data)).convert('RGB').tobytes()==raw
    base[fmt]=len(data)
   variants=[]
   for mode in range(4):
    for backend in (0,1):
     t=time.perf_counter();blob=encode(raw,96,96,mode,backend);et=time.perf_counter()-t
     t=time.perf_counter();decoded,w,h=decode(blob);dt=time.perf_counter()-t
     assert decoded==raw and (w,h)==(96,96)
     variants.append(dict(mode=mode,backend=backend,bytes=len(blob),encode_s=et,decode_s=dt))
   rows.append(dict(split=split,kind=kind,seed=seed,sha256=hashlib.sha256(raw).hexdigest(),baselines=base,variants=variants))
print(json.dumps(dict(environment=dict(python=platform.python_version(),pillow=Image.__version__,platform=platform.platform(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),rows=rows),indent=2))
