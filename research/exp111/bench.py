import io,json,platform,sys,time,resource,hashlib
import numpy as np
from PIL import Image,features
from skimage import data,transform
from exp111 import encode,decode,encode_mode
names=['astronaut','chelsea','coffee','rocket','hubble_deep_field','immunohistochemistry','colorwheel']
rows=[]
for name in names:
 a=np.asarray(getattr(data,name)())
 if a.ndim!=3 or a.shape[2]<3:continue
 a=a[:,:,:3];h,w=a.shape[:2]
 if max(w,h)>96:
  scale=96/max(w,h);a=transform.resize(a,(max(1,round(h*scale)),max(1,round(w*scale))),preserve_range=True,anti_aliasing=True).round().clip(0,255).astype('uint8')
 a=np.ascontiguousarray(a);h,w=a.shape[:2];raw=a.tobytes();img=Image.fromarray(a,'RGB')
 variants={};times={}
 for mode in range(2):
  for backend in range(2):
   t=time.perf_counter();blob=encode_mode(raw,w,h,mode,backend);dt=time.perf_counter()-t
   t=time.perf_counter();out,rw,rh=decode(blob);dd=time.perf_counter()-t
   assert out==raw and (rw,rh)==(w,h)
   key=f'{"global" if mode==0 else "channel"}_{"zlib" if backend==0 else "lzma"}'
   variants[key]=len(blob);times[key]=[round(dt,6),round(dd,6)]
 best=encode(raw,w,h);assert decode(best)==(raw,w,h)
 base={}
 for fmt,kwargs in [('PNG',dict(compress_level=9)),('WEBP',dict(lossless=True,quality=100,method=6))]:
  buf=io.BytesIO();img.save(buf,format=fmt,**kwargs);b=buf.getvalue();assert np.array_equal(np.asarray(Image.open(io.BytesIO(b)).convert('RGB')),a);base[fmt]=len(b)
 rows.append(dict(name=name,split='development' if name in names[:3] else 'holdout_exploratory',w=w,h=h,sha256=hashlib.sha256(raw).hexdigest(),variants=variants,baselines=base,best=len(best),selected=['global','channel'][best[12]],backend=['zlib','lzma'][best[13]],times_s=times,roundtrip=True))
print(json.dumps(dict(environment=dict(python=sys.version,platform=platform.platform(),pillow=Image.__version__,webp=features.check('webp'),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),rows=rows),indent=2))
