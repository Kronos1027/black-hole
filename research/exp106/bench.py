import io,time,json,hashlib,platform,resource
from PIL import Image
from skimage import data
import numpy as np
from exp106 import encode,decode
names=['astronaut','chelsea','coffee','rocket','hubble_deep_field']
results=[]
for name in names:
 img=np.asarray(getattr(data,name)())
 if img.ndim==2:img=np.stack([img]*3,axis=-1)
 if img.shape[2]==4:img=img[:,:,:3]
 img=Image.fromarray(img.astype('uint8'),'RGB');img.thumbnail((128,128));w,h=img.size;raw=img.tobytes()
 t=time.perf_counter();blob=encode(raw,w,h);et=time.perf_counter()-t
 t=time.perf_counter();rr,rw,rh=decode(blob);dt=time.perf_counter()-t
 assert (rr,rw,rh)==(raw,w,h)
 baselines={}
 for fmt,kwargs in [('PNG',{'compress_level':9}),('WEBP',{'lossless':True,'method':6})]:
  f=io.BytesIO();t=time.perf_counter();img.save(f,format=fmt,**kwargs);elapsed=time.perf_counter()-t
  f.seek(0);assert Image.open(f).convert('RGB').tobytes()==raw
  baselines[fmt.lower()]={'bytes':len(f.getvalue()),'encode_s':elapsed}
 results.append(dict(name=name,split='dev' if name in names[:2] else 'exploratory_holdout',width=w,height=h,prototype_bytes=len(blob),mode=blob[12],predictor=blob[13],encode_s=et,decode_s=dt,sha256=hashlib.sha256(raw).hexdigest(),roundtrip=True,baselines=baselines))
print(json.dumps({'python':platform.python_version(),'results':results,'maxrss_kb_process':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
