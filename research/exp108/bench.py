import hashlib,json,platform,time,resource,io
from PIL import Image,features
from skimage import data
from exp108 import encode,decode
names=['astronaut','chelsea','coffee','rocket','hubble_deep_field','immunohistochemistry','colorwheel']
records=[]
for name in names:
 im=Image.fromarray(getattr(data,name)()).convert('RGB');im.thumbnail((96,96));w,h=im.size;raw=im.tobytes()
 png=io.BytesIO();im.save(png,format='PNG',compress_level=9)
 webp=io.BytesIO();im.save(webp,format='WEBP',lossless=True,quality=100,method=6)
 assert Image.open(io.BytesIO(png.getvalue())).convert('RGB').tobytes()==raw
 assert Image.open(io.BytesIO(webp.getvalue())).convert('RGB').tobytes()==raw
 rec={'name':name,'partition':'development' if name in names[:2] else 'exploratory_same_library','width':w,'height':h,'png':len(png.getvalue()),'webp_lossless':len(webp.getvalue()),'variants':{}}
 for adaptive in (False,True):
  for decorrelate in (False,True):
   key=f'adaptive{int(adaptive)}_color{int(decorrelate)}';t=time.perf_counter();blob=encode(raw,w,h,adaptive,decorrelate);et=time.perf_counter()-t
   t=time.perf_counter();restored,rw,rh=decode(blob);dt=time.perf_counter()-t
   assert (restored,rw,rh)==(raw,w,h)
   rec['variants'][key]={'bytes':len(blob),'encode_seconds':round(et,6),'decode_seconds':round(dt,6),'sha256':hashlib.sha256(restored).hexdigest()}
 records.append(rec)
print(json.dumps({'python':platform.python_version(),'pillow':Image.__version__,'webp_support':features.check('webp'),'peak_rss_kb_process':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'images':records},indent=2))
