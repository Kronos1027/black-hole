import io,json,time,resource,hashlib,sys
from PIL import Image
from skimage import data
from exp109 import encode,decode
names=['astronaut','chelsea','coffee','rocket','hubble_deep_field','immunohistochemistry','colorwheel']
rows=[]
for name in names:
 im=Image.fromarray(getattr(data,name)()).convert('RGB');im.thumbnail((96,96));w,h=im.size;raw=im.tobytes()
 t=time.perf_counter();blob=encode(raw,w,h);et=time.perf_counter()-t
 t=time.perf_counter();restored,rw,rh=decode(blob);dt=time.perf_counter()-t
 assert (restored,rw,rh)==(raw,w,h)
 baselines={}
 for fmt,kw in [('PNG',{'compress_level':9}),('WEBP',{'lossless':True,'quality':100,'method':6})]:
  buf=io.BytesIO();im.save(buf,format=fmt,**kw);b=buf.getvalue()
  assert Image.open(io.BytesIO(b)).convert('RGB').tobytes()==raw
  baselines[fmt]=len(b)
 rows.append(dict(name=name,width=w,height=h,exp109=len(blob),backend=['zlib','brotli','lzma'][blob[13]],color=blob[12],encode_s=round(et,4),decode_s=round(dt,4),sha256=hashlib.sha256(raw).hexdigest(),**baselines))
print(json.dumps(dict(python=sys.version,pillow=Image.__version__,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rows=rows),indent=2))
