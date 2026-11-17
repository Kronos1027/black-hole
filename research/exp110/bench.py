import io,json,time,resource,sys,platform,zlib,lzma
from PIL import Image
from skimage import data
from exp110 import encode,decode,color,predict,layout
names=['astronaut','chelsea','coffee','rocket','hubble_deep_field','immunohistochemistry','colorwheel']
rows=[]
for name in names:
 im=Image.fromarray(getattr(data,name)()).convert('RGB');im.thumbnail((96,96));w,h=im.size;raw=im.tobytes()
 t=time.perf_counter();blob=encode(raw,w,h);enc=time.perf_counter()-t
 t=time.perf_counter();recovered,ww,hh=decode(blob);dec=time.perf_counter()-t
 assert (recovered,ww,hh)==(raw,w,h)
 baselines={}
 for fmt,opts in [('PNG',{'compress_level':9}),('WEBP',{'lossless':True,'quality':100,'method':6})]:
  f=io.BytesIO();im.save(f,format=fmt,**opts);encoded=f.getvalue();assert Image.open(io.BytesIO(encoded)).convert('RGB').tobytes()==raw;baselines[fmt]=len(encoded)
 residual=predict(color(raw),w,h)
 variants={}
 for mode in range(3):
  stream=layout(residual,mode)
  for backend in range(2):
   payload=zlib.compress(stream,9) if backend==0 else lzma.compress(stream,preset=6)
   variants[f'{mode}-{backend}']=len(payload)+14
 rows.append(dict(image=name,width=w,height=h,exp110=len(blob),mode=blob[12],backend=blob[13],encode_s=enc,decode_s=dec,baselines=baselines,variants=variants,roundtrip=True))
print(json.dumps({'environment':{'python':sys.version,'pillow':Image.__version__,'platform':platform.platform(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},'results':rows},indent=2))
