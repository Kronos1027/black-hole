import io,json,time,platform,resource,hashlib
import numpy as np
from PIL import Image
from skimage import data
from exp112 import encode_mode,decode
NAMES=['astronaut','chelsea','coffee','rocket','hubble_deep_field','immunohistochemistry','colorwheel']
rows=[]
for name in NAMES:
 arr=np.asarray(getattr(data,name)());arr=arr[:,:,:3] if arr.ndim==3 else np.repeat(arr[:,:,None],3,axis=2)
 h,w=arr.shape[:2];scale=min(96/h,96/w,1)
 if scale<1:arr=np.asarray(Image.fromarray(arr.astype('uint8')).resize((round(w*scale),round(h*scale)),Image.Resampling.LANCZOS))
 h,w=arr.shape[:2];raw=arr.astype('uint8').tobytes();image=Image.frombytes('RGB',(w,h),raw)
 out=io.BytesIO();image.save(out,format='PNG',optimize=True);png=out.getvalue()
 out=io.BytesIO();image.save(out,format='WEBP',lossless=True,quality=100,method=6);webp=out.getvalue()
 assert Image.open(io.BytesIO(png)).tobytes()==raw and Image.open(io.BytesIO(webp)).tobytes()==raw
 variants=[]
 for n in (1,2,4):
  for backend in ('zlib','lzma'):
   t=time.perf_counter();blob=encode_mode(raw,w,h,n,backend);enc=time.perf_counter()-t
   t=time.perf_counter();dec,dw,dh=decode(blob);dt=time.perf_counter()-t
   assert dec==raw and (dw,dh)==(w,h)
   variants.append(dict(contexts=n,backend=backend,bytes=len(blob),encode_s=enc,decode_s=dt))
 rows.append(dict(name=name,width=w,height=h,raw_sha256=hashlib.sha256(raw).hexdigest(),png_bytes=len(png),webp_bytes=len(webp),variants=variants))
report=dict(python=platform.python_version(),platform=platform.platform(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rows=rows)
print(json.dumps(report,indent=2))
