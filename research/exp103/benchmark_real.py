"""Reproducible bundled real-image smoke benchmark; not independent held-out validation."""
import io,json,time,hashlib,platform,sys
import numpy as np
from PIL import Image,features
from skimage import data,color
from exp103 import encode,decode
cases={'camera':data.camera(),'coins':data.coins(),'moon':data.moon(),'page':data.page(),'astronaut_gray':(color.rgb2gray(data.astronaut())*255).astype('uint8')}
for name,arr in cases.items():
 arr=np.asarray(arr,dtype=np.uint8);h,w=arr.shape;raw=arr.tobytes();im=Image.fromarray(arr,'L')
 def pilcodec(fmt,**opts):
  f=io.BytesIO();im.save(f,format=fmt,**opts);blob=f.getvalue()
  assert Image.open(io.BytesIO(blob)).convert('L').tobytes()==raw
  return len(blob)
 t=time.perf_counter();b=encode(raw,w,h);enc=time.perf_counter()-t
 t=time.perf_counter();r,rw,rh=decode(b);dec=time.perf_counter()-t
 assert (r,rw,rh)==(raw,w,h)
 row=dict(image=name,width=w,height=h,prototype_bytes=len(b),png_bytes=pilcodec('PNG',compress_level=9),webp_lossless_bytes=pilcodec('WEBP',lossless=True,quality=100,method=6),encode_s=round(enc,5),decode_s=round(dec,5),predictor=b[12],sha256=hashlib.sha256(raw).hexdigest())
 try:row['jpeg2000_lossless_bytes']=pilcodec('JPEG2000',irreversible=False)
 except Exception as e:row['jpeg2000_error']=str(e)
 print(json.dumps(row),flush=True)
print('VERSIONS',json.dumps(dict(python=sys.version,pillow=Image.__version__,platform=platform.platform(),webp=features.version('webp'))))
