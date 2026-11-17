import io,json,platform,resource,time,hashlib
import numpy as np
from PIL import Image
from skimage import data,color
from exp105 import encode,decode
def gray(im):
 a=np.asarray(im)
 if a.ndim==3:a=color.rgb2gray(a[:,:,:3])*255
 return np.asarray(np.clip(np.rint(a),0,255),dtype=np.uint8)
def run():
 dev={'camera':gray(data.camera()),'coins':gray(data.coins()),'moon':gray(data.moon())}
 test={'page':gray(data.page()),'astronaut':gray(data.astronaut()),'chelsea':gray(data.chelsea()),'coffee':gray(data.coffee())}
 results=[]
 for split,images in [('dev',dev),('test',test)]:
  for name,a in images.items():
   a=a[:128,:128].copy();h,w=a.shape;raw=a.tobytes()
   buf=io.BytesIO();Image.fromarray(a).save(buf,format='PNG',optimize=True);png=len(buf.getvalue())
   buf=io.BytesIO();Image.fromarray(a).save(buf,format='WEBP',lossless=True,method=6);webp=len(buf.getvalue())
   for rows in [1,8,32,128]:
    t=time.perf_counter();b=encode(raw,w,h,rows);et=time.perf_counter()-t
    t=time.perf_counter();rest=decode(b);dt=time.perf_counter()-t
    assert rest==(raw,w,h)
    results.append(dict(split=split,image=name,rows=rows,bytes=len(b),png=png,webp=webp,encode_s=round(et,5),decode_s=round(dt,5),sha256=hashlib.sha256(raw).hexdigest(),exact=True))
 print(json.dumps(dict(platform=platform.platform(),python=platform.python_version(),peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,results=results),indent=2))
if __name__=='__main__':run()
