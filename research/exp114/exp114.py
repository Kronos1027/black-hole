"""Exp114: reversible RGB transforms with zlib/LZMA; lossless holdout benchmark."""
import io,json,lzma,platform,resource,struct,time,zlib,hashlib,unittest
import numpy as np
from PIL import Image
MAGIC=b'B114'
METHODS=('raw','green','ycocg')
ENTS=('zlib','lzma')
def transform(a,method):
 a=a.astype(np.int32);r,g,b=[a[:,:,i] for i in range(3)]
 if method=='raw':return a.astype('uint8')
 if method=='green':return np.stack(((r-g)&255,g,(b-g)&255),2).astype('uint8')
 co=(r-b)&255;t=(b+(co>>1))&255;cg=(g-t)&255;y=(t+(cg>>1))&255
 return np.stack((y,co,cg),2).astype('uint8')
def inverse(a,method):
 a=a.astype(np.int32);x,y,z=[a[:,:,i] for i in range(3)]
 if method=='raw':return a.astype('uint8')
 if method=='green':return np.stack(((x+y)&255,y,(z+y)&255),2).astype('uint8')
 t=(x-(z>>1))&255;g=(z+t)&255;b=(t-(y>>1))&255;r=(y+b)&255
 return np.stack((r,g,b),2).astype('uint8')
def encode(a,method='green',ent='zlib'):
 h,w,c=a.shape
 assert c==3 and 0<h<65536 and 0<w<65536
 t=transform(a,method);res=np.empty_like(t);res[0]=t[0]
 res[1:]=(t[1:].astype('int16')-t[:-1].astype('int16'))&255
 p=zlib.compress(res.tobytes(),9) if ent=='zlib' else lzma.compress(res.tobytes(),preset=6)
 return MAGIC+struct.pack('>HHBBI',w,h,METHODS.index(method),ENTS.index(ent),len(p))+p
def decode(blob):
 if len(blob)<14 or blob[:4]!=MAGIC:raise ValueError('bad header')
 w,h,m,e,n=struct.unpack('>HHBBI',blob[4:14])
 if not w or not h or m>=3 or e>=2 or n!=len(blob)-14:raise ValueError('invalid fields')
 raw=zlib.decompress(blob[14:]) if e==0 else lzma.decompress(blob[14:])
 if len(raw)!=w*h*3:raise ValueError('invalid decoded length')
 a=np.frombuffer(raw,dtype='uint8').reshape(h,w,3).copy()
 for row in range(1,h):a[row]=(a[row].astype('uint16')+a[row-1].astype('uint16'))&255
 return inverse(a,METHODS[m])
def images():
 from skimage import data
 for name in ('brick','grass','gravel','page','text'):
  a=np.asarray(getattr(data,name)())
  if a.ndim==2:a=np.repeat(a[:,:,None],3,axis=2)
  yield name,a[:192,:192,:3].astype('uint8')
class Test(unittest.TestCase):
 def test_roundtrip(self):
  rng=np.random.default_rng(20261117)
  for shape in ((1,1,3),(3,7,3),(31,37,3)):
   a=rng.integers(0,256,shape,dtype='uint8')
   for m in METHODS:
    for e in ENTS:self.assertTrue(np.array_equal(a,decode(encode(a,m,e))))
 def test_reject(self):
  for b in (b'',b'B114',b'B114'+bytes(10)):
   with self.assertRaises(Exception):decode(b)
def benchmark():
 records=[]
 for name,a in images():
  pil=Image.fromarray(a,'RGB');wb=io.BytesIO();pil.save(wb,format='WEBP',lossless=True,method=6)
  webp=wb.getvalue();assert np.array_equal(np.array(Image.open(io.BytesIO(webp)).convert('RGB')),a)
  pb=io.BytesIO();pil.save(pb,format='PNG',optimize=True);png=pb.getvalue()
  assert np.array_equal(np.array(Image.open(io.BytesIO(png)).convert('RGB')),a)
  row={'image':name,'sha256':hashlib.sha256(a.tobytes()).hexdigest(),'webp':len(webp),'png':len(png),'variants':{}}
  for m in METHODS:
   for e in ENTS:
    t=time.perf_counter();blob=encode(a,m,e);enc_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter();rec=decode(blob);dec_ms=(time.perf_counter()-t)*1000
    assert np.array_equal(rec,a)
    row['variants'][m+'_'+e]={'bytes':len(blob),'encode_ms':round(enc_ms,3),'decode_ms':round(dec_ms,3)}
  records.append(row)
 return {'environment':{'python':platform.python_version(),'numpy':np.__version__,'pillow':Image.__version__,'platform':platform.platform(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},'records':records}
if __name__=='__main__':
 if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Test)).wasSuccessful():raise SystemExit(1)
 print(json.dumps(benchmark(),indent=2))
