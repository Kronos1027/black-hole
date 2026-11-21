# Exp211: adaptive PNG-like row filters, lossless RGB
# Standalone Python 3; pip install numpy pillow scikit-image
import io,struct,zlib,numpy as np
from PIL import Image
from skimage import data
NAMES=['astronaut','coffee','chelsea','rocket','camera','coins','moon','horse']
def predictors(row,prev):
 left=np.concatenate([np.zeros((1,3),dtype=np.int16),row[:-1]])
 ul=np.concatenate([np.zeros((1,3),dtype=np.int16),prev[:-1]])
 p=left+prev-ul
 pa,pb,pc=np.abs(p-left),np.abs(p-prev),np.abs(p-ul)
 return [np.zeros_like(row),left,prev,(left+prev)//2,np.where((pa<=pb)&(pa<=pc),left,np.where(pb<=pc,prev,ul))]
def encode(a,mode):
 h,w,_=a.shape;prev=np.zeros((w,3),dtype=np.int16);raw=bytearray()
 for r in a:
  row=r.astype(np.int16);ps=predictors(row,prev)
  idx=int(np.argmin([np.abs((row-p+128)%256-128).sum() for p in ps])) if mode=='adaptive' else {'raw':0,'sub':1,'up':2,'avg':3,'paeth':4}[mode]
  raw.append(idx);raw.extend(((row-ps[idx])%256).astype('uint8').tobytes());prev=row
 return b'B211'+struct.pack('>HH',w,h)+zlib.compress(raw,9)
def decode(b):
 assert b[:4]==b'B211'
 w,h=struct.unpack('>HH',b[4:8]);raw=zlib.decompress(b[8:]);assert len(raw)==h*(1+3*w)
 out=np.empty((h,w,3),dtype='uint8');prev=np.zeros((w,3),dtype=np.int16);i=0
 for y in range(h):
  idx=raw[i];i+=1;assert idx<5
  residual=np.frombuffer(raw[i:i+3*w],dtype='uint8').reshape(w,3).astype(np.int16);i+=3*w
  row=np.zeros((w,3),dtype=np.int16)
  for x in range(w):
   left=row[x-1] if x else np.zeros(3,dtype=np.int16)
   up=prev[x];ul=prev[x-1] if x else np.zeros(3,dtype=np.int16)
   p=left+up-ul;pa,pb,pc=np.abs(p-left),np.abs(p-up),np.abs(p-ul)
   pred=[np.zeros(3,dtype=np.int16),left,up,(left+up)//2,np.where((pa<=pb)&(pa<=pc),left,np.where(pb<=pc,up,ul))][idx]
   row[x]=(residual[x]+pred)%256
  out[y]=row.astype('uint8');prev=row
 return out
if __name__=='__main__':
 for name in NAMES:
  a=np.asarray(getattr(data,name)())
  if a.ndim==2:a=np.stack([a]*3,axis=-1)
  a=np.asarray(Image.fromarray(a[:,:,:3].astype('uint8')).resize((128,128)))
  buf=io.BytesIO();Image.fromarray(a).save(buf,format='WEBP',lossless=True,method=6)
  assert np.array_equal(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGB')),a)
  sizes={}
  for mode in ['raw','sub','up','avg','paeth','adaptive']:
   b=encode(a,mode);assert np.array_equal(decode(b),a);sizes[mode]=len(b)
  print(name,'webp',len(buf.getvalue()),sizes)
