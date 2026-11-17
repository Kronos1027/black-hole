import random,unittest
from exp112 import encode_mode,decode
class Test(unittest.TestCase):
 def test_roundtrip(self):
  rng=random.Random(112)
  for w,h in [(1,1),(1,17),(17,1),(8,8),(32,24)]:
   for raw in [bytes(3*w*h),bytes(i%256 for i in range(3*w*h)),rng.randbytes(3*w*h)]:
    for n in (1,2,4):
     for backend in ('zlib','lzma'):
      self.assertEqual(decode(encode_mode(raw,w,h,n,backend)),(raw,w,h))
 def test_bad(self):
  b=encode_mode(bytes(3*8*8),8,8)
  for x in [b'',b[:13],b[:-1],b+b'X',b[:12]+b'\x03'+b[13:]]:
   with self.assertRaises(ValueError):decode(x)
if __name__=='__main__':unittest.main()
