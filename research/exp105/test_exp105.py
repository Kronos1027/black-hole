import random,unittest
from exp105 import encode,decode
class Test(unittest.TestCase):
 def test_roundtrip(self):
  r=random.Random(105)
  for w,h in [(1,1),(1,17),(17,1),(8,8),(31,19),(128,128)]:
   for raw in [bytes(w*h),bytes(i%256 for i in range(w*h)),bytes(r.randrange(256) for _ in range(w*h))]:
    for rows in [1,8,32,128]:
     self.assertEqual(decode(encode(raw,w,h,rows)),(raw,w,h))
 def test_corrupt(self):
  b=encode(bytes(range(64)),8,8,8)
  for bad in [b'',b[:13],b[:-1],b+b'junk']:
   with self.assertRaises(ValueError):decode(bad)
if __name__=='__main__':unittest.main()
