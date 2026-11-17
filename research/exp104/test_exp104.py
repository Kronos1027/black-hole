import random,unittest
from exp104 import encode,decode
class TestRowPredictor(unittest.TestCase):
 def test_roundtrip(self):
  r=random.Random(104)
  for w,h in [(1,1),(1,19),(19,1),(17,31),(64,64)]:
   for raw in [bytes(w*h),bytes(i%256 for i in range(w*h)),r.randbytes(w*h)]:
    self.assertEqual(decode(encode(raw,w,h)),(raw,w,h))
 def test_corruption(self):
  b=encode(bytes(range(64)),8,8)
  for corrupt in [b'',b[:11],b[:-1],b+b'junk']:
   with self.assertRaises(ValueError):decode(corrupt)
if __name__=='__main__':unittest.main()
