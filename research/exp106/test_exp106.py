import random,unittest
from exp106 import *
class Test(unittest.TestCase):
 def test_roundtrip(self):
  rng=random.Random(106)
  for w,h in [(1,1),(1,9),(9,1),(8,8),(17,13)]:
   for raw in [bytes(3*w*h),bytes(i%256 for i in range(3*w*h)),bytes(rng.randrange(256) for _ in range(3*w*h))]:
    self.assertEqual(decode(encode(raw,w,h)),(raw,w,h))
 def test_transforms(self):
  rng=random.Random(3);raw=bytes(rng.randrange(256) for _ in range(3*17*19))
  for m in range(3):self.assertEqual(unplanes(planes(raw,17,19,m),m),raw)
 def test_corruption(self):
  blob=encode(bytes(range(192)),8,8)
  for x in [b'',blob[:13],blob[:-1],blob+b'x']:
   with self.assertRaises(ValueError):decode(x)
if __name__=='__main__':unittest.main()
