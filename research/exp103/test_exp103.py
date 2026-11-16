import random,unittest
from exp103 import encode,decode
class TestExp103(unittest.TestCase):
 def test_roundtrip(self):
  rng=random.Random(103)
  for w,h in [(1,1),(1,17),(17,1),(8,8),(31,19),(64,64)]:
   for raw in [bytes(w*h),bytes(i%256 for i in range(w*h)),bytes(rng.randrange(256) for _ in range(w*h))]:
    self.assertEqual(decode(encode(raw,w,h)),(raw,w,h))
 def test_corrupt(self):
  blob=encode(bytes(range(64)),8,8)
  for bad in [b'',blob[:12],blob[:13],blob[:-1],blob+b'junk']:
   with self.assertRaises(Exception):decode(bad)
if __name__=='__main__':unittest.main()
