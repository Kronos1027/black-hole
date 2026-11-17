import random,unittest
from exp108 import encode,decode
class Tests(unittest.TestCase):
 def test_roundtrip(self):
  rng=random.Random(108)
  for w,h in [(1,1),(1,9),(9,1),(16,16),(33,19)]:
   for raw in (bytes(w*h*3),bytes(rng.randrange(256) for _ in range(w*h*3)),bytes((i//3)%256 for i in range(w*h*3))):
    for adaptive in (False,True):
     for decorrelate in (False,True):self.assertEqual(decode(encode(raw,w,h,adaptive,decorrelate)),(raw,w,h))
 def test_invalid(self):
  b=encode(bytes(8*8*3),8,8)
  for x in (b'',b[:12],b[:-1],b+b'junk',b[:12]+b'\\xff'+b[13:]):
   with self.assertRaises(ValueError):decode(x)
if __name__=='__main__':unittest.main()
