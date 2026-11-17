import unittest,random
from exp110 import *
class Tests(unittest.TestCase):
 def test_layout(self):
  for n in (1,3,19,128):
   raw=bytes(random.Random(n).randrange(256) for _ in range(n*3))
   for mode in range(3):self.assertEqual(layout(layout(raw,mode),mode,True),raw)
 def test_roundtrip(self):
  for w,h in ((1,1),(1,8),(8,1),(8,8),(19,17)):
   rng=random.Random(w*1000+h)
   for raw in (bytes(w*h*3),bytes(rng.randrange(256) for _ in range(w*h*3)),bytes(i%256 for i in range(w*h*3))):
    self.assertEqual(decode(encode(raw,w,h)),(raw,w,h))
 def test_invalid(self):
  b=encode(bytes(8*8*3),8,8)
  for x in (b'',b[:12],b[:-1],b+b'extra',b'BAD!'+b[4:]):
   with self.assertRaises(Exception):decode(x)
if __name__=='__main__':unittest.main()
