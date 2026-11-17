import unittest,random
from exp111 import encode,decode,encode_mode
class Test(unittest.TestCase):
 def test_roundtrips(self):
  rng=random.Random(111)
  for w,h in [(1,1),(1,13),(13,1),(9,9),(17,23)]:
   for raw in [bytes(w*h*3),bytes(i%256 for i in range(w*h*3)),bytes(rng.randrange(256) for _ in range(w*h*3))]:
    for m in range(2):
     for b in range(2):self.assertEqual(decode(encode_mode(raw,w,h,m,b)),(raw,w,h))
    self.assertEqual(decode(encode(raw,w,h)),(raw,w,h))
 def test_bad(self):
  x=encode(bytes(range(48)),4,4)
  for bad in [b'',x[:13],x[:-1],x+b'x',x[:4]+b'\0'*10+x[14:]]:
   with self.assertRaises(Exception):decode(bad)
if __name__=='__main__':unittest.main()
