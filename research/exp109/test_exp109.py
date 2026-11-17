import random,unittest
from exp109 import encode,decode
class Tests(unittest.TestCase):
 def test_roundtrip(self):
  rng=random.Random(109)
  for w,h in [(1,1),(1,7),(7,1),(8,8),(31,19)]:
   for raw in [bytes(w*h*3),bytes(i%256 for i in range(w*h*3)),bytes(rng.randrange(256) for _ in range(w*h*3))]:
    for backend in (0,1,2):
     self.assertEqual(decode(encode(raw,w,h,(backend,))),(raw,w,h))
 def test_header(self):
  with self.assertRaises(ValueError):decode(b'')
  with self.assertRaises(ValueError):decode(b'B109'+bytes(10))
if __name__=='__main__':unittest.main()
