import unittest,random
from exp113 import encode,decode
class Test(unittest.TestCase):
 def test_roundtrip(self):
  for w,h in [(1,1),(2,3),(17,13)]:
   for seed in range(3):
    raw=random.Random(seed).randbytes(w*h*3)
    for mode in range(4):
     for backend in (0,1):
      self.assertEqual(decode(encode(raw,w,h,mode,backend)),(raw,w,h))
 def test_invalid(self):
  for blob in [b'',b'B113',b'B113'+b'\x00'*14]:
   with self.assertRaises(ValueError):decode(blob)
if __name__=='__main__':unittest.main()
