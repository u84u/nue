import unittest
import torch
from native1bit.binary import sign_quantize, pack_bits

class TestBinaryRepresentation(unittest.TestCase):
    def test_sign_quantize(self):
        w = torch.tensor([-0.5, 0.1, 0.0, 1.0, -1.0])
        q = sign_quantize(w)
        expected = torch.tensor([-1.0, 1.0, 1.0, 1.0, -1.0])
        self.assertTrue(torch.equal(q, expected))

    def test_pack_bits(self):
        # 8 bits: {-1, 1, -1, -1, 1, 1, 1, -1}
        # mapped to: {0, 1, 0, 0, 1, 1, 1, 0}
        # little-endian: idx 0 is LSB. 
        # bits (idx 0 to 7): 0, 1, 0, 0, 1, 1, 1, 0
        # values: 1*0 + 2*1 + 4*0 + 8*0 + 16*1 + 32*1 + 64*1 + 128*0 = 2 + 16 + 32 + 64 = 114 (0x72)
        binary_tensor = torch.tensor([-1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0])
        packed = pack_bits(binary_tensor)
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0], 0x72)

if __name__ == '__main__':
    unittest.main()
