import unittest
import torch
from native1bit.quantization import binary_quantize_ste

class TestQuantization(unittest.TestCase):
    def test_ste_forward(self):
        w = torch.tensor([-0.5, 0.1, 0.0, 1.0, -1.0], requires_grad=True)
        q = binary_quantize_ste(w)
        expected = torch.tensor([-1.0, 1.0, 1.0, 1.0, -1.0])
        self.assertTrue(torch.equal(q, expected))

    def test_ste_backward(self):
        w = torch.tensor([0.5], requires_grad=True)
        q = binary_quantize_ste(w)
        loss = q.sum()
        loss.backward()
        # Identity gradient
        self.assertEqual(w.grad.item(), 1.0)

if __name__ == '__main__':
    unittest.main()
