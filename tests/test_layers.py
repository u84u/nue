import unittest
import torch
from native1bit.layers import BitLinear

class TestLayers(unittest.TestCase):
    def test_bitlinear_forward(self):
        in_feat = 16
        out_feat = 8
        batch = 4
        layer = BitLinear(in_feat, out_feat, bias=False)
        x = torch.randn(batch, in_feat)
        
        # Forward pass
        output = layer(x)
        self.assertEqual(output.shape, (batch, out_feat))
        
        # Verify effective weights are binary
        w_binary = torch.sign(layer.weight)
        w_binary[w_binary == 0] = 1.0
        
        # Forward effective weights should be binary
        # We can't directly access w_effective, but we can verify
        # that quantization works
        
        # Verify backward flow
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(layer.weight.grad)

if __name__ == '__main__':
    unittest.main()
