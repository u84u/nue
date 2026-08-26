import unittest
import torch
from native1bit.transformer import TinyTransformer

class TestTransformer(unittest.TestCase):
    def test_forward(self):
        vocab_size = 1024
        hidden_size = 64
        num_layers = 2
        num_heads = 4
        head_dim = 16
        mlp_size = 256
        batch = 2
        seq_len = 8
        
        model = TinyTransformer(vocab_size, hidden_size, num_layers, num_heads, 2, head_dim, mlp_size)
        idx = torch.randint(0, vocab_size, (batch, seq_len))
        
        logits = model(idx)
        self.assertEqual(logits.shape, (batch, seq_len, vocab_size))
        
        # Test backward
        loss = logits.sum()
        loss.backward()
        
        # Check that gradients exist for weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")

if __name__ == '__main__':
    unittest.main()
