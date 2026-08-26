import unittest
import torch
import os
from nue.transformer import TinyTransformer
from nue.weights_format import save_model_weights

class TestCheckpoint(unittest.TestCase):
    def test_save_load(self):
        vocab_size = 64
        hidden_size = 16
        num_layers = 1
        num_heads = 2
        head_dim = 8
        mlp_size = 32
        
        model = TinyTransformer(vocab_size, hidden_size, num_layers, num_heads, 2, head_dim, mlp_size)
        filepath = "test_model.weights"
        
        save_model_weights(model, filepath)
        
        self.assertTrue(os.path.exists(filepath))
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    unittest.main()
