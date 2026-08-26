import unittest
import torch
from nue.tokenizer import Tokenizer
from nue.data import prepare_instruct_data, masked_ntp_loss

class TestInstructData(unittest.TestCase):
    def test_masking(self):
        tokenizer = Tokenizer()
        system = "You are a helpful assistant."
        user = "Hello"
        assistant = "Hi there!"
        
        system_tokens = tokenizer.encode(f"system: {system}\n")
        user_tokens = tokenizer.encode(f"user: {user}\n")
        assistant_tokens = tokenizer.encode(f"assistant: {assistant}")
        
        tokens, mask = prepare_instruct_data(tokenizer, system, user, assistant, 1024)
        
        assistant_len = len(assistant_tokens)
        
        # Assistant tokens should have mask=1
        self.assertEqual(mask[-assistant_len:].sum(), assistant_len)
        # System+user should have mask=0
        self.assertEqual(mask[:-assistant_len].sum(), 0)

if __name__ == '__main__':
    unittest.main()
