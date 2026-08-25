import unittest
from native1bit.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    def test_basic_encoding_decoding(self):
        tokenizer = Tokenizer()
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(text, decoded)

    def test_byte_values(self):
        tokenizer = Tokenizer()
        text = "a"
        encoded = tokenizer.encode(text)
        self.assertEqual(encoded, [ord("a")])

if __name__ == '__main__':
    unittest.main()
