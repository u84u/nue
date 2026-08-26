import unittest
import time
from native1bit.tokenizer import Tokenizer


class TestTokenizerRoundtrip(unittest.TestCase):
    """Verify encode→decode roundtrip for various text types."""

    def test_ascii(self):
        tokenizer = Tokenizer()
        text = "Hello, world! This is a test."
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_multibyte_utf8(self):
        tokenizer = Tokenizer()
        text = "café résumé naïve über"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_emoji(self):
        tokenizer = Tokenizer()
        text = "Hello 🌍! 🚀 is cool."
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_newlines_and_tabs(self):
        tokenizer = Tokenizer()
        text = "line1\nline2\n\tindented"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_empty_string(self):
        tokenizer = Tokenizer()
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")

    def test_single_char(self):
        tokenizer = Tokenizer()
        for ch in ["a", "Z", "0", " ", "\n", "中"]:
            encoded = tokenizer.encode(ch)
            self.assertEqual(tokenizer.decode(encoded), ch)

    def test_long_text(self):
        tokenizer = Tokenizer()
        text = "The quick brown fox jumps over the lazy dog. " * 1000
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


class TestTokenizerTraining(unittest.TestCase):
    """Verify BPE training produces a working tokenizer."""

    def test_train_and_encode(self):
        tokenizer = Tokenizer(vocab_size=512)
        text = "hello world " * 500 + "foo bar " * 500
        tokenizer.train(text)
        # Should have learned some multi-byte tokens
        self.assertGreater(len(tokenizer.decoder), 256)
        # Roundtrip must still work
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_vocab_size_respected(self):
        tokenizer = Tokenizer(vocab_size=1024)
        text = "abcdefghij" * 1000
        tokenizer.train(text)
        self.assertLessEqual(len(tokenizer.decoder), 1024)

    def test_merges_reduce_token_count(self):
        tokenizer = Tokenizer(vocab_size=1024)
        text = "aaaa bbbb cccc " * 200
        before = len(tokenizer.encode(text))
        tokenizer.train(text)
        after = len(tokenizer.encode(text))
        self.assertLess(after, before)

    def test_trained_encode_is_fast(self):
        tokenizer = Tokenizer(vocab_size=1024)
        text = "The quick brown fox. " * 500
        tokenizer.train(text)
        t0 = time.time()
        for _ in range(100):
            tokenizer.encode(text)
        elapsed = time.time() - t0
        # 100 encodes of ~11k chars should be well under 1 second
        self.assertLess(elapsed, 1.0)

    def test_token_ids_within_vocab_size(self):
        """All token IDs from encode must be < vocab_size after training."""
        for vocab_size in [256, 512, 1024, 4096]:
            tokenizer = Tokenizer(vocab_size=vocab_size)
            text = "The quick brown fox jumps over the lazy dog. " * 200
            tokenizer.train(text)
            tokens = tokenizer.encode(text)
            max_id = max(tokens)
            self.assertLess(
                max_id, vocab_size,
                f"Max token ID {max_id} >= vocab_size {vocab_size}",
            )
            self.assertEqual(tokenizer.decode(tokens), text)

    def test_training_time_medium_scale(self):
        """Tokenizer training on ~500KB text with vocab 4096 should be fast."""
        # ~500KB of repetitive text
        lines = []
        for i in range(5000):
            lines.append(f"Line {i}: The quick brown fox jumps over the lazy dog.")
            lines.append("ROMEO. But soft, what light through yonder window breaks?")
        text = "\n".join(lines)

        tokenizer = Tokenizer(vocab_size=4096)
        t0 = time.time()
        tokenizer.train(text)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 30)  # under 30 seconds


class TestTokenizerVocab(unittest.TestCase):
    """Verify vocabulary properties after training."""

    def test_all_byte_tokens_present(self):
        tokenizer = Tokenizer(vocab_size=1024)
        tokenizer.train("hello " * 200)
        for i in range(256):
            self.assertIn(i, tokenizer.decoder)
            self.assertEqual(tokenizer.decoder[i], bytes([i]))

    def test_multi_byte_tokens_are_concatenations(self):
        tokenizer = Tokenizer(vocab_size=1024)
        tokenizer.train("hello world " * 200)
        for token_id, token_bytes in tokenizer.decoder.items():
            if token_id >= 256:
                # Multi-byte token must be composed of valid byte sequences
                self.assertGreater(len(token_bytes), 1)

    def test_vocabulary_completeness(self):
        """Every single byte value must have a decoder entry and roundtrip."""
        tokenizer = Tokenizer(vocab_size=512)
        tokenizer.train("sample text " * 100)
        for byte_val in range(256):
            # Each byte must map to exactly itself in the decoder
            self.assertIn(byte_val, tokenizer.decoder)
            self.assertEqual(tokenizer.decoder[byte_val], bytes([byte_val]))
        # Encoding valid UTF-8 text must always roundtrip
        for text in ["hello", "café", "日本語", "🚀"]:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
