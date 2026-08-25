import numpy as np
from collections import Counter, defaultdict

class Tokenizer:
    def __init__(self, vocab_size=32768):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        self.merges = {}
        # Base vocabulary is all bytes
        for i in range(256):
            self.encoder[bytes([i])] = i
            self.decoder[i] = bytes([i])

    def train(self, text: str):
        # Initial byte-level training: BPE algorithm
        data = text.encode("utf-8")
        
        # Current vocabulary consists of individual bytes
        vocab = {bytes([i]): i for i in range(256)}
        
        # Simple BPE Training
        # Note: This is a simplified BPE implementation for proof-of-concept
        # A robust implementation would use a more efficient data structure
        words = [bytes([b]) for b in data]
        
        num_merges = self.vocab_size - 256
        for i in range(num_merges):
            pairs = Counter(zip(words[:-1], words[1:]))
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            
            new_token = best_pair[0] + best_pair[1]
            token_id = 256 + i
            
            self.encoder[new_token] = token_id
            self.decoder[token_id] = new_token
            self.merges[best_pair] = new_token
            
            # Update sequence
            new_words = []
            j = 0
            while j < len(words):
                if j < len(words) - 1 and (words[j], words[j+1]) == best_pair:
                    new_words.append(new_token)
                    j += 2
                else:
                    new_words.append(words[j])
                    j += 1
            words = new_words
            
    def encode(self, text: str) -> list[int]:
        # Simple greedy encoding for proof-of-concept
        # In a real BPE, this would use the merges map efficiently
        data = text.encode("utf-8")
        # For now, just return byte values
        return list(data)

    def decode(self, token_ids: list[int]) -> str:
        # Simple decoding for proof-of-concept
        return b"".join([bytes([id]) for id in token_ids]).decode("utf-8", errors="replace")
