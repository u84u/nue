import numpy as np
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

class Tokenizer:
    def __init__(self, vocab_size=32768):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        # Base vocabulary is all bytes
        for i in range(256):
            self.encoder[bytes([i])] = i
            self.decoder[i] = bytes([i])

    def _count_pairs(self, words):
        return Counter(zip(words[:-1], words[1:]))

    def train(self, text: str):
        data = text.encode("utf-8")
        words = [bytes([b]) for b in data]
        
        num_merges = self.vocab_size - 256
        for i in range(num_merges):
            # Parallel pair counting
            num_workers = os.cpu_count() or 1
            chunk_size = max(1, len(words) // num_workers)
            chunks = [words[j:j+chunk_size] for j in range(0, len(words), chunk_size)]
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(self._count_pairs, chunks))
            
            pairs = Counter()
            for r in results:
                pairs.update(r)
            
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            
            new_token = best_pair[0] + best_pair[1]
            token_id = 256 + i
            
            self.encoder[new_token] = token_id
            self.decoder[token_id] = new_token
            
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
        data = text.encode("utf-8")
        return list(data)

    def decode(self, token_ids: list[int]) -> str:
        return b"".join([bytes([id]) for id in token_ids]).decode("utf-8", errors="replace")
