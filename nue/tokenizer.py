from collections import Counter
from tqdm import tqdm
import numpy as np
import heapq


class _TrieNode:
    """Trie node for O(1)-per-byte token lookup during encoding."""

    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children: dict = {}
        self.token_id = None


class Tokenizer:
    def __init__(self, vocab_size=32768):
        self.vocab_size = vocab_size
        self.decoder: dict = {i: bytes([i]) for i in range(256)}
        self._byte_to_token = list(range(256))
        self._trie_root = self._build_trie()

    # ── Training ──────────────────────────────────────────────────────

    def train(self, text: str, num_workers: int = 0):
        """Train BPE tokenizer using pair-index approach.

        Maintains:
        - tokens list (growable, gaps marked with -1)
        - nxt/prev arrays for linked list traversal
        - pair_pos dict: (a,b) -> set of positions where pair starts
        - heap for efficient max-frequency lookup

        Each merge only touches O(1) positions and updates O(1) pair sets.
        Final cleanup pass removes gaps in O(n).

        Args:
            text: training text
            num_workers: unused (kept for API compatibility)
        """
        data = text.encode("utf-8")
        tokens = list(np.frombuffer(data, dtype=np.uint8))
        n = len(tokens)
        num_merges = self.vocab_size - 256

        # Doubly-linked list. -1 = null.
        nxt = list(range(1, n + 1))
        nxt[n - 1] = -1
        prev = [-1] + list(range(n - 1))

        head = 0
        next_id = 256  # merged tokens get IDs starting at 256, not n

        # pair_pos[(a, b)] = set of positions where pair (a,b) starts
        pair_pos = {}
        for i in range(n - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair not in pair_pos:
                pair_pos[pair] = set()
            pair_pos[pair].add(i)

        # Max-heap: (-count, a, b)
        heap = [(-len(positions), a, b) for (a, b), positions in pair_pos.items()]
        heapq.heapify(heap)

        pbar = tqdm(range(num_merges), desc="Tokenizer Training")

        for _ in pbar:
            # Pop most frequent valid pair
            best_pair = None
            while heap:
                neg_cnt, a, b = heapq.heappop(heap)
                pair = (a, b)
                positions = pair_pos.get(pair)
                if positions and len(positions) == -neg_cnt:
                    best_pair = pair
                    break

            if best_pair is None:
                break

            a, b = best_pair
            positions = pair_pos.pop(best_pair)
            new_token_id = next_id
            next_id += 1
            self.decoder[new_token_id] = self.decoder[a] + self.decoder[b]

            changed_pairs = set()  # track pairs whose counts changed

            for i in positions:
                j = nxt[i]
                # Validate: both alive and correct tokens
                if j == -1 or tokens[i] != a or tokens[j] != b:
                    continue

                # Capture neighbors before modification
                left = prev[i]
                right = nxt[j]

                left_val = tokens[left] if left != -1 else None
                right_val = tokens[right] if right != -1 else None

                # ── Remove old pairs from pair_pos ──
                if left != -1:
                    old_l = (left_val, a)
                    if old_l in pair_pos:
                        pair_pos[old_l].discard(left)
                        changed_pairs.add(old_l)
                if right != -1:
                    old_r = (b, right_val)
                    if old_r in pair_pos:
                        pair_pos[old_r].discard(j)
                        changed_pairs.add(old_r)
                        if not pair_pos[old_r]:
                            del pair_pos[old_r]

                # ── Perform merge ──
                tokens[i] = new_token_id
                nxt[i] = right
                if right != -1:
                    prev[right] = i

                # ── Add new pairs to pair_pos ──
                if left != -1:
                    new_l = (left_val, new_token_id)
                    if new_l not in pair_pos:
                        pair_pos[new_l] = set()
                    pair_pos[new_l].add(left)
                    changed_pairs.add(new_l)
                if right != -1:
                    new_r = (new_token_id, right_val)
                    if new_r not in pair_pos:
                        pair_pos[new_r] = set()
                    pair_pos[new_r].add(i)
                    changed_pairs.add(new_r)

            # Push only changed pairs onto heap (lazy stale skipping handles the rest)
            for pair in changed_pairs:
                positions = pair_pos.get(pair)
                if positions:
                    heapq.heappush(heap, (-len(positions), pair[0], pair[1]))

            pbar.set_postfix({"vocab": len(self.decoder)})

        # ── Rebuild flat token list from linked list ──
        result = []
        pos = head
        while pos != -1:
            result.append(tokens[pos])
            pos = nxt[pos]
        tokens = result

        # Build trie for fast encoding
        self._trie_root = self._build_trie()

    # ── Trie construction ─────────────────────────────────────────────

    def _build_trie(self) -> _TrieNode:
        """Build a trie from decoder for O(1)-per-byte encoding."""
        root = _TrieNode()
        for token_id, token_bytes in self.decoder.items():
            node = root
            for byte_val in token_bytes:
                if byte_val not in node.children:
                    node.children[byte_val] = _TrieNode()
                node = node.children[byte_val]
            node.token_id = token_id
        return root

    # ── Encoding ──────────────────────────────────────────────────────

    def encode(self, text: str) -> list:
        """Encode text to token ids using trie-based longest-match."""
        data = text.encode("utf-8")
        result = []
        i = 0
        n = len(data)
        root = self._trie_root

        while i < n:
            node = root
            last_match = self._byte_to_token[data[i]]
            last_match_len = 1
            j = i

            while j < n and data[j] in node.children:
                node = node.children[data[j]]
                j += 1
                if node.token_id is not None:
                    last_match = node.token_id
                    last_match_len = j - i

            result.append(last_match)
            i += last_match_len

        return result

    # ── Decoding ──────────────────────────────────────────────────────

    def decode(self, token_ids: list) -> str:
        buf = bytearray()
        d = self.decoder
        for tid in token_ids:
            buf.extend(d.get(tid, b""))
        return buf.decode("utf-8", errors="replace")
