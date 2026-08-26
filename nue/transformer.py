import torch
import torch.nn as nn
from nue.layers import BitLinear, SubLN, BinaryEmbedding
import math
from typing import Optional


class KVCache:
    """Pre-allocated key/value cache for autoregressive generation.

    Stores past K and V tensors so that during generation only the new
    token needs to be projected, concatenated with the cache, and attended.
    """
    def __init__(self, max_seq_len: int, batch_size: int, num_kv_heads: int, head_dim: int, dtype=torch.float32, device='cpu'):
        self.max_seq_len = max_seq_len
        self.cur_len = 0
        # Pre-allocate full buffers — no reallocation during generation
        self.k = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append new K, V tokens to the cache.

        Args:
            k_new: (batch, num_kv_heads, seq_len, head_dim)
            v_new: (batch, num_kv_heads, seq_len, head_dim)
        """
        seq_len = k_new.shape[2]
        self.k[:, :, self.cur_len:self.cur_len + seq_len] = k_new
        self.v[:, :, self.cur_len:self.cur_len + seq_len] = v_new
        self.cur_len += seq_len

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached K, V up to cur_len."""
        # Use narrow() + reshape to avoid contiguous() copy
        # The cache is pre-allocated contiguous, so slicing then reshape is safe
        k = self.k[:, :, :self.cur_len]
        v = self.v[:, :, :self.cur_len]
        return k, v

    def reset(self):
        """Reset cache to empty (for new sequences)."""
        self.cur_len = 0


class KVCacheList:
    """Per-layer KV cache. Holds one KVCache per transformer layer."""
    def __init__(self, num_layers: int, max_seq_len: int, batch_size: int, num_kv_heads: int, head_dim: int, dtype=torch.float32, device='cpu'):
        self.caches = [
            KVCache(max_seq_len, batch_size, num_kv_heads, head_dim, dtype, device)
            for _ in range(num_layers)
        ]

    def __getitem__(self, layer_idx: int) -> KVCache:
        return self.caches[layer_idx]

    @property
    def cur_len(self) -> int:
        return self.caches[0].cur_len if self.caches else 0

    def reset(self):
        for c in self.caches:
            c.reset()


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Precomputes cos/sin frequencies for all positions up to max_seq_len.
    Applies position-dependent rotation to Q and K tensors.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        # Frequencies: theta_i = 1 / (base^(2i / head_dim)) for i in 0..head_dim//2
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # Position indices
        positions = torch.arange(max_seq_len).float()
        # Outer product: (max_seq_len, head_dim//2)
        angles = torch.outer(positions, freqs)
        # Register as buffer — moves with the module but is not a parameter
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """Apply rotary embeddings to q and k.

        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_kv_heads, seq_len, head_dim)
            offset: starting position offset (for cached inference)
        """
        seq_len = q.shape[2]
        cos = self.cos[offset : offset + seq_len]  # (seq_len, head_dim//2)
        sin = self.sin[offset : offset + seq_len]  # (seq_len, head_dim//2)

        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)
        return q_rot, k_rot

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate pairs of dimensions: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2] * cos/sin mix.

        x: (batch, heads, seq_len, head_dim)
        cos, sin: (seq_len, head_dim//2)
        """
        d2 = x.shape[-1] // 2
        x1 = x[..., :d2]
        x2 = x[..., d2:]
        # Broadcast cos/sin: (1, 1, seq_len, head_dim//2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, max_seq_len=8192):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.gqa_ratio = num_heads // num_kv_heads

        self.q_proj = BitLinear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = BitLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = BitLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = BitLinear(num_heads * head_dim, hidden_size, bias=False)

        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)

    def forward(self, x, offset=0, cache: Optional[KVCache] = None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V)
        q, k = self.rope(q, k, offset=offset)

        # KV-cache: append new K/V, then use full cached K/V for attention
        if cache is not None:
            cache.append(k, v)
            k, v = cache.get()  # (B, num_kv_heads, cache_len, head_dim), contiguous

        # GQA: expand k/v using expand (no memory copy) + reshape
        kv_len = k.shape[2]
        if self.gqa_ratio != 1:
            k = k.unsqueeze(2).expand(-1, -1, self.gqa_ratio, -1, -1).reshape(batch_size, self.num_heads, kv_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.gqa_ratio, -1, -1).reshape(batch_size, self.num_heads, kv_len, self.head_dim)

        # Flash / memory-efficient attention via PyTorch SDPA
        # During generation with cache: q_len=1, use no mask (single token attends to all past)
        # During training: q_len=kv_len, use is_causal=True
        use_causal = (seq_len > 1)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            is_causal=use_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    """SwiGLU MLP with fused gate+up projection.

    Instead of two separate BitLinear(hidden, mlp) calls, uses one
    BitLinear(hidden, 2*mlp) and splits, halving kernel launch overhead.
    """
    def __init__(self, hidden_size, mlp_size):
        super().__init__()
        self.gate_up_proj = BitLinear(hidden_size, 2 * mlp_size, bias=False)
        self.down_proj = BitLinear(mlp_size, hidden_size, bias=False)
        self.act = nn.SiLU()
        self.mlp_size = mlp_size

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.split(self.mlp_size, dim=-1)
        return self.down_proj(self.act(gate) * up)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, mlp_size, max_seq_len=8192):
        super().__init__()
        self.norm_attn = SubLN(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, num_kv_heads, head_dim, max_seq_len)
        self.norm_ffn = SubLN(hidden_size)
        self.mlp = MLP(hidden_size, mlp_size)

    def forward(self, x, offset=0, cache: Optional[KVCache] = None):
        x = x + self.attn(self.norm_attn(x), offset=offset, cache=cache)
        x = x + self.mlp(self.norm_ffn(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, head_dim, mlp_size, max_seq_len=8192):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Binary Embedding — STE applied in forward so effective weights are always {-1,+1}
        self.tok_embed = BinaryEmbedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, mlp_size, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm_f = SubLN(hidden_size)
        self.lm_head = BitLinear(hidden_size, vocab_size, bias=False)

    def forward(self, idx, offset=0, cache: Optional[KVCacheList] = None):
        x = self.tok_embed(idx)
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, offset=offset, cache=layer_cache)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        """Autoregressive generation with KV-cache.

        Args:
            idx: (batch, seq_len) initial token indices
            max_new_tokens: number of new tokens to generate
            temperature: sampling temperature
            top_k: if set, only sample from top-k logits

        Returns:
            (batch, seq_len + max_new_tokens) full token sequence
        """
        device = idx.device
        batch_size = idx.shape[0]

        # Create per-layer KV-cache pre-allocated for full sequence
        cache = KVCacheList(
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            batch_size=batch_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=next(self.parameters()).dtype,
            device=device,
        )

        # Prefill: process the entire prompt at once
        logits = self(idx, offset=0, cache=cache)
        next_logits = logits[:, -1, :]  # (batch, vocab_size)

        for _ in range(max_new_tokens):
            # Sample next token
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = next_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = float('-inf')
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)

            # Decode: only process the new token (cache handles the rest)
            logits = self(next_token, offset=cache.cur_len, cache=cache)
            next_logits = logits[:, -1, :]

        return idx

    def generate_stream(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        """Yield one token at a time during autoregressive generation.

        Yields:
            token_id: int — each generated token ID
        """
        device = idx.device
        batch_size = idx.shape[0]

        cache = KVCacheList(
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            batch_size=batch_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=next(self.parameters()).dtype,
            device=device,
        )

        logits = self(idx, offset=0, cache=cache)
        next_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = next_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = float('-inf')
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token[0, 0].item()
            yield token_id

            next_token_2d = next_token.unsqueeze(0) if next_token.dim() == 1 else next_token
            logits = self(next_token_2d, offset=cache.cur_len, cache=cache)
            next_logits = logits[:, -1, :]
