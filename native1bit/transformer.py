import torch
import torch.nn as nn
from native1bit.layers import BitLinear, SubLN
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = BitLinear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = BitLinear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = BitLinear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = BitLinear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Causal mask
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_size):
        super().__init__()
        self.gate_proj = BitLinear(hidden_size, mlp_size, bias=False)
        self.up_proj = BitLinear(hidden_size, mlp_size, bias=False)
        self.down_proj = BitLinear(mlp_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, mlp_size):
        super().__init__()
        self.norm_attn = SubLN(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, head_dim)
        self.norm_ffn = SubLN(hidden_size)
        self.mlp = MLP(hidden_size, mlp_size)

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_ffn(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, head_dim, mlp_size):
        super().__init__()
        # Binary Embedding
        self.tok_embed = nn.Embedding(vocab_size, hidden_size)
        # Manually binarize initial weights
        with torch.no_grad():
            self.tok_embed.weight.data = torch.sign(self.tok_embed.weight.data)
            self.tok_embed.weight.data[self.tok_embed.weight.data == 0] = 1.0
            
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, head_dim, mlp_size)
            for _ in range(num_layers)
        ])
        self.norm_f = SubLN(hidden_size)
        self.lm_head = BitLinear(hidden_size, vocab_size, bias=False)

    def forward(self, idx):
        x = self.tok_embed(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)
