import torch
import torch.nn as nn
from native1bit.binary import sign_quantize

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, group_size=128):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = group_size

    def forward(self, x):
        # 1. Quantize weights
        # STE: forward uses binary weights, backward uses gradients on master weights
        w_master = self.weight
        
        # Binary weights {-1, 1}
        w_binary = sign_quantize(w_master)
        
        # Effective weights (STE)
        w_effective = w_master + (w_binary - w_master).detach()
        
        # 2. Linear projection using binary-native weights
        return nn.functional.linear(x, w_effective, self.bias)

class SubLN(nn.Module):
    """Sub-layer normalization (RMSNorm) placed before each sub-layer."""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # RMSNorm
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * norm) * self.weight
