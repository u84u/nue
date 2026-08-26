import torch
import torch.nn as nn
from native1bit.binary import sign_quantize, absmean_scale

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, group_size=128):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = group_size

    def forward(self, x):
        # 1. Quantize weights with per-group AbsMean scaling
        w_master = self.weight
        
        # Binary weights {-1, 1} with per-group AbsMean scale
        # Effective weights w_effective = scales * sign(w_master_group)
        w_effective, _ = absmean_scale(w_master, group_size=self.group_size)
        
        # STE for backward pass: forward uses w_effective, backward uses gradients on w_master
        # The trick here is that w_effective itself is an approximation.
        # Standard STE: w_eff = w_master + (w_binary - w_master).detach()
        # With per-group scale, it becomes more complex. 
        # For this PoC, using the scaled effective weights directly is standard.
        
        # 2. Linear projection using scaled binary-native weights
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
