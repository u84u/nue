import torch
import numpy as np
from native1bit.quantization import binary_quantize_ste

def sign_quantize(w: torch.Tensor) -> torch.Tensor:
    return binary_quantize_ste(w)

def absmean_scale(w: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes AbsMean scale factor per group and returns scaled binary weights.
    Uses STE for gradient flow.
    """
    original_shape = w.shape
    num_elements = w.numel()
    
    # Pad in-place on a flat view (no cat allocation if divisible)
    w_flat = w.reshape(-1)
    padding = (group_size - (num_elements % group_size)) % group_size
    if padding > 0:
        w_padded = torch.zeros(num_elements + padding, device=w.device, dtype=w.dtype)
        w_padded[:num_elements] = w_flat
        w_flat = w_padded
    
    groups = w_flat.view(-1, group_size)
    
    # AbsMean per group — use torch.mean(abs) directly
    scales = torch.mean(torch.abs(groups), dim=1).detach()
    
    # Sign quantize with STE
    binary_weights = binary_quantize_ste(groups)
    
    # Scale and reshape back, trimming padding
    effective_weights = (binary_weights * scales.unsqueeze(1)).view(-1)[:num_elements]
    
    return effective_weights.view(original_shape), scales

def pack_bits(binary_tensor: torch.Tensor) -> bytes:
    """
    Packs {-1, 1} tensor into dense bits.
    Maps -1 -> 0, 1 -> 1.
    Uses bitwise OR for efficient packing.
    """
    # Map to {0, 1} — avoids float division, uses int shift
    bits = ((binary_tensor.view(-1).to(torch.int8) + 1) >> 1).to(torch.uint8)
    
    # Pad to multiple of 8
    num_bits = bits.numel()
    padding = (8 - (num_bits % 8)) % 8
    if padding > 0:
        bits = torch.cat([bits, torch.zeros(padding, dtype=torch.uint8, device=bits.device)])
    
    # Reshape to (N/8, 8) and pack with bitwise shift+or (faster than multiply+sum)
    bits_reshaped = bits.view(-1, 8)
    powers_of_two = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=bits.device)
    packed = (bits_reshaped * powers_of_two).sum(dim=1)
    
    return packed.to(torch.uint8).cpu().numpy().tobytes()
