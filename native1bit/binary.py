import torch
import numpy as np

def sign_quantize(w: torch.Tensor) -> torch.Tensor:
    """
    Binary quantization: maps to {-1, +1}
    0.0 is mapped to 1.0 (torch.sign(0)=0, map 0 to 1)
    """
    s = torch.sign(w)
    s[s == 0] = 1.0
    return s

def absmean_scale(w: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes AbsMean scale factor per group and returns scaled binary weights.
    """
    # Flatten weights to 1D
    original_shape = w.shape
    w_flat = w.view(-1)
    
    # Pad if not divisible by group_size
    num_elements = w_flat.numel()
    padding = (group_size - (num_elements % group_size)) % group_size
    if padding > 0:
        w_flat = torch.cat([w_flat, torch.zeros(padding, device=w.device)])
        
    # Reshape into groups
    num_groups = (w_flat.numel() + group_size - 1) // group_size
    groups = w_flat.view(-1, group_size)
    
    # Compute AbsMean per group
    scales = torch.mean(torch.abs(groups), dim=1)
    
    # Sign quantize
    binary_weights = torch.sign(groups)
    # Ensure no zero, map to {-1, 1}
    binary_weights[binary_weights == 0] = 1.0
    
    # Scale back
    effective_weights = binary_weights * scales.view(-1, 1)
    
    # Reshape back to flat and remove padding
    effective_flat = effective_weights.view(-1)[:num_elements]
    
    return effective_flat.view(original_shape), scales

def pack_bits(binary_tensor: torch.Tensor) -> bytes:
    """
    Packs {-1, 1} tensor into dense bits.
    Maps -1 -> 0, 1 -> 1.
    """
    # Map to {0, 1}
    # binary_tensor is {-1, 1}
    # bits = (tensor + 1) / 2
    bits = ((binary_tensor.view(-1) + 1) / 2).to(torch.uint8)
    
    # Pad to multiple of 8
    num_bits = bits.numel()
    padding = (8 - (num_bits % 8)) % 8
    if padding > 0:
        bits = torch.cat([bits, torch.zeros(padding, dtype=torch.uint8, device=bits.device)])
        
    # Reshape to (N/8, 8)
    bits_reshaped = bits.view(-1, 8)
    
    # Pack bits (little-endian: col 0 is bit 0, col 7 is bit 7)
    powers_of_two = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=bits.device)
    packed = (bits_reshaped * powers_of_two).sum(dim=1)
    
    # Ensure it's a numpy array of uint8 before tobytes
    return packed.cpu().numpy().astype(np.uint8).tobytes()
