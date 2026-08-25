import struct
import torch
import numpy as np
import os
import mmap
from native1bit.binary import pack_bits

# Simple header for model.weights
# Magic bytes: b"N1BT"
# Version: 1
MAGIC = b"N1BT"
VERSION = 1

def save_model_weights(model: torch.nn.Module, filepath: str):
    """
    Serializes a model with binary weights to model.weights format.
    """
    tensors_to_save = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            # Assume binary for "weight" parameters for this PoC
            # Real implementation would need to handle mixed precision
            tensors_to_save[name] = param.data.detach().cpu()
    
    with open(filepath, "wb") as f:
        # Write header
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        
        # Write tensor directory (name, shape, offset, length)
        # This is a simplified directory
        tensor_data_list = []
        directory_offset = 8
        
        # Calculate offsets
        current_offset = 1024 # Reserve space for directory
        
        for name, tensor in tensors_to_save.items():
            # For this PoC, pack everything as binary if possible
            binary_w = torch.sign(tensor)
            binary_w[binary_w == 0] = 1.0
            
            packed = pack_bits(binary_w)
            
            tensor_info = {
                "name": name.encode("utf-8"),
                "shape": tensor.shape,
                "offset": current_offset,
                "length": len(packed),
            }
            tensor_data_list.append((tensor_info, packed))
            current_offset += len(packed)
            
        # Write directory
        for info, _ in tensor_data_list:
            # name (64s), shape (2 Qs - assuming rank 2 tensors for PoC), offset (Q), length (Q)
            # Adjusting to match struct.pack arguments: name, shape_dim0, shape_dim1, offset, length
            # Shape might vary, this is PoC
            shape = info["shape"]
            # Simplified for rank 2
            f.write(struct.pack("<64sQQQQ", info["name"][:64], shape[0], shape[1] if len(shape) > 1 else 0, info["offset"], info["length"]))
            
        # Write data
        f.seek(1024)
        for _, data in tensor_data_list:
            f.write(data)
