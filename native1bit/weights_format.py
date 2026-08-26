"""Binary weight serialization for model.weights format.

File layout:
    Header (12 bytes):
        [0:4]   magic "N1BT"
        [4:8]   version (uint32 LE)
        [8:12]  directory_size (uint32 LE)

    Directory (variable, starts at offset 12):
        For each tensor (128 bytes per entry):
            [0:64]    name (64 bytes, null-padded UTF-8)
            [64:68]   ndim (uint32 LE)
            [68:100]  shape (8 × uint32 LE, zero-padded)
            [100:108] offset (uint64 LE)
            [108:116] byte_length (uint64 LE)
            [116:120] dtype_code (uint32 LE)
            [120:128] reserved (8 bytes, zeros)

    Data (starts at 12 + directory_size, aligned to 4096):

dtype_codes:
    0 = binary_packed  (packed bits, -1→0, +1→1)
    1 = fp32
    2 = fp16
    3 = fp32_master    (float32 master weights for STE — preserves exact scale)
"""

import struct
import torch
import numpy as np
import os
from native1bit.binary import pack_bits

MAGIC = b"N1BT"
VERSION = 1
ENTRY_SIZE = 128
ALIGNMENT = 4096

# dtype codes
DTYPE_BINARY_PACKED = 0
DTYPE_FP32 = 1
DTYPE_FP16 = 2
DTYPE_FP32_MASTER = 3

DTYPE_CODES = {
    "torch.float32": DTYPE_FP32,
    "torch.float16": DTYPE_FP16,
    "torch.bfloat16": DTYPE_FP16,
}


def _align_up(offset: int, alignment: int = ALIGNMENT) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def _pack_entry(name: bytes, shape: tuple, offset: int, byte_length: int, dtype_code: int) -> bytes:
    """Pack one directory entry into 128 bytes."""
    name_padded = name[:64].ljust(64, b"\x00")
    ndim = len(shape)
    shape_padded = list(shape) + [0] * (8 - len(shape))
    return struct.pack(
        "<64s I 8I Q Q I 8s",
        name_padded,
        ndim,
        *shape_padded[:8],
        offset,
        byte_length,
        dtype_code,
        b"\x00" * 8,
    )


def _unpack_entry(data: bytes) -> dict:
    """Unpack one 128-byte directory entry."""
    name_raw, ndim, *shape_vals, offset, byte_length, dtype_code, _reserved = struct.unpack(
        "<64s I 8I Q Q I 8s", data
    )
    name = name_raw.rstrip(b"\x00").decode("utf-8")
    shape = tuple(shape_vals[:ndim])
    return {
        "name": name,
        "shape": shape,
        "offset": offset,
        "byte_length": byte_length,
        "dtype_code": dtype_code,
    }


def _is_weight_bearing(name: str) -> bool:
    """Check if a parameter name refers to a weight-bearing tensor."""
    return any(k in name for k in ("weight", "proj", "embed", "head"))


def save_model_weights(model: torch.nn.Module, filepath: str):
    """Serialize model parameters to model.weights binary format.

    Weight-bearing tensors are saved as fp32_master (dtype_code=3) so that
    AbsMean scaling produces identical results on reload. Packed binary
    format is reserved for the final deployment checkpoint.
    """
    entries = []
    for name, param in model.named_parameters():
        entries.append((name, param.data.detach().cpu(), _is_weight_bearing(name)))

    directory_size = len(entries) * ENTRY_SIZE
    data_start = _align_up(12 + directory_size)

    directory_bytes = b""
    data_bytes = b""
    current_offset = data_start

    for name, tensor, is_wb in entries:
        if is_wb:
            # Save as fp32_master: preserves exact values for AbsMean scale
            dtype_code = DTYPE_FP32_MASTER
            raw_data = tensor.numpy().astype(np.float32).tobytes()
        else:
            # Norms and biases: save as fp32
            dtype_code = DTYPE_FP32
            raw_data = tensor.numpy().astype(np.float32).tobytes()

        entry = _pack_entry(
            name=name.encode("utf-8"),
            shape=tuple(tensor.shape),
            offset=current_offset,
            byte_length=len(raw_data),
            dtype_code=dtype_code,
        )
        directory_bytes += entry
        data_bytes += raw_data
        current_offset += len(raw_data)

    assert len(directory_bytes) == directory_size

    with open(filepath, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", directory_size))
        f.write(directory_bytes)
        current_pos = f.tell()
        if current_pos < data_start:
            f.write(b"\x00" * (data_start - current_pos))
        f.write(data_bytes)

    file_size = data_start + len(data_bytes)
    print(f"Saved {filepath}: {file_size / 1e6:.2f} MB, {len(entries)} tensors")


def load_model_weights(filepath: str) -> dict:
    """Load model weights from model.weights file.

    Returns dict mapping tensor name -> numpy array.
    Binary-packed tensors are unpacked to {-1, +1} int8 arrays.
    fp32_master tensors are loaded as float32 (identical to original).
    """
    import mmap

    with open(filepath, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        magic = mm[0:4]
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}, expected {MAGIC!r}")
        version = struct.unpack("<I", mm[4:8])[0]
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")
        directory_size = struct.unpack("<I", mm[8:12])[0]

        num_entries = directory_size // ENTRY_SIZE
        tensors = {}
        for i in range(num_entries):
            entry_start = 12 + i * ENTRY_SIZE
            entry_data = mm[entry_start : entry_start + ENTRY_SIZE]
            info = _unpack_entry(entry_data)

            raw = mm[info["offset"] : info["offset"] + info["byte_length"]]

            if info["dtype_code"] == DTYPE_BINARY_PACKED:
                packed = np.frombuffer(raw, dtype=np.uint8)
                bits = np.unpackbits(packed, bitorder="little")
                numel = 1
                for s in info["shape"]:
                    numel *= s
                bits = bits[:numel]
                arr = np.where(bits == 1, 1, -1).astype(np.int8).reshape(info["shape"])
                tensors[info["name"]] = arr
            elif info["dtype_code"] in (DTYPE_FP32, DTYPE_FP32_MASTER):
                arr = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])
                tensors[info["name"]] = arr
            elif info["dtype_code"] == DTYPE_FP16:
                arr = np.frombuffer(raw, dtype=np.float16).reshape(info["shape"])
                tensors[info["name"]] = arr
            else:
                raise ValueError(f"Unknown dtype_code: {info['dtype_code']}")

        mm.close()

    return tensors


def load_model_weights_torch(filepath: str, device: str = "cpu") -> dict:
    """Load model weights as torch tensors."""
    numpy_dict = load_model_weights(filepath)
    return {name: torch.from_numpy(arr).to(device) for name, arr in numpy_dict.items()}
