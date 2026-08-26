"""Memory-mapped model loader for model.weights files.

Provides MMapLoader for zero-copy tensor access and load_inference_model
for reconstructing a TinyTransformer from a weights file.
"""

import mmap
import struct
import torch
import numpy as np
from nue.weights_format import (
    MAGIC, VERSION, ENTRY_SIZE,
    _unpack_entry, DTYPE_BINARY_PACKED, DTYPE_FP32, DTYPE_FP16, DTYPE_FP32_MASTER,
)
from nue.transformer import TinyTransformer


class MMapLoader:
    """Zero-copy mmap loader for model.weights files.

    Usage:
        loader = MMapLoader("model.weights")
        tensors = loader.load_all()
        loader.close()

    Or use as a context manager:
        with MMapLoader("model.weights") as loader:
            tensors = loader.load_all()
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

        # Parse header
        magic = self.mm[0:4]
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}, expected {MAGIC!r}")
        self.version = struct.unpack("<I", self.mm[4:8])[0]
        if self.version != VERSION:
            raise ValueError(f"Unsupported version: {self.version}")
        self.directory_size = struct.unpack("<I", self.mm[8:12])[0]
        self.num_entries = self.directory_size // ENTRY_SIZE

        # Parse directory entries (lazy — store offsets)
        self._entries = []
        for i in range(self.num_entries):
            entry_start = 12 + i * ENTRY_SIZE
            entry_data = self.mm[entry_start : entry_start + ENTRY_SIZE]
            self._entries.append(_unpack_entry(entry_data))

    def list_tensors(self) -> list[dict]:
        """List all tensors in the file without reading data."""
        return [
            {"name": e["name"], "shape": e["shape"], "dtype_code": e["dtype_code"]}
            for e in self._entries
        ]

    def load_tensor(self, name: str) -> np.ndarray:
        """Load a single tensor by name (zero-copy for binary-packed)."""
        for entry in self._entries:
            if entry["name"] == name:
                raw = self.mm[entry["offset"] : entry["offset"] + entry["byte_length"]]
                return self._decode(raw, entry)
        raise KeyError(f"Tensor '{name}' not found")

    def load_all(self) -> dict[str, np.ndarray]:
        """Load all tensors into a dict."""
        result = {}
        for entry in self._entries:
            raw = self.mm[entry["offset"] : entry["offset"] + entry["byte_length"]]
            result[entry["name"]] = self._decode(raw, entry)
        return result

    def _decode(self, raw: bytes, entry: dict) -> np.ndarray:
        """Decode raw bytes based on dtype_code."""
        if entry["dtype_code"] == DTYPE_BINARY_PACKED:
            packed = np.frombuffer(raw, dtype=np.uint8)
            bits = np.unpackbits(packed, bitorder="little")
            numel = 1
            for s in entry["shape"]:
                numel *= s
            bits = bits[:numel]
            return np.where(bits == 1, 1, -1).astype(np.int8).reshape(entry["shape"])
        elif entry["dtype_code"] in (DTYPE_FP32, DTYPE_FP32_MASTER):
            return np.frombuffer(raw, dtype=np.float32).reshape(entry["shape"])
        elif entry["dtype_code"] == DTYPE_FP16:
            return np.frombuffer(raw, dtype=np.float16).reshape(entry["shape"])
        else:
            raise ValueError(f"Unknown dtype_code: {entry['dtype_code']}")

    def close(self):
        self.mm.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_inference_model(
    filepath: str,
    vocab_size: int = 32768,
    hidden_size: int = 1024,
    num_layers: int = 12,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    mlp_size: int = 4096,
    device: str = "cpu",
) -> TinyTransformer:
    """Load a TinyTransformer from a model.weights file.

    Reconstructs the model architecture, then loads weights from the file.
    Binary-packed weights are loaded as int8 {-1, +1} and cast to the model's dtype.
    """
    model = TinyTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        mlp_size=mlp_size,
    )

    with MMapLoader(filepath) as loader:
        loaded = loader.load_all()

    # Load each parameter into the model
    model_state = model.state_dict()
    loaded_state = {}
    for name, arr in loaded.items():
        if name in model_state:
            target_dtype = model_state[name].dtype
            loaded_state[name] = torch.from_numpy(arr.copy()).to(target_dtype)
        else:
            print(f"Warning: loaded tensor '{name}' not found in model state_dict")

    # Load with strict=False to allow missing/extra keys
    result = model.load_state_dict(loaded_state, strict=False)
    if result.missing_keys:
        print(f"Warning: missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Warning: unexpected keys: {result.unexpected_keys}")

    return model.to(device)
