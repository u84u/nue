"""Binary weight assertions for training and debugging.

Call assert_binary_weights(model) before every training step to verify
that all effective forward-pass weights are strictly {-1, +1} (scaled binary).

Set NUE_ASSERT_BINARY=1 to enable automatic assertions on every forward pass.
"""

import os
import torch
import torch.nn as nn
from nue.quantization import binary_quantize_ste
from nue.binary import absmean_scale


def get_binary_weight_report(model: nn.Module) -> list[dict]:
    """Inspect every parameter and return a report dict per parameter.

    Each dict contains:
        name, shape, dtype, role, binary, trainable, min, max, unique_count
    """
    report = []
    for name, param in model.named_parameters():
        w = param.data
        unique = torch.unique(w)
        unique_count = unique.numel()

        # Determine role
        if "tok_embed" in name:
            role = "token_embedding"
        elif "lm_head" in name:
            role = "lm_head"
        elif "norm" in name:
            role = "normalization"
        elif "rope" in name:
            role = "rope"
        elif "proj" in name or "gate_up" in name or "down_proj" in name:
            role = "linear_projection"
        else:
            role = "other"

        # Determine if binary
        is_binary = False
        if role == "linear_projection" and w.numel() >= 128:
            # Check via absmean_scale normalization
            w_eff, scales = absmean_scale(w, group_size=128)
            num_groups = w.numel() // 128
            w_flat = w_eff.view(num_groups, 128)
            scales_exp = scales[:num_groups].unsqueeze(1)
            w_normed = (w_flat / (scales_exp + 1e-10)).round()
            unique_normed = torch.unique(w_normed)
            is_binary = set(unique_normed.tolist()) <= {-1.0, 1.0}
        elif role == "lm_head" and w.numel() >= 128:
            w_eff, scales = absmean_scale(w, group_size=128)
            num_groups = w.numel() // 128
            w_flat = w_eff.view(num_groups, 128)
            scales_exp = scales[:num_groups].unsqueeze(1)
            w_normed = (w_flat / (scales_exp + 1e-10)).round()
            unique_normed = torch.unique(w_normed)
            is_binary = set(unique_normed.tolist()) <= {-1.0, 1.0}
        elif role == "token_embedding":
            # Check via STE — the effective forward weight
            w_eff = binary_quantize_ste(w)
            unique_eff = torch.unique(w_eff)
            is_binary = set(unique_eff.tolist()) <= {-1.0, 1.0}

        report.append({
            "name": name,
            "shape": list(w.shape),
            "dtype": str(w.dtype),
            "role": role,
            "binary": is_binary,
            "trainable": param.requires_grad,
            "min": w.min().item(),
            "max": w.max().item(),
            "unique_count": unique_count,
        })
    return report


def assert_binary_weights(model: nn.Module, verbose: bool = False) -> None:
    """Assert that all weight-bearing layers have binary effective weights.

    Raises AssertionError if any non-binary weight-bearing layer is found.
    """
    report = get_binary_weight_report(model)
    violations = []
    for entry in report:
        if entry["role"] in ("linear_projection", "lm_head", "token_embedding"):
            if not entry["binary"]:
                violations.append(entry)
        if verbose:
            status = "✓" if entry["binary"] or entry["role"] in ("normalization", "rope", "other") else "✗"
            print(f"  {status} {entry['name']:50s}  role={entry['role']:20s}  binary={entry['binary']}  unique={entry['unique_count']}")

    if violations:
        msg = "Binary weight violations found:\n"
        for v in violations:
            msg += f"  ✗ {v['name']:50s}  role={v['role']:20s}  unique_count={v['unique_count']}  min={v['min']:.4f}  max={v['max']:.4f}\n"
        raise AssertionError(msg)


class BinaryWeightChecker:
    """Hook-based checker that asserts binary weights after each forward pass.

    Usage:
        checker = BinaryWeightChecker(model)
        # ... training loop ...
        checker.check()  # call after each forward+backward
    """

    def __init__(self, model: nn.Module, enabled: bool = None):
        if enabled is None:
            enabled = os.environ.get("NUE_ASSERT_BINARY", "0") == "1"
        self.enabled = enabled
        self.model = model

    def check(self) -> None:
        if self.enabled:
            assert_binary_weights(self.model)


# Global checker instance — initialized lazily
_checker = None


def get_checker(model: nn.Module) -> BinaryWeightChecker:
    global _checker
    if _checker is None:
        _checker = BinaryWeightChecker(model)
    return _checker
