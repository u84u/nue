#!/usr/bin/env python3
"""Inspect all model parameters and verify binary weight compliance.

Usage:
    python scripts/inspect_weights.py [--verbose]

Run before training and before every checkpoint write (per SPEC §24).
"""

import sys
import os
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nue.transformer import TinyTransformer
from nue.assertions import get_binary_weight_report, assert_binary_weights


def print_report(model: torch.nn.Module, verbose: bool = False) -> None:
    report = get_binary_weight_report(model)

    print()
    print("=" * 90)
    print("MODEL WEIGHT INSPECTION REPORT")
    print("=" * 90)
    print(f"{'Name':<50s} {'Shape':<16s} {'Dtype':<12s} {'Role':<20s} {'Binary':<8s} {'Unique':<8s}")
    print("-" * 90)

    total_params = 0
    binary_params = 0
    non_binary_weight_bearing = 0

    for entry in report:
        shape_str = str(entry["shape"])
        trainable_str = "" if entry["trainable"] else " [frozen]"
        binary_str = "✓" if entry["binary"] else (
            "N/A" if entry["role"] in ("normalization", "rope", "other") else "✗"
        )

        print(f"{entry['name']:<50s} {shape_str:<16s} {entry['dtype']:<12s} "
              f"{entry['role']:<20s} {binary_str:<8s} {entry['unique_count']:<8d}{trainable_str}")

        numel = 1
        for s in entry["shape"]:
            numel *= s
        total_params += numel

        if entry["role"] in ("linear_projection", "lm_head", "token_embedding"):
            binary_params += numel
            if not entry["binary"]:
                non_binary_weight_bearing += numel

        if verbose and entry["unique_count"] <= 10:
            print(f"  {'':50s} values: {torch.unique(torch.zeros(1)).tolist()}")

    print("-" * 90)
    print(f"Total parameters:       {total_params:>12,d} ({total_params / 1e6:.2f}M)")
    print(f"Weight-bearing params:  {binary_params:>12,d} ({binary_params / 1e6:.2f}M)")
    if non_binary_weight_bearing > 0:
        print(f"NON-BINARY violation:   {non_binary_weight_bearing:>12,d} ({non_binary_weight_bearing / 1e6:.2f}M)")
    else:
        print(f"All weight-bearing:     BINARY ✓")
    print("=" * 90)


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    large = "--125m" in sys.argv

    if large:
        print("Creating 125M model...")
        model = TinyTransformer(
            vocab_size=32768, hidden_size=1024, num_layers=12,
            num_heads=8, num_kv_heads=2, head_dim=128, mlp_size=4096,
        )
    else:
        print("Creating small model (use --125m for full size)...")
        model = TinyTransformer(
            vocab_size=1024, hidden_size=64, num_layers=2,
            num_heads=4, num_kv_heads=2, head_dim=16, mlp_size=256,
        )

    print_report(model, verbose=verbose)

    # Run assertion
    print()
    try:
        assert_binary_weights(model, verbose=verbose)
        print("Binary weight assertion: PASSED ✓")
    except AssertionError as e:
        print(f"Binary weight assertion: FAILED ✗")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
