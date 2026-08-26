#!/usr/bin/env python3
"""Regenerate tokenizer.json and config.json for a trained checkpoint.

Use this when you trained before the config/tokenizer saving was added.

Usage:
    python regenerate_checkpoint.py implementations/1M
"""

import sys
import os
import json
from nue.tokenizer import Tokenizer

# Known model configs by directory name
KNOWN_CONFIGS = {
    "implementations/1M": {
        "vocab_size": 1024,
        "hidden_size": 128,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 32,
        "mlp_size": 512,
    },
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python regenerate_checkpoint.py <checkpoint_dir>")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]

    # Save config.json
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        print(f"config.json already exists in {checkpoint_dir}")
    else:
        if checkpoint_dir not in KNOWN_CONFIGS:
            print(f"Error: no known config for '{checkpoint_dir}'")
            print(f"Known configs: {list(KNOWN_CONFIGS.keys())}")
            sys.exit(1)
        config = KNOWN_CONFIGS[checkpoint_dir]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved {config_path}")

    # Save tokenizer.json
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        print(f"tokenizer.json already exists in {checkpoint_dir}")
    else:
        # Find training text
        txt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".txt")]
        if not txt_files:
            print(f"Error: no .txt training data found in {checkpoint_dir}")
            sys.exit(1)

        with open(os.path.join(checkpoint_dir, txt_files[0])) as f:
            text = f.read()

        with open(config_path) as f:
            config = json.load(f)

        print(f"Training tokenizer on {txt_files[0]} (vocab_size={config['vocab_size']})...")
        tokenizer = Tokenizer(vocab_size=config["vocab_size"])
        tokenizer.train(text)
        tokenizer.save(tokenizer_path)
        print(f"Saved {tokenizer_path}")


if __name__ == "__main__":
    main()
