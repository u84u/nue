#!/usr/bin/env python3
"""Interactive REPL for running a trained Nue model.    Usage:
    python run.py <checkpoint_dir>
    python run.py implementations/1M/generated

Loads model weights, tokenizer, and config from the checkpoint directory,
then enters an interactive loop for text generation.

Commands:
    /temp <value>    Set temperature (default: 0.8)
    /topk <value>    Set top-k (default: None)
    /tokens <value>  Set max new tokens (default: 256)
    /quit            Exit
"""

import sys
import os
import json
import time
import torch

from nue.transformer import TinyTransformer, KVCacheList
from nue.tokenizer import Tokenizer
from nue.mmap_loader import load_inference_model


def load_model(checkpoint_dir: str):
    """Load model, tokenizer, and config from a checkpoint directory."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "model_final.weights")
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")

    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {checkpoint_dir}")
        sys.exit(1)
    if not os.path.exists(weights_path):
        print(f"Error: model_final.weights not found in {checkpoint_dir}")
        sys.exit(1)
    if not os.path.exists(tokenizer_path):
        print(f"Error: tokenizer.json not found in {checkpoint_dir}")
        print("Re-run training to save the tokenizer.")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    print(f"Loading model ({config})...")
    model = load_inference_model(
        weights_path,
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_kv_heads=config["num_kv_heads"],
        head_dim=config["head_dim"],
        mlp_size=config["mlp_size"],
        max_seq_len=config.get("max_seq_len", 8192),
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params / 1e6:.2f}M params, {os.path.getsize(weights_path) / 1e6:.2f} MB")

    print("Loading tokenizer...")
    tokenizer = Tokenizer.load(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, {len(tokenizer.decoder)} tokens")

    return model, tokenizer, config


def generate_stream(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_k: int | None):
    """Stream tokens one at a time. Yields (token_text, token_id)."""
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        print("Warning: prompt encoded to empty token list.")
        return

    idx = torch.tensor([token_ids], dtype=torch.long)

    with torch.inference_mode():
        for token_id in model.generate_stream(idx, max_new_tokens, temperature, top_k):
            text = tokenizer.decode([token_id])
            yield text, token_id


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <checkpoint_dir>")
        print("Example: python run.py implementations/1M")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    model, tokenizer, config = load_model(checkpoint_dir)

    # Compile for inference speedup (~7x on single-token decode)
    print("Compiling model for inference...")
    model = torch.compile(model, mode="reduce-overhead")
    # Trigger compilation with a warmup pass
    _ = model(torch.randint(0, config["vocab_size"], (1, 1)))
    print("Compilation done.")

    # Generation defaults
    temperature = 0.8
    top_k = None
    max_new_tokens = 256

    print("\n" + "=" * 50)
    print("Nue REPL — type a prompt to generate text")
    print("Commands: /temp, /topk, /tokens, /quit")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        # Handle commands
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()

            if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                print("Bye!")
                break
            elif cmd == "/temp" and len(parts) > 1:
                temperature = float(parts[1])
                print(f"Temperature set to {temperature}")
            elif cmd == "/topk" and len(parts) > 1:
                val = parts[1].lower()
                top_k = None if val in ("none", "off", "0") else int(val)
                print(f"Top-k set to {top_k}")
            elif cmd == "/tokens" and len(parts) > 1:
                max_new_tokens = int(parts[1])
                print(f"Max tokens set to {max_new_tokens}")
            else:
                print(f"Unknown command: {cmd}")
                print("Available: /temp <v>, /topk <v>, /tokens <v>, /quit")
            continue

        # Generate with streaming
        t0 = time.perf_counter()
        gen_count = 0
        for token_text, _ in generate_stream(model, tokenizer, prompt, max_new_tokens, temperature, top_k):
            sys.stdout.write(token_text)
            sys.stdout.flush()
            gen_count += 1
        elapsed = time.perf_counter() - t0
        if gen_count > 0:
            tok_per_sec = gen_count / elapsed if elapsed > 0 else 0
            print(f"\n\n--- {gen_count} tokens | {elapsed:.2f}s | {tok_per_sec:.1f} tok/s ---")
        print()


if __name__ == "__main__":
    main()
