"""
Nue 250M — Binary transformer training on TinyStories.

Usage:
    PYTHONPATH=. python implementations/250M/train.py
    PYTHONPATH=. python implementations/250M/train.py --resume
    PYTHONPATH=. python implementations/250M/train.py --data /path/to/dataset.txt
"""

import torch
import torch.optim as optim
import os
import json
import time
import sys
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nue.transformer import TinyTransformer
from nue.losses import TrainingLosses
from nue.tokenizer import Tokenizer
from nue.weights_format import save_model_weights
from nue.assertions import assert_binary_weights, BinaryWeightChecker
from plot_loss import plot_loss

# Optimal thread count — beyond this, contention > benefit for this model size
torch.set_num_threads(min(8, os.cpu_count() or 8))
torch.set_float32_matmul_precision("high")

CHECKPOINT_DIR = "implementations/250M"
GENERATED_DIR = os.path.join(CHECKPOINT_DIR, "generated")
LOSS_LOG = os.path.join(GENERATED_DIR, "loss_history.csv")
TRAIN_STATE_FILE = os.path.join(GENERATED_DIR, "train_state.pt")
TOKEN_CACHE = os.path.join(GENERATED_DIR, "tokens.pt")
DEFAULT_DATA = "roneneldan/TinyStories"


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def load_text(data_source):
    """Load text from a local file or HuggingFace dataset."""
    if os.path.isfile(data_source):
        print(f"Loading local file: {data_source}")
        with open(data_source, "r") as f:
            return f.read()

    # Treat as HuggingFace dataset
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install datasets: venv/bin/pip install datasets")

    print(f"Loading HuggingFace dataset: {data_source}")
    ds = load_dataset(data_source, split="train", trust_remote_code=True)
    print(f"  {len(ds)} examples")
    if "text" in ds.column_names:
        return "\n".join(ds["text"])
    else:
        for col in ds.column_names:
            if ds[col].dtype == "string":
                return "\n".join(ds[col])
        sys.exit(f"No text column found. Columns: {ds.column_names}")


def save_train_state(filepath, model, optimizer, step, tokens_processed, epoch, loss_history):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "tokens_processed": tokens_processed,
        "epoch": epoch,
        "loss_history": loss_history,
    }, filepath)


def load_train_state(filepath, model, optimizer):
    ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"], ckpt["tokens_processed"], ckpt["epoch"], ckpt["loss_history"]


def save_loss_history(loss_history, filepath):
    with open(filepath, "w") as f:
        f.write("step,loss,tokens,lr\n")
        for entry in loss_history:
            f.write(f"{entry['step']},{entry['loss']:.6f},{entry['tokens']},{entry['lr']:.8f}\n")


def train(resume=False, data_source=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

    data_source = data_source or DEFAULT_DATA

    # 250M parameter config (Power-of-Two)
    config = {
        "vocab_size": 32768,
        "hidden_size": 1024,
        "num_layers": 12,
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 128,
        "mlp_size": 4096,
        "max_seq_len": 2048,
    }

    model = TinyTransformer(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Created: {n_params/1e6:.1f}M parameters ({n_params/1e6:.2f}M)")

    # Compile — benefits more on larger models
    model = torch.compile(model, mode="reduce-overhead")
    print("Model compiled (torch.compile, reduce-overhead)")

    # Dataset — cache tokenized data to disk
    tokenizer_file = os.path.join(GENERATED_DIR, "tokenizer.json")
    if resume and os.path.exists(TOKEN_CACHE):
        print(f"Loading cached tokens from {TOKEN_CACHE}")
        data = torch.load(TOKEN_CACHE, weights_only=True)
        tokenizer = Tokenizer.load(tokenizer_file)
        print(f"Tokenizer loaded from {tokenizer_file}")
    else:
        text = load_text(data_source)
        if os.path.exists(tokenizer_file):
            tokenizer = Tokenizer.load(tokenizer_file)
            print(f"Tokenizer loaded from {tokenizer_file}")
        else:
            tokenizer = Tokenizer(vocab_size=config["vocab_size"])
            tokenizer.train(text)
        tokens = tokenizer.encode(text)
        data = torch.tensor(tokens, dtype=torch.long)
        torch.save(data, TOKEN_CACHE)
        print(f"Tokens cached to {TOKEN_CACHE}")
    print(f"Tokens: {len(data)}")

    seq_len = 1024
    batch_size = 4  # conservative for 250M on CPU
    dataset = TextDataset(data, seq_len)
    num_workers = min(4, max(1, (os.cpu_count() or 4) // 2))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    print(f"DataLoader: {num_workers} workers, prefetch_factor=4")
    print(f"Batch: {batch_size} x {seq_len} = {batch_size * seq_len} tokens/step")

    # Training config — scaled for larger model
    max_grad_norm = 1.0
    warmup_steps = 200
    lr = 3e-4

    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        return lr

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = TrainingLosses()
    checker = BinaryWeightChecker(model)

    # Resume
    step = 0
    tokens_processed = 0
    start_epoch = 0
    loss_history = []

    if resume and os.path.exists(TRAIN_STATE_FILE):
        print(f"Resuming from {TRAIN_STATE_FILE}...")
        step, tokens_processed, start_epoch, loss_history = load_train_state(
            TRAIN_STATE_FILE, model, optimizer
        )
        print(f"  Resumed at step {step}, {tokens_processed} tokens")
        print(f"  Last loss: {loss_history[-1]['loss']:.4f}" if loss_history else "")
    else:
        print("\nPre-training weight inspection:")
        assert_binary_weights(model, verbose=False)
        print("All weight-bearing layers: BINARY ✓")

    model.train()

    pbar = tqdm(desc="Training", initial=tokens_processed)
    t_start = time.time()
    last_save = time.time()

    def save_checkpoint():
        pbar.clear()
        save_model_weights(model, os.path.join(GENERATED_DIR, "model_final.weights"))
        save_train_state(TRAIN_STATE_FILE, model, optimizer, step, tokens_processed, epoch, loss_history)
        save_loss_history(loss_history, LOSS_LOG)
        tokenizer.save(os.path.join(GENERATED_DIR, "tokenizer.json"))
        with open(os.path.join(GENERATED_DIR, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"Checkpoint saved (step {step}, loss {loss_history[-1]['loss']:.4f})")
        pbar.refresh()

    try:
        for epoch in range(start_epoch, 10_000):
            for x, y in loader:
                step += 1
                current_lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn.ntp_loss(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                checker.check()

                tokens_in_batch = x.numel()
                tokens_processed += tokens_in_batch

                loss_val = loss.item()
                loss_history.append({
                    "step": step,
                    "loss": loss_val,
                    "tokens": tokens_processed,
                    "lr": current_lr,
                })

                pbar.set_postfix({"loss": f"{loss_val:.4f}", "tokens": tokens_processed, "lr": f"{current_lr:.2e}"})
                pbar.update(tokens_in_batch)

                # Save every 5 minutes
                if time.time() - last_save > 300:
                    save_checkpoint()
                    pbar.clear()
                    try:
                        plot_loss(GENERATED_DIR)
                    except Exception as e:
                        print(f"Plot failed: {e}")
                    pbar.refresh()
                    last_save = time.time()
    except KeyboardInterrupt:
        print("\n\nInterrupted — saving...")
    finally:
        pbar.close()
        save_checkpoint()
        elapsed = time.time() - t_start
        print(f"Training: {step} steps, {elapsed:.0f}s ({elapsed/step:.1f}s/step)")
        print(f"Final loss: {loss_history[-1]['loss']:.4f}")
        print(f"Loss history: {LOSS_LOG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nue 250M training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Data source: local .txt file or HuggingFace dataset name")
    args = parser.parse_args()
    train(resume=args.resume, data_source=args.data)
