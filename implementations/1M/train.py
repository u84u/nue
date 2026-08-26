import torch
import torch.optim as optim
import math
import os
import json
import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nue.transformer import TinyTransformer
from nue.losses import TrainingLosses
from nue.tokenizer import Tokenizer
from nue.weights_format import save_model_weights
from nue.assertions import assert_binary_weights, BinaryWeightChecker

CHECKPOINT_DIR = "implementations/1M"
LOSS_LOG = os.path.join(CHECKPOINT_DIR, "loss_history.csv")
TRAIN_STATE_FILE = os.path.join(CHECKPOINT_DIR, "train_state.pt")


class ShakespeareDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
    def __len__(self):
        return len(self.tokens) - self.seq_len
    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


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


def train(resume=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1M parameter config (Power-of-Two)
    config = {
        "vocab_size": 1024,
        "hidden_size": 128,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 32,
        "mlp_size": 512,
    }

    model = TinyTransformer(**config)
    print(f"Model Created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Dataset
    with open("implementations/1M/shakespeare.txt", "r") as f:
        text = f.read()

    tokenizer = Tokenizer(vocab_size=config["vocab_size"])
    tokenizer.train(text)

    tokens = tokenizer.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)
    print(f"Tokens: {len(data)}")

    seq_len = 256
    batch_size = 8
    dataset = ShakespeareDataset(data, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Training config
    max_grad_norm = 1.0
    warmup_steps = 100
    max_lr = 3e-4
    min_lr = 1e-5
    total_steps = 2000

    def get_lr(step):
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    optimizer = optim.Adam(model.parameters(), lr=max_lr)
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

    save_every_steps = 200

    pbar = tqdm(desc="Training", total=total_steps * seq_len * batch_size, initial=tokens_processed)
    t_start = time.time()

    for epoch in range(start_epoch, 100):
        for x, y in loader:
            step += 1
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
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
                "lr": lr,
            })

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "tokens": tokens_processed, "lr": f"{lr:.2e}"})
            pbar.update(tokens_in_batch)

            if step % save_every_steps == 0:
                save_train_state(TRAIN_STATE_FILE, model, optimizer, step, tokens_processed, epoch, loss_history)
                save_loss_history(loss_history, LOSS_LOG)

            if step >= total_steps:
                break
        if step >= total_steps:
            break

    elapsed = time.time() - t_start
    pbar.close()

    save_model_weights(model, os.path.join(CHECKPOINT_DIR, "model_final.weights"))
    save_train_state(TRAIN_STATE_FILE, model, optimizer, step, tokens_processed, epoch, loss_history)
    save_loss_history(loss_history, LOSS_LOG)

    print(f"\nTraining complete: {step} steps, {elapsed:.0f}s ({elapsed/step:.1f}s/step)")
    print(f"Final loss: {loss_history[-1]['loss']:.4f}")
    print(f"Loss history saved to {LOSS_LOG}")
    print(f"Final checkpoint saved.")


if __name__ == "__main__":
    import sys
    resume = "--resume" in sys.argv
    train(resume=resume)
