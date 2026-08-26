import torch
import torch.optim as optim
import math
from nue.transformer import TinyTransformer
from nue.losses import TrainingLosses
from nue.weights_format import save_model_weights
from nue.assertions import assert_binary_weights, BinaryWeightChecker


def get_lr(step, warmup_steps, max_lr, min_lr, total_steps):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train():
    # Model config
    vocab_size = 1024
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    num_kv_heads = 2
    head_dim = 32
    mlp_size = 512

    model = TinyTransformer(vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, head_dim, mlp_size)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = TrainingLosses()

    # Training stability: gradient clipping and LR schedule
    max_grad_norm = 1.0
    warmup_steps = 100
    total_steps = 10000
    max_lr = 3e-4
    min_lr = 3e-5

    # Binary weight assertion (enabled via NUE_ASSERT_BINARY=1)
    checker = BinaryWeightChecker(model)

    # Dummy data
    batch, seq_len = 2, 16
    idx = torch.randint(0, vocab_size, (batch, seq_len))
    targets = torch.randint(0, vocab_size, (batch, seq_len))

    # Training step
    model.train()

    lr = get_lr(1, warmup_steps, max_lr, min_lr, total_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.zero_grad()

    logits = model(idx)
    loss = loss_fn.ntp_loss(logits, targets)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    # Verify binary weights after optimizer step
    checker.check()

    print(f"Training step complete. Loss: {loss.item():.4f}, LR: {lr:.6f}")

    # Save checkpoint
    save_model_weights(model, "final_model.weights")
    print("Checkpoint saved.")

if __name__ == "__main__":
    train()
