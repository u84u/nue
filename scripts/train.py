import torch
import torch.optim as optim
from native1bit.transformer import TinyTransformer
from native1bit.losses import TrainingLosses
from native1bit.weights_format import save_model_weights

def train():
    # Model config
    vocab_size = 1024
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    head_dim = 32
    mlp_size = 512
    
    model = TinyTransformer(vocab_size, hidden_size, num_layers, num_heads, head_dim, mlp_size)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = TrainingLosses()
    
    # Dummy data
    batch, seq_len = 2, 16
    idx = torch.randint(0, vocab_size, (batch, seq_len))
    targets = torch.randint(0, vocab_size, (batch, seq_len))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    logits = model(idx)
    loss = loss_fn.ntp_loss(logits, targets)
    
    loss.backward()
    optimizer.step()
    
    print(f"Training step complete. Loss: {loss.item()}")
    
    # Save checkpoint
    save_model_weights(model, "final_model.weights")
    print("Checkpoint saved.")

if __name__ == "__main__":
    train()
