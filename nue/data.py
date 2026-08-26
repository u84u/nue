import torch

def prepare_instruct_data(tokenizer, system, user, assistant, vocab_size):
    """
    Format sequence: system + user + assistant
    Mask loss for system + user.
    """
    system_tokens = tokenizer.encode(f"system: {system}\n")
    user_tokens = tokenizer.encode(f"user: {user}\n")
    assistant_tokens = tokenizer.encode(f"assistant: {assistant}")
    
    all_tokens = system_tokens + user_tokens + assistant_tokens
    
    # Create mask: 0 for system+user, 1 for assistant
    mask = [0] * (len(system_tokens) + len(user_tokens)) + [1] * len(assistant_tokens)
    
    return torch.tensor(all_tokens), torch.tensor(mask)

def masked_ntp_loss(logits, targets, mask):
    """
    NTP loss masked to only assistant tokens.
    logits: (T, V), targets: (T), mask: (T)
    """
    import torch.nn.functional as F
    
    # Flatten
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    mask = mask.view(-1)
    
    loss = F.cross_entropy(logits, targets, reduction='none')
    
    # Apply mask
    masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    return masked_loss
