import torch
import torch.nn.functional as F

def calculate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Menghitung CrossEntropy loss.
    logits: [batch_size, seq_len, vocab_size]
    targets: [batch_size, seq_len]
    """
    # CrossEntropyLoss pada PyTorch memerlukan input dalam bentuk [batch_size * seq_len, vocab_size]
    # dan target dalam bentuk [batch_size * seq_len]
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    return F.cross_entropy(logits_flat, targets_flat)
