import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        """
        Menerima input token IDs dan mengubahnya menjadi vektor embedding.
        Kita tidak memakai absolute positional embedding, karena akan memakai RoPE di layer Attention.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        # output: [batch_size, seq_len, hidden_size]
        return self.embedding(x)
