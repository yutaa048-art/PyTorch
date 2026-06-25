import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        """
        SwiGLU Feed Forward Network.
        Arsitektur FFN modern yang lebih efisien dari pada ReLU MLP standar.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu ekuivalen dengan fungsi aktivasi Swish
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
