import torch
import torch.nn as nn

from .attention import Attention
from .mlp import SwiGLU
from .layernorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, rms_norm_eps: float):
        """
        Satu blok Transformer yang terdiri dari Attention dan Feed Forward.
        Menggunakan arsitektur Pre-Norm (RMSNorm diletakkan sebelum Attention dan MLP).
        """
        super().__init__()
        self.attention = Attention(hidden_size, num_heads)
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        
        self.attention_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, output_attentions: bool = False):
        # Residual connection 1
        if output_attentions:
            attn_out, attn_weights = self.attention(self.attention_norm(x), freqs_cis, output_attentions=True)
            h = x + attn_out
        else:
            h = x + self.attention(self.attention_norm(x), freqs_cis)
            attn_weights = None
            
        # Residual connection 2
        out = h + self.mlp(self.ffn_norm(h))
        
        if output_attentions:
            return out, attn_weights
        return out
