import torch
import torch.nn as nn

from .attention import Attention
from .mlp import SwiGLU
from .layernorm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Satu blok Transformer dengan:
    - Pre-Norm (RMSNorm sebelum Attention dan MLP)
    - Residual Connection
    - GQA Attention (num_kv_heads configurable)
    - LayerDrop: probabilitas kecil untuk melewati blok ini saat training.

    LayerDrop membuat model tidak bergantung pada satu layer tertentu.
    Saat inference, LayerDrop dimatikan (semua layer aktif).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        layerdrop_prob: float = 0.0
    ):
        super().__init__()
        self.layerdrop_prob = layerdrop_prob

        self.attention      = Attention(hidden_size, num_heads, num_kv_heads)
        self.mlp            = SwiGLU(hidden_size, intermediate_size)
        self.attention_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffn_norm       = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                output_attentions: bool = False):

        # LayerDrop: skip seluruh blok saat training dengan probabilitas kecil
        # Residual tetap terjaga karena kita mengembalikan x apa adanya.
        # Efek: model tidak bisa bergantung pada satu layer tertentu.
        if self.training and self.layerdrop_prob > 0.0:
            if torch.rand(1).item() < self.layerdrop_prob:
                # Kembalikan x tanpa transformasi apapun
                if output_attentions:
                    return x, None
                return x

        # Residual connection 1: Attention
        if output_attentions:
            attn_out, attn_weights = self.attention(
                self.attention_norm(x), freqs_cis, output_attentions=True
            )
            h = x + attn_out
        else:
            h = x + self.attention(self.attention_norm(x), freqs_cis)
            attn_weights = None

        # Residual connection 2: Feed Forward (SwiGLU)
        out = h + self.mlp(self.ffn_norm(h))

        if output_attentions:
            return out, attn_weights
        return out
