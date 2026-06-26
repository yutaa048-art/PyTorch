import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cuda

from .rope import apply_rotary_emb

class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        """
        Causal Multi-Head Attention.
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size harus kelipatan dari num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, output_attentions: bool = False):
        batch_size, seq_len, _ = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Terapkan RoPE (hanya pada Query dan Key)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Ubah dimensi untuk dot product: [batch_size, num_heads, seq_len, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        if output_attentions:
            # Path manual hanya untuk visualisasi heatmap (tidak digunakan saat training)
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            # Causal mask: agar token tidak melihat ke depan
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores.float(), dim=-1).to(xq.dtype)
            output = torch.matmul(attn_weights, xv)
        else:
            # FlashAttention via SDPA: O(N) memory, 2-4x lebih cepat di T4
            # enable_flash=True  → pakai FlashAttention kernel jika didukung
            # enable_mem_efficient=True → fallback ke mem-efficient kernel (T4 support)
            # is_causal=True     → causal mask otomatis tanpa alokasi matriks N×N
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,   # fallback ke math jika GPU tidak support flash
                enable_mem_efficient=True
            ):
                output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True
                )
            attn_weights = None
            
        # Kembalikan dimensi awal
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.wo(output)
        
        if output_attentions:
            return out, attn_weights
        return out
