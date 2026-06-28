import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cuda

from .rope import apply_rotary_emb


class Attention(nn.Module):
    """
    Causal Multi-Head Attention dengan dukungan GQA (Grouped Query Attention).

    GQA (Grouped Query Attention):
    - Query heads tetap banyak (num_heads)
    - Key dan Value memakai lebih sedikit heads (num_kv_heads)
    - Setiap KV head dipakai bersama oleh (num_heads / num_kv_heads) Query heads
    - Manfaat: KV Cache saat inferensi turun 4x, VRAM lebih hemat

    Contoh (Deep 500M config):
    - num_heads = 16 (Q heads)
    - num_kv_heads = 4 (KV heads)
    - groups = 16 / 4 = 4 (setiap KV head dipakai 4 Q heads)
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int = None):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size harus kelipatan dari num_heads"

        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) harus kelipatan dari num_kv_heads ({self.num_kv_heads})"

        self.head_dim  = hidden_size // num_heads
        self.kv_groups = num_heads // self.num_kv_heads   # = 4 jika 16Q / 4KV

        # Q: proyeksi ke num_heads heads penuh
        self.wq = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        # K & V: proyeksi ke num_kv_heads heads saja (lebih kecil)
        self.wk = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        # Output: tetap sama
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                output_attentions: bool = False):
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x)  # [B, T, num_heads * head_dim]
        xk = self.wk(x)  # [B, T, num_kv_heads * head_dim]
        xv = self.wv(x)  # [B, T, num_kv_heads * head_dim]

        # Reshape ke per-head
        xq = xq.view(batch_size, seq_len, self.num_heads,    self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Terapkan RoPE hanya pada Q dan K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Transpose ke [B, heads, T, head_dim]
        xq = xq.transpose(1, 2)                  # [B, num_heads,    T, head_dim]
        xk = xk.transpose(1, 2)                  # [B, num_kv_heads, T, head_dim]
        xv = xv.transpose(1, 2)                  # [B, num_kv_heads, T, head_dim]

        # GQA: Expand K dan V agar cocok dengan jumlah Q heads
        # repeat_interleave: setiap KV head direplikasi sebanyak kv_groups kali
        if self.kv_groups > 1:
            xk = xk.repeat_interleave(self.kv_groups, dim=1)  # [B, num_heads, T, head_dim]
            xv = xv.repeat_interleave(self.kv_groups, dim=1)  # [B, num_heads, T, head_dim]

        if output_attentions:
            # Path manual untuk visualisasi heatmap (tidak dipakai saat training normal)
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores.float(), dim=-1).to(xq.dtype)
            output = torch.matmul(attn_weights, xv)
        else:
            # FlashAttention via SDPA: O(N) memory, 2-4x lebih cepat di T4
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,           # fallback jika flash tidak tersedia
                enable_mem_efficient=True
            ):
                output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True          # Causal mask otomatis (tanpa alokasi N×N)
                )
            attn_weights = None

        # Kembalikan ke [B, T, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.wo(output)

        if output_attentions:
            return out, attn_weights
        return out
