import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Menghitung precomputed frequencies untuk Rotary Positional Embedding (RoPE).
    dim: ukuran head dimension.
    end: panjang maksimum urutan (context length).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Kembalikan freqs (real) untuk menghindari bug DataParallel pada complex tensor
    return freqs

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Menyesuaikan bentuk tensor frekuensi agar bisa dibroadcast ke ukuran input.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape != (x.shape[1], x.shape[-1]):
        raise ValueError(f"Shape mismatch in RoPE: freqs_cis={freqs_cis.shape}, expected=({x.shape[1]}, {x.shape[-1]}), x.shape={x.shape}")
    
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Menerapkan RoPE ke Query dan Key.
    """
    # xq: [batch_size, seq_len, num_heads, head_dim]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
