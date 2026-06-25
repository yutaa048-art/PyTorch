import pytest
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.rope import precompute_freqs_cis, apply_rotary_emb

def test_rope_precompute():
    dim = 16
    end = 32
    freqs = precompute_freqs_cis(dim, end)
    
    assert freqs.shape == (end, dim // 2)
    assert freqs.dtype == torch.complex64

def test_rope_apply():
    batch_size = 2
    seq_len = 8
    num_heads = 2
    head_dim = 16
    
    xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    freqs = precompute_freqs_cis(head_dim, seq_len * 2)
    freqs_seq = freqs[:seq_len]
    
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_seq)
    
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape
    
    # Karena ini rotasi, norm-nya tidak boleh berubah
    norm_before = torch.linalg.norm(xq[0, 0, 0])
    norm_after = torch.linalg.norm(xq_out[0, 0, 0])
    assert torch.isclose(norm_before, norm_after, atol=1e-5)
