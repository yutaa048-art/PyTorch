import pytest
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.attention import Attention
from model.rope import precompute_freqs_cis

def test_attention_forward():
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    num_heads = 4
    
    attention = Attention(hidden_size, num_heads)
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Dummy RoPE freqs
    freqs_cis = precompute_freqs_cis(hidden_size // num_heads, seq_len * 2)
    freqs_cis_seq = freqs_cis[:seq_len]
    
    out = attention(x, freqs_cis_seq)
    
    # Memastikan dimensi output benar
    assert out.shape == (batch_size, seq_len, hidden_size)
    
    # Memastikan gradien mengalir
    out.sum().backward()
    assert attention.wq.weight.grad is not None
