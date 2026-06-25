import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import SentinelConfig

from .embedding import TokenEmbedding
from .transformer_block import TransformerBlock
from .layernorm import RMSNorm
from .rope import precompute_freqs_cis

class SentinelLM(nn.Module):
    def __init__(self, config: SentinelConfig):
        """
        Arsitektur utama SentinelLM.
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.tok_embeddings = TokenEmbedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps
                )
            )
            
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings * 2, # Buffer ukuran
            theta=config.rope_theta
        )
        # Register sebagai buffer agar tidak ikut diperbarui (tidak perlu backprop)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
    def forward(self, tokens: torch.Tensor, output_attentions: bool = False):
        # tokens: [batch_size, seq_len]
        batch_size, seq_len = tokens.shape
        
        h = self.tok_embeddings(tokens)
        
        # Ambil precomputed frequencies sepanjang seq_len
        freqs_cis = self.freqs_cis[:seq_len]
        
        all_attentions = []
        for layer in self.layers:
            if output_attentions:
                h, attn = layer(h, freqs_cis, output_attentions=True)
                all_attentions.append(attn)
            else:
                h = layer(h, freqs_cis)
            
        h = self.norm(h)
        logits = self.output(h) # [batch_size, seq_len, vocab_size]
        
        if output_attentions:
            return logits, all_attentions
        return logits

if __name__ == "__main__":
    from utils.config import load_config
    config = load_config("config/tiny.yaml")
    model = SentinelLM(config)
    print(f"Total parameter: {sum(p.numel() for p in model.parameters())}")
    
    # Dummy forward pass
    dummy_input = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(dummy_input)
    print(f"Logits shape: {logits.shape}") # Harus [2, 10, 8000]
