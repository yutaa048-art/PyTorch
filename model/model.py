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
    """
    SentinelLM — Multi-Level Thinking Transformer (500M Deep)

    Fitur:
    ─────
    ✅ GQA (Grouped Query Attention)     — KV Cache 4x lebih hemat
    ✅ LayerDrop                         — Regularisasi antar-layer
    ✅ Dynamic RoPE                      — Konteks panjang lebih stabil
    ✅ Multi-Level Thinking              — 5 Head di 5 titik kedalaman
    ✅ Confidence-Weighted Aux Loss      — Label tidak dipercaya 100%

    Arsitektur forward:
    ─────────────────
    tokens → Embedding → Transformer × 32
           → [L8]  Syntax Head       (4-class)
           → [L12] Concept Head      (6-class)
           → [L16] Semantic Head     (4-class)
           → [L24] Architecture Head (7-class)
           → [L32] Reasoning Head    (10-class) + RMSNorm → LM Head

    Multi-Level Thinking Heads (Hanya aktif saat training):
    ───────────────────────────────────────────────────────
    • syntax_head       (L8)  : Teks, Deklarasi, Flow, Kompleks
    • concept_head      (L12) : Python, JS, Systems, JVM, Infra, NatLang
    • semantic_head     (L16) : Netral, IO, Algo, Security
    • architecture_head (L24) : Standalone, MVC, Clean, Repo, Micro, Event, API
    • reasoning_head    (L32) : None,DataFlow,CtrlFlow,Auth,Concur,Mem,Proto,Parse,Opt,SecReason
    """

    def __init__(self, config: SentinelConfig):
        super().__init__()
        self.config = config

        # Hyperparameter utama
        vocab_size      = config.vocab_size
        hidden_size     = config.hidden_size
        num_heads       = config.num_attention_heads
        num_kv_heads    = getattr(config, 'num_key_value_heads', num_heads)
        layerdrop_prob  = getattr(config, 'layerdrop', 0.0)
        num_layers      = config.num_hidden_layers
        intermediate    = config.intermediate_size

        # ── Embedding ──────────────────────────────────────────────
        self.tok_embeddings = TokenEmbedding(vocab_size, hidden_size)

        # ── Transformer Layers ─────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size     = hidden_size,
                num_heads       = num_heads,
                num_kv_heads    = num_kv_heads,
                intermediate_size = intermediate,
                rms_norm_eps    = config.rms_norm_eps,
                layerdrop_prob  = layerdrop_prob
            )
            for _ in range(num_layers)
        ])

        # ── Final Norm + LM Head utama ─────────────────────────────
        self.norm   = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

        # ── RoPE: precompute frequencies ───────────────────────────
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        freqs = precompute_freqs_cis(
            hidden_size // num_heads,
            config.max_position_embeddings * 2,
            theta=rope_theta
        )
        self.register_buffer("freqs", freqs, persistent=False)

        # ── Multi-Level Thinking Objective Heads ───────────────────
        # 5 titik hook: L8, L12, L16, L24, L32
        self.mlt_indices = {}
        if num_layers >= 8:
            chunk = num_layers // 4   # = 8 untuk 32-layer model
            self.mlt_indices['syntax']       = chunk - 1           # idx 7  (L8)
            self.mlt_indices['concept']      = (chunk + chunk // 2) - 1  # idx 11 (L12)
            self.mlt_indices['semantic']     = (chunk * 2) - 1     # idx 15 (L16)
            self.mlt_indices['architecture'] = (chunk * 3) - 1     # idx 23 (L24)
            self.mlt_indices['reasoning']    = num_layers - 1      # idx 31 (L32)

        self.syntax_head       = nn.Linear(hidden_size, 4,  bias=False)
        self.concept_head      = nn.Linear(hidden_size, 6,  bias=False)
        self.semantic_head     = nn.Linear(hidden_size, 4,  bias=False)
        self.architecture_head = nn.Linear(hidden_size, 7,  bias=False)
        self.reasoning_head    = nn.Linear(hidden_size, 10, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        output_attentions: bool = False,
        return_aux: bool = False
    ):
        """
        Args:
            tokens          : [batch, seq_len] token IDs
            output_attentions: Kembalikan bobot attention
            return_aux      : Kembalikan semua aux logits (untuk training)

        Returns:
            Jika return_aux=False: logits [B, T, vocab]
            Jika return_aux=True : (logits, aux_dict)
                aux_dict berisi:
                    'syntax':       [B, 4]
                    'concept':      [B, 6]
                    'semantic':     [B, 4]
                    'architecture': [B, 7]
                    'reasoning':    [B, 10]
        """
        batch_size, seq_len = tokens.shape

        # Embedding
        h = self.tok_embeddings(tokens)

        # RoPE frequencies
        freqs    = self.freqs[:seq_len]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        aux_logits = {}
        all_attentions = [] if output_attentions else None

        # Head mapping untuk loop yang bersih
        head_map = {
            'syntax':       self.syntax_head,
            'concept':      self.concept_head,
            'semantic':     self.semantic_head,
            'architecture': self.architecture_head,
            'reasoning':    self.reasoning_head,
        }

        for i, layer in enumerate(self.layers):
            if output_attentions:
                h, attn = layer(h, freqs_cis, output_attentions=True)
                all_attentions.append(attn)
            else:
                h = layer(h, freqs_cis)

            # Multi-Level Thinking: tangkap state di batas zona
            if return_aux:
                for head_name, head_idx in self.mlt_indices.items():
                    if i == head_idx:
                        aux_logits[head_name] = head_map[head_name](h.mean(dim=1))

        # Final norm + LM Head utama
        h_normed = self.norm(h)
        logits   = self.output(h_normed)

        if return_aux:
            if output_attentions:
                return logits, aux_logits, all_attentions
            return logits, aux_logits

        if output_attentions:
            return logits, all_attentions
        return logits


if __name__ == "__main__":
    from utils.config import load_config

    for cfg_path in ["config/500m_deep.yaml", "config/small.yaml"]:
        try:
            config = load_config(cfg_path)
            model  = SentinelLM(config)
            total  = sum(p.numel() for p in model.parameters())
            print(f"\n[{cfg_path}]")
            print(f"  Total params  : {total:,} ({total/1e6:.1f}M)")
            print(f"  mlt_indices   : {model.mlt_indices}")
            print(f"  GQA heads     : Q={config.num_attention_heads} "
                  f"KV={getattr(config, 'num_key_value_heads', config.num_attention_heads)}")

            # Test forward — normal
            dummy = torch.randint(0, config.vocab_size, (2, 16))
            logits = model(dummy)
            print(f"  logits shape  : {logits.shape}")

            # Test forward — dengan aux
            logits, aux = model(dummy, return_aux=True)
            print(f"  aux keys      : {list(aux.keys())}")
            for k, v in aux.items():
                print(f"    {k:>15}: {v.shape}")
        except FileNotFoundError:
            print(f"  [skip] {cfg_path} tidak ditemukan")
