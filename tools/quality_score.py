import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.encode import encode
from utils.config import load_config

SECURITY_KEYWORDS = [
    "Prompt Injection",
    "Tool Calling",
    "FastAPI",
    "LangChain",
    "MCP",
    "Function Calling",
    "OpenAI SDK",
    "Anthropic SDK",
    "JWT",
    "Authorization",
    "Bearer"
]

def analyze_quality(corpus_dir: str, config_path: str):
    config = load_config(config_path)
    
    total_chars = 0
    total_tokens = 0
    used_vocab = set()
    
    # Context distribution
    context_buckets = {100: 0, 500: 0, 1000: 0, 4000: 0}
    
    # Security coverage
    security_counts = {kw: 0 for kw in SECURITY_KEYWORDS}
    
    # Karena memproses 200MB sekaligus lambat, kita iterasi per blok (file)
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue
                
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            total_chars += len(content)
            
            # Hitung keyword
            content_lower = content.lower()
            for kw in SECURITY_KEYWORDS:
                security_counts[kw] += content_lower.count(kw.lower())
                
            # Tokenization analysis per blok file
            blocks = content.split("# --- FILE: ")
            for block in blocks:
                if len(block) < 10: continue
                
                try:
                    # Tokenisasi (ini memakan waktu, tapi penting untuk statistik)
                    ids = encode(block, config)
                    total_tokens += len(ids)
                    used_vocab.update(ids)
                    
                    # Context length
                    length = len(ids)
                    if length <= 100: context_buckets[100] += 1
                    elif length <= 500: context_buckets[500] += 1
                    elif length <= 1000: context_buckets[1000] += 1
                    else: context_buckets[4000] += 1
                except Exception:
                    pass
                    
    avg_token_len = total_chars / total_tokens if total_tokens > 0 else 0
    vocab_usage = len(used_vocab) / config.vocab_size * 100
    
    total_docs = sum(context_buckets.values())
    context_dist = {}
    if total_docs > 0:
        for k, v in context_buckets.items():
            context_dist[k] = (v / total_docs) * 100
            
    return {
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "avg_token_len": avg_token_len,
        "vocab_usage": vocab_usage,
        "context_dist": context_dist,
        "security_counts": security_counts
    }
