import os
import torch
import torch.nn.functional as F

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from model.model import SentinelLM
from tokenizer.encode import encode
from tokenizer.decode import decode
from utils.logger import get_logger

logger = get_logger("Generate")

def generate_text(prompt: str, config, model, device, max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = 10):
    model.eval()
    
    # SentencePiece encode mengembalikan scalar int
    input_ids = encode(prompt, config)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    logger.info(f"Prompt: '{prompt}'")
    logger.info("Generating teks...")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -config.max_position_embeddings:]
            logits = model(x_cond)
            
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, next_token), dim=1)
            
    output_ids = x[0].tolist()
    output_text = decode(output_ids, config)
    
    return output_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="GET /")
    parser.add_argument("--config", type=str, default="config/small.yaml")
    parser.add_argument("--max_tokens", type=int, default=100)
    args = parser.parse_args()
    
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "checkpoints/latest.pt"
    if not os.path.exists(save_path):
        logger.error(f"Model checkpoint tidak ditemukan di {save_path}")
        sys.exit(1)
        
    model = SentinelLM(config)
    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    
    output = generate_text(args.prompt, config, model, device, max_new_tokens=args.max_tokens)
    print("\n--- HASIL GENERASI ---")
    print(output)
    print("----------------------\n")
