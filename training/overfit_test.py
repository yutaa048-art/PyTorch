import os
import torch
import torch.optim as optim

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import SentinelLM
from utils.config import load_config
from training.loss import calculate_loss
from utils.logger import get_logger

logger = get_logger("OverfitTest")

def run_overfit_test(config_path="config/tiny.yaml", num_steps=500):
    logger.info(f"Memulai Overfit Test menggunakan {config_path}")
    
    config = load_config(config_path)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    model = SentinelLM(config)
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3) # Learning rate lebih besar untuk overfit cepat
    
    # Buat 1 batch data statis (Dummy sequence)
    # Ini merepresentasikan 100 token data berurutan
    batch_size = 2
    seq_len = 50
    vocab_size = config.vocab_size
    
    # Input random, tapi TETAP (statis)
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    logger.info(f"Target Overfit: Menghafal 1 Batch (Size: {batch_size}, SeqLen: {seq_len})")
    
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == "cuda"))
    
    for step in range(1, num_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = calculate_loss(logits, y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % 50 == 0 or step == 1:
            logger.info(f"Step {step}/{num_steps} | Loss: {loss.item():.4f}")
            
    final_loss = loss.item()
    logger.info(f"Final Loss setelah {num_steps} langkah: {final_loss:.4f}")
    
    if final_loss < 0.1:
        logger.info("OVERFIT TEST LULUS! Arsitektur Transformer (Attention, RoPE, Masking) berfungsi sempurna.")
    else:
        logger.error("OVERFIT TEST GAGAL! Model tidak mampu menghafal 1 batch. Terdapat bug kritis pada arsitektur (Masking/Attention/Loss).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/tiny.yaml")
    args = parser.parse_args()
    
    run_overfit_test(args.config)
