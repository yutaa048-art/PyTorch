import os
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from utils.logger import get_logger
from tokenizer.encode import encode

logger = get_logger("Preprocess")

def preprocess_dataset(config_path="config/small.yaml"):
    config = load_config(config_path)
    base_dir = os.path.dirname(config.data_path)
    
    # Karena data sekarang dipisah per kategori, kita encode per folder
    categories = ["code", "docs", "security", "writeups"]
    found_any_category = False
    
    for cat in categories:
        data_file = os.path.join(base_dir, cat, "data.txt")
        if not os.path.exists(data_file):
            continue
            
        found_any_category = True
        logger.info(f"Membaca {data_file}...")
        with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            
        logger.info(f"Meng-encode dataset {cat} (ini bisa memakan waktu)...")
        token_ids = encode(text, config)
        
        # Split train & val
        split_idx = int(len(token_ids) * config.train_split)
        train_ids = token_ids[:split_idx]
        val_ids = token_ids[split_idx:]
        
        train_tensor = torch.tensor(train_ids, dtype=torch.long)
        val_tensor = torch.tensor(val_ids, dtype=torch.long)
        
        train_path = os.path.join(base_dir, cat, "train.pt")
        val_path = os.path.join(base_dir, cat, "val.pt")
        
        torch.save(train_tensor, train_path)
        torch.save(val_tensor, val_path)
        
        logger.info(f"[{cat.upper()}] Total tokens: {len(token_ids)}")
        logger.info(f"[{cat.upper()}] Disimpan: {train_path} ({len(train_tensor)} tokens), {val_path} ({len(val_tensor)} tokens)")

    # Fallback jika tidak ada struktur kategori (seperti corpus_v0.1.txt tunggal)
    if not found_any_category:
        if os.path.exists(config.data_path):
            logger.info(f"Struktur kategori tidak ditemukan. Membaca file tunggal {config.data_path}...")
            with open(config.data_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                
            logger.info("Meng-encode dataset tunggal (ini bisa memakan waktu)...")
            token_ids = encode(text, config)
            
            split_idx = int(len(token_ids) * config.train_split)
            train_tensor = torch.tensor(token_ids[:split_idx], dtype=torch.long)
            val_tensor = torch.tensor(token_ids[split_idx:], dtype=torch.long)
            
            train_path = os.path.join(base_dir, "train.pt")
            val_path = os.path.join(base_dir, "val.pt")
            
            torch.save(train_tensor, train_path)
            torch.save(val_tensor, val_path)
            
            logger.info(f"[TUNGGAL] Total tokens: {len(token_ids)}")
            logger.info(f"[TUNGGAL] Disimpan: {train_path} ({len(train_tensor)} tokens), {val_path} ({len(val_tensor)} tokens)")
        else:
            logger.error(f"File {config.data_path} tidak ditemukan!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/small.yaml")
    args = parser.parse_args()
    preprocess_dataset(args.config)
