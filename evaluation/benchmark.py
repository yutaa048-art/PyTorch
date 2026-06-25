import os
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.logger import get_logger
from dataset.loader import get_dataloaders
from model.model import SentinelLM
from evaluation.perplexity import calculate_perplexity
from evaluation.next_token import calculate_next_token_accuracy

logger = get_logger("Benchmark")

def evaluate_model(config_path="config/small.yaml"):
    config = load_config(config_path)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Prioritaskan load dari checkpoints/latest.pt
    save_path = "checkpoints/latest.pt"
    if not os.path.exists(save_path):
        logger.error(f"Model checkpoint tidak ditemukan di {save_path}")
        return
        
    model = SentinelLM(config)
    
    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Backward compatibility jika disave murni state_dict
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    try:
        _, val_dl = get_dataloaders(config)
        
        logger.info("Menjalankan evaluasi komprehensif pada validation set...")
        
        perplexity = calculate_perplexity(model, val_dl, device)
        top1_acc, top5_acc = calculate_next_token_accuracy(model, val_dl, device)
        
        logger.info(f"--- Hasil Evaluasi SentinelLM ---")
        logger.info(f"Perplexity         : {perplexity:.4f}")
        logger.info(f"Top-1 Accuracy     : {top1_acc * 100:.2f}%")
        logger.info(f"Top-5 Accuracy     : {top5_acc * 100:.2f}%")
    except FileNotFoundError as e:
        logger.error(f"Dataset tidak ditemukan: {e}")
        logger.warning("Melewati evaluasi kuantitatif (Perplexity & Accuracy).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/small.yaml")
    args = parser.parse_args()
    
    evaluate_model(args.config)
