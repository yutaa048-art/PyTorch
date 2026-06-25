import math
import torch
from training.loss import calculate_loss

def calculate_perplexity(model: torch.nn.Module, val_dl, device: torch.device) -> float:
    """
    Menghitung Perplexity model pada validation dataloader.
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_dl:
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = calculate_loss(logits, y)
                
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_dl)
    try:
        perplexity = math.exp(avg_val_loss)
    except OverflowError:
        perplexity = float('inf')
        
    return perplexity
