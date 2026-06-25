import torch

def calculate_next_token_accuracy(model: torch.nn.Module, val_dl, device: torch.device):
    """
    Menghitung akurasi prediksi Next-Token pada validation dataloader.
    Mengembalikan Top-1 dan Top-5 Accuracy.
    """
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_dl:
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(x) # [batch, seq_len, vocab_size]
                
            # Top-1
            preds_top1 = torch.argmax(logits, dim=-1) # [batch, seq_len]
            top1_correct += (preds_top1 == y).sum().item()
            
            # Top-5
            _, preds_top5 = torch.topk(logits, 5, dim=-1) # [batch, seq_len, 5]
            y_expanded = y.unsqueeze(-1) # [batch, seq_len, 1]
            top5_correct += (preds_top5 == y_expanded).any(dim=-1).sum().item()
            
            total += y.numel()
            
    top1_acc = top1_correct / total if total > 0 else 0.0
    top5_acc = top5_correct / total if total > 0 else 0.0
    return top1_acc, top5_acc
