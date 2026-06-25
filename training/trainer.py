import os
import csv
import yaml
import torch
import glob
from datetime import datetime
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed
from dataset.loader import get_dataloaders
from model.model import SentinelLM
from training.loss import calculate_loss
from training.optimizer import create_optimizer
from training.scheduler import CosineLRScheduler

logger = get_logger("Trainer")

def setup_experiment(config):
    """Membuat folder eksperimen berurutan (exp001, exp002, dst)."""
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    exp_dirs = glob.glob("experiments/exp*")
    exp_id = len(exp_dirs) + 1
    exp_dir = f"experiments/exp{exp_id:03d}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)
        
    return exp_dir

def train(config_path="config/small.yaml"):
    config = load_config(config_path)
    set_seed(42)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Menggunakan device: {device}")
    
    # Setup AMP
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
    
    train_dl, val_dl = get_dataloaders(config)
    
    model = SentinelLM(config).to(device)
    logger.info(f"Total parameter: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = create_optimizer(model, config.learning_rate, config.weight_decay)
    
    total_steps = len(train_dl) * config.max_epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = CosineLRScheduler(optimizer, warmup_steps, total_steps)
    
    start_epoch = 0
    start_step = 0
    save_path = "checkpoints/latest.pt"
    
    if os.path.exists(save_path):
        logger.info(f"Menemukan checkpoint di {save_path}, mencoba resume...")
        checkpoint = torch.load(save_path, map_location=device, weights_only=True)
        
        if 'model' in checkpoint and 'optimizer' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            if 'scaler' in checkpoint and checkpoint['scaler'] is not None and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'step_in_epoch' in checkpoint:
                start_step = checkpoint['step_in_epoch'] + 1
        else:
            model.load_state_dict(checkpoint)
            
        logger.info(f"Berhasil resume dari Epoch {start_epoch + 1}, Step {start_step}")
        
        exp_dirs = sorted(glob.glob("experiments/exp*"))
        if exp_dirs:
            exp_dir = exp_dirs[-1]
            logger.info(f"Melanjutkan logging di direktori eksperimen: {exp_dir}")
        else:
            exp_dir = setup_experiment(config)
            
        csv_file = open(os.path.join(exp_dir, "loss.csv"), "a", newline="")
        csv_writer = csv.writer(csv_file)
        
    else:
        exp_dir = setup_experiment(config)
        logger.info(f"Eksperimen dimulai. Log tersimpan di {exp_dir}")
        
        csv_file = open(os.path.join(exp_dir, "loss.csv"), "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "step", "train_loss", "lr"])
    
    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        total_loss = 0.0
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[info]}"),
        ) as progress:
            task = progress.add_task(f"Epoch {epoch+1}/{config.max_epochs}", total=len(train_dl), info="")
            
            train_iter = iter(train_dl)
            if epoch == start_epoch and start_step > 0:
                logger.info(f"Fast-forwarding DataLoader sebanyak {start_step} langkah...")
                for _ in range(start_step):
                    next(train_iter)
                progress.update(task, advance=start_step)
            
            for step_idx in range(start_step if epoch == start_epoch else 0, len(train_dl)):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break
                
                step = step_idx
                x = batch["input_ids"].to(device)
                y = batch["target_ids"].to(device)
                
                optimizer.zero_grad()
                
                # AMP Autocast
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = calculate_loss(logits, y)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                lr = scheduler.step()
                total_loss += loss.item()
                
                csv_writer.writerow([epoch+1, step+1, loss.item(), lr])
                
                progress.update(task, advance=1, info=f"Loss: {loss.item():.4f} | LR: {lr:.2e}")
                
                # Simpan checkpoint sementara tiap 500 step
                if (step + 1) % 500 == 0:
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None,
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'step_in_epoch': step
                    }, "checkpoints/latest.pt")
                    
        denominator = len(train_dl) - (start_step if epoch == start_epoch else 0)
        avg_loss = total_loss / denominator if denominator > 0 else 0.0
        start_step = 0
        logger.info(f"Epoch {epoch+1} selesai | Train Loss: {avg_loss:.4f}")
        
    csv_file.close()
    
    # Simpan model final di folder eksperimen
    final_model_path = os.path.join(exp_dir, "model.pt")
    torch.save(model.state_dict(), final_model_path)
    
    # Update latest.pt 
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, "checkpoints/latest.pt")
    
    # Catatan otomatis
    with open(os.path.join(exp_dir, "notes.md"), "w") as f:
        f.write(f"# Eksperimen Selesai\n- Waktu: {datetime.now()}\n- Final Train Loss: {avg_loss:.4f}\n")
        
    logger.info(f"Model final tersimpan di {final_model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/small.yaml")
    args = parser.parse_args()
    
    train(args.config)
