import os
import csv
import math
import yaml
import torch
import glob
from datetime import datetime
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from torch.utils.tensorboard import SummaryWriter

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
from inference.generate import generate_text

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

@torch.no_grad()
def validate(model, val_dl, device):
    if val_dl is None or len(val_dl) == 0:
        return float('nan')
    model.eval()
    total_loss = 0.0
    for batch in val_dl:
        x = batch["input_ids"].to(device)
        y = batch["target_ids"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = calculate_loss(logits, y)
        total_loss += loss.item()
    return total_loss / len(val_dl)

def train(config_path="config/small.yaml", resume_path=""):
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
    
    # Gradient Accumulation: effective batch = batch_size * accumulation_steps
    # Contoh: batch_size=16, accumulation_steps=4 → effective batch_size=64
    accumulation_steps = getattr(config, 'accumulation_steps', 4)
    logger.info(f"Gradient Accumulation: {accumulation_steps} steps (effective batch_size={config.batch_size * accumulation_steps})")
    
    # FIX: inisialisasi lr sebelum training loop agar tidak NameError di step 0-2
    # (sebelum accumulation step pertama terpenuhi)
    lr = config.learning_rate
    
    start_epoch = 0
    start_step = 0
    save_path = resume_path if resume_path else "checkpoints/latest.pt"
    
    if os.path.exists(save_path):
        logger.info(f"Menemukan checkpoint di {save_path}, mencoba resume...")
        checkpoint = torch.load(save_path, map_location=device, weights_only=True)
        
        if 'model' in checkpoint and 'optimizer' in checkpoint:
            state_dict = checkpoint['model']
            clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict)
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
            clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
            model.load_state_dict(clean_state_dict)
            
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
        
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "logs"))
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
        
    if torch.cuda.device_count() > 1:
        logger.info(f"Menggunakan {torch.cuda.device_count()} GPUs dengan DataParallel")
        model = torch.nn.DataParallel(model)
    
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
            
            optimizer.zero_grad()  # Reset gradient di awal setiap epoch
            
            for step_idx in range(start_step if epoch == start_epoch else 0, len(train_dl)):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break
                
                step = step_idx
                x = batch["input_ids"].to(device, non_blocking=True)
                y = batch["target_ids"].to(device, non_blocking=True)
                
                # AMP Autocast
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = calculate_loss(logits, y)
                    # Bagi loss agar rata-rata gradien akumulasi tetap setara 1 batch
                    loss = loss / accumulation_steps
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Lakukan optimizer step hanya tiap accumulation_steps
                is_accumulation_step = (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dl)
                if is_accumulation_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    lr = scheduler.step()
                    optimizer.zero_grad()
                
                # Loss yang dicatat adalah nilai sebelum dibagi (skala penuh)
                raw_loss = loss.item() * accumulation_steps
                total_loss += raw_loss
                
                csv_writer.writerow([epoch+1, step+1, raw_loss, lr])
                
                progress.update(task, advance=1, info=f"Loss: {raw_loss:.4f} | LR: {lr:.2e}")
                
                # Simpan checkpoint sementara tiap 500 step
                if (step + 1) % 500 == 0:
                    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                    torch.save({
                        'model': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None,
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'step_in_epoch': step
                    }, "checkpoints/latest.pt")
                    # Flush CSV agar data tidak hilang jika Kaggle crash
                    csv_file.flush()
                    
        denominator = len(train_dl) - (start_step if epoch == start_epoch else 0)
        avg_loss = total_loss / denominator if denominator > 0 else 0.0
        start_step = 0
        
        # Validation & Metrics
        val_loss = validate(model, val_dl, device)
        train_perplexity = math.exp(avg_loss) if avg_loss < 50 else float('inf')
        val_perplexity = math.exp(val_loss) if (val_loss and not math.isnan(val_loss) and val_loss < 50) else float('nan')
        gpu_mem = torch.cuda.memory_reserved(device) / (1024**3) if device.type == 'cuda' else 0.0
        
        # Deteksi NaN loss: jika avg_loss NaN, scaler mungkin sudah skip semua update
        if math.isnan(avg_loss) or math.isinf(avg_loss):
            logger.error(f"  FATAL: avg_loss={avg_loss:.4f} - training tidak stabil! Cek data dan learning rate.")
            break
        
        # Logging
        logger.info(f"Epoch {epoch+1}/{config.max_epochs} Selesai")
        logger.info(f"  Train Loss: {avg_loss:.4f} | Train PPL: {train_perplexity:.2f}")
        if not math.isnan(val_loss):
            logger.info(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_perplexity:.2f}")
        else:
            logger.info(f"  Val Loss:   N/A (val dataset terlalu kecil)")
        logger.info(f"  GPU Mem:    {gpu_mem:.2f} GB | LR: {lr:.2e}")
        
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        if not math.isnan(val_loss):
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Perplexity/Validation", val_perplexity, epoch)
        writer.add_scalar("Perplexity/Train", train_perplexity, epoch)
        
        # Sample Generation + bersihkan VRAM setelahnya
        model_to_eval = model.module if isinstance(model, torch.nn.DataParallel) else model
        sample_prompt = "def scan_port("
        try:
            logger.info("  Sample Generation:")
            with torch.no_grad():
                generated = generate_text(sample_prompt, config, model_to_eval, device, max_new_tokens=30)
            logger.info(f"    {generated.replace(chr(10), ' ')}")
        except Exception as e:
            logger.error(f"    Gagal melakukan sample generation: {e}")
        finally:
            # FIX: kembalikan model ke train mode setelah generate_text memanggil model.eval()
            model.train()
            # Bersihkan memori sisa generation sebelum epoch berikutnya
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        # Checkpoint Best & Early Stopping (hanya jika val_loss valid)
        if not math.isnan(val_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model': model_to_eval.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, "checkpoints/best.pt")
                logger.info("  ✅ New Best Checkpoint Saved!")
            else:
                patience_counter += 1
                logger.info(f"  ⚠️ Validation loss tidak membaik. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info("🛑 Early Stopping triggered! Training dihentikan.")
                    break
        else:
            logger.info("  ⏭️ Melewati Early Stopping (val dataset tidak tersedia)")
        
    csv_file.close()
    writer.close()
    
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # Simpan model final di folder eksperimen
    final_model_path = os.path.join(exp_dir, "model.pt")
    torch.save(model_to_save.state_dict(), final_model_path)
    
    # Update latest.pt 
    torch.save({
        'model': model_to_save.state_dict(),
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
    parser.add_argument("--resume", type=str, default="", help="Path ke checkpoint untuk resume. Kosongkan untuk auto-resume latest.pt")
    args = parser.parse_args()
    
    train(args.config, args.resume)
