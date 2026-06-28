import os
import csv
import math
import yaml
import torch
import glob
import random
import time
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed
from dataset.loader import get_dataloaders
from dataset.aux_labeler import compute_aux_labels_batch
from dataset.topic_classifier import classify_topic
from model.model import SentinelLM
from training.loss import calculate_loss, calculate_loss_unreduced
from training.optimizer import create_optimizer
from training.scheduler import CosineLRScheduler
from inference.generate import generate_text
import sentencepiece as spm

logger = get_logger("Trainer")
console = Console()

# ==============================================================================
# Curriculum Context Scheduler
# ==============================================================================

class CurriculumScheduler:
    """
    Mengubah seq_len secara bertahap selama training.
    Fase: 512 → 1024 → 2048 → 4096
    """
    PHASES = [
        (0.00, 512),
        (0.25, 1024),
        (0.50, 2048),
        (0.75, 4096),
    ]

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_seq_len = self.PHASES[0][1]

    def get_seq_len(self, global_step: int) -> int:
        progress = global_step / max(1, self.total_steps)
        seq_len = self.PHASES[0][1]
        for threshold, sl in self.PHASES:
            if progress >= threshold:
                seq_len = sl
        return seq_len

    def step(self, global_step: int) -> tuple[int, bool]:
        """Mengembalikan (seq_len, changed) — apakah seq_len berubah."""
        new_sl = self.get_seq_len(global_step)
        changed = (new_sl != self.current_seq_len)
        self.current_seq_len = new_sl
        return new_sl, changed


# ==============================================================================
# Knowledge Replay Buffer (KRB)
# ==============================================================================

class KnowledgeReplayBuffer:
    """
    Buffer replay berbasis topik. Setiap topik mendapat slot tersendiri.
    """
    def __init__(self, max_per_slot: int = 200):
        self.max_per_slot = max_per_slot
        self.slots = defaultdict(list)  # topic -> list of (x_tensor, y_tensor)

    def add(self, x: torch.Tensor, y: torch.Tensor, topic: str | None, is_hard: bool):
        slot_key = topic if topic else "_hard_examples"
        if not topic and not is_hard:
            return

        self.slots[slot_key].append((x.cpu().clone(), y.cpu().clone()))
        if len(self.slots[slot_key]) > self.max_per_slot:
            self.slots[slot_key] = self.slots[slot_key][-self.max_per_slot:]

    def total_size(self) -> int:
        return sum(len(v) for v in self.slots.values())

    def sample_batch(self, batch_size: int, device: torch.device):
        filled_slots = {k: v for k, v in self.slots.items() if len(v) > 0}
        if not filled_slots or self.total_size() < batch_size:
            return None

        num_slots = len(filled_slots)
        per_slot = max(1, batch_size // num_slots)
        remainder = batch_size - (per_slot * num_slots)

        samples = []
        hard_count = 0
        topic_count = 0
        slot_keys = list(filled_slots.keys())
        
        for i, key in enumerate(slot_keys):
            n = per_slot + (1 if i < remainder else 0)
            pool = filled_slots[key]
            chosen = random.choices(pool, k=min(n, len(pool)))
            for c in chosen:
                samples.append(c)
                if key == "_hard_examples":
                    hard_count += 1
                else:
                    topic_count += 1

        while len(samples) < batch_size:
            all_items = [(k, item) for k, sl in filled_slots.items() for item in sl]
            k, item = random.choice(all_items)
            samples.append(item)
            if k == "_hard_examples":
                hard_count += 1
            else:
                topic_count += 1

        samples = samples[:batch_size]
        xs = torch.stack([s[0] for s in samples]).to(device, non_blocking=True)
        ys = torch.stack([s[1] for s in samples]).to(device, non_blocking=True)
        return xs, ys, hard_count, topic_count

    def stats(self) -> dict:
        return {k: len(v) for k, v in self.slots.items() if len(v) > 0}


# ==============================================================================
# Confidence-Weighted Combined Loss
# ==============================================================================

def compute_combined_loss(
    logits, targets,
    aux_logits: dict,
    aux_labels: dict,
    aux_confidences: dict,
    accumulation_steps: int,
    weights: dict = None
):
    if weights is None:
        weights = {
            'syntax':       0.08,
            'concept':      0.10,
            'semantic':     0.12,
            'architecture': 0.15,
            'reasoning':    0.20,
        }

    lm_loss_unreduced = calculate_loss_unreduced(logits, targets)
    lm_loss = lm_loss_unreduced.mean()
    total_loss = lm_loss

    head_loss_dict = {}
    head_acc_dict = {}
    head_conf_dict = {}

    ce = torch.nn.CrossEntropyLoss()
    for head_name in ['syntax', 'concept', 'semantic', 'architecture', 'reasoning']:
        if head_name in aux_logits and head_name in aux_labels:
            head_loss = ce(aux_logits[head_name], aux_labels[head_name])
            
            preds = torch.argmax(aux_logits[head_name], dim=-1)
            acc = (preds == aux_labels[head_name]).float().mean().item()

            conf = aux_confidences.get(head_name)
            if conf is not None:
                conf_weight = conf.mean().item()
            else:
                conf_weight = 0.5
                
            head_loss_dict[head_name] = head_loss.item()
            head_acc_dict[head_name] = acc
            head_conf_dict[head_name] = conf_weight
            
            total_loss = total_loss + weights[head_name] * conf_weight * head_loss

    return total_loss / accumulation_steps, lm_loss.item(), lm_loss_unreduced, head_loss_dict, head_acc_dict, head_conf_dict


def setup_experiment(config):
    os.makedirs("experiments", exist_ok=True)
    exp_dirs = sorted(glob.glob("experiments/exp*"))
    exp_id = len(exp_dirs) + 1
    exp_dir = f"experiments/exp{exp_id:03d}"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)
        
    try:
        import subprocess
        git_commit = subprocess.getoutput("git rev-parse HEAD").strip()
    except Exception:
        git_commit = "Unknown"
        
    notes_content = f"""# Experiment {exp_id:03d}

**Git Commit:** `{git_commit}`
**Corpus Path:** `{getattr(config, 'data_path', 'Unknown')}`

## Konfigurasi Utama
```yaml
{yaml.dump(vars(config))}
```

## Perubahan dibanding eksperimen sebelumnya:
- (Tulis perubahan di sini)
"""
    with open(os.path.join(exp_dir, "notes.md"), "w") as f:
        f.write(notes_content)
        
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


# ==============================================================================
# Main Training Loop
# ==============================================================================

def train(config_path="config/small.yaml", resume_path=""):
    config = load_config(config_path)
    set_seed(42)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Menggunakan device: {device}")

    tokenizer_path = getattr(config, 'tokenizer_path', 'tokenizer/tokenizer.model')
    sp = None
    if os.path.exists(tokenizer_path):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        logger.info(f"Tokenizer dimuat: {tokenizer_path} (vocab={sp.get_piece_size()})")
    else:
        logger.warning(f"Tokenizer tidak ditemukan di {tokenizer_path}. Aux labels dinonaktifkan.")

    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    curriculum_start_sl = getattr(config, 'curriculum_start_sl', 512)
    train_dl, val_dl = get_dataloaders(config, seq_len_override=curriculum_start_sl)

    model = SentinelLM(config).to(device)
    logger.info(f"Total parameter: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = create_optimizer(model, config.learning_rate, config.weight_decay)

    total_steps = len(train_dl) * config.max_epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = CosineLRScheduler(optimizer, warmup_steps, total_steps)

    accumulation_steps = getattr(config, 'accumulation_steps', 4)
    logger.info(f"Gradient Accumulation: {accumulation_steps} steps")

    krb = KnowledgeReplayBuffer(max_per_slot=200)
    curriculum = CurriculumScheduler(total_steps)

    lr = config.learning_rate
    global_step = 0
    start_epoch = 0
    start_step = 0
    
    # Deteksi eksperimen terakhir
    os.makedirs("experiments", exist_ok=True)
    exp_dirs = sorted(glob.glob("experiments/exp*"))
    
    if args.resume:
        save_path = args.resume
        exp_dir = os.path.dirname(os.path.dirname(save_path)) if "checkpoints" in save_path else "experiments/exp_resume"
    else:
        if exp_dirs:
            exp_dir = exp_dirs[-1]
            save_path = os.path.join(exp_dir, "checkpoints", "latest.pt")
        else:
            exp_dir = ""
            save_path = ""

    if save_path and os.path.exists(save_path):
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
            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
        else:
            clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
            model.load_state_dict(clean_state_dict)

        logger.info(f"Berhasil resume dari Epoch {start_epoch + 1}, Step {start_step}")
        csv_file = open(os.path.join(exp_dir, "loss.csv"), "a", newline="")
        csv_writer = csv.writer(csv_file)
        jsonl_file = open(os.path.join(exp_dir, "metrics.jsonl"), "a")
    else:
        exp_dir = setup_experiment(config)
        logger.info(f"Eksperimen dimulai. Output tersimpan di {exp_dir}")
        csv_file = open(os.path.join(exp_dir, "loss.csv"), "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "step", "train_loss", "lr", "seq_len"])
        jsonl_file = open(os.path.join(exp_dir, "metrics.jsonl"), "w")

    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "tensorboard"))
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    if torch.cuda.device_count() > 1:
        logger.info(f"Menggunakan {torch.cuda.device_count()} GPUs dengan DataParallel")
        model = torch.nn.DataParallel(model)

    start_time = time.time()
    
    # Kalkulasi ulang total token yang sudah diproses berdasarkan global_step (sangat penting untuk akurasi ETA saat resume)
    total_tokens_processed = sum(curriculum.get_seq_len(i) * config.batch_size for i in range(global_step))
    total_tokens_target = sum(curriculum.get_seq_len(i) * config.batch_size for i in range(total_steps))

    ema_lm_loss = 0.0
    ema_head_losses = defaultdict(float)
    ema_head_accs = defaultdict(float)
    ema_head_confs = defaultdict(float)
    ema_grad_norm = 0.0
    grad_scale = 1.0
    
    total_hard_replayed = 0
    total_topic_replayed = 0
    total_sequences_seen = 0
    last_500_tokens = 0
    last_500_time = time.time()

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

            optimizer.zero_grad()

            for step_idx in range(start_step if epoch == start_epoch else 0, len(train_dl)):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break

                step = step_idx
                global_step += 1

                new_sl, sl_changed = curriculum.step(global_step)
                if sl_changed:
                    logger.info(f"📐 Curriculum Context: seq_len changed to {new_sl}")
                    train_dl.dataset.set_seq_len(new_sl)
                    writer.add_scalar("Curriculum/SeqLen", new_sl, global_step)
                    break

                x = batch["input_ids"].to(device, non_blocking=True)
                y = batch["target_ids"].to(device, non_blocking=True)

                batch_tokens = config.batch_size * curriculum.current_seq_len
                total_tokens_processed += batch_tokens
                last_500_tokens += batch_tokens
                total_sequences_seen += config.batch_size

                is_replay_step = False
                if step > 0 and step % 50 == 0:
                    replay_out = krb.sample_batch(config.batch_size, device)
                    if replay_out is not None:
                        is_replay_step = True
                        x, y, h_cnt, t_cnt = replay_out
                        total_hard_replayed += h_cnt
                        total_topic_replayed += t_cnt

                use_aux = (sp is not None) and (step % 10 == 0) and not is_replay_step
                aux_labels = {}
                aux_confidences = {}
                decoded_texts = None
                
                if use_aux:
                    try:
                        decoded_texts = [sp.decode(ids.tolist()) for ids in x.cpu()]
                        aux_labels, aux_confidences = compute_aux_labels_batch(decoded_texts, device)
                    except Exception:
                        use_aux = False

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    if use_aux:
                        logits, aux_logits = model(x, return_aux=True)
                        loss, raw_lm, unreduced, hl_dict, ha_dict, hc_dict = compute_combined_loss(
                            logits, y, aux_logits, aux_labels, aux_confidences, accumulation_steps
                        )
                    else:
                        logits = model(x)
                        unreduced = calculate_loss_unreduced(logits, y)
                        raw_lm = unreduced.mean().item()
                        loss = unreduced.mean() / accumulation_steps

                if not is_replay_step:
                    with torch.no_grad():
                        avg_batch_loss = unreduced.mean().item()
                        if decoded_texts is None and sp is not None:
                            try:
                                decoded_texts = [sp.decode(ids.tolist()) for ids in x.cpu()]
                            except Exception:
                                decoded_texts = None

                        for i in range(len(x)):
                            seq_loss = unreduced[i].item()
                            is_hard = (seq_loss > avg_batch_loss * 1.5) or (seq_loss > 5.0)

                            topic = None
                            if decoded_texts:
                                topic = classify_topic(decoded_texts[i])

                            is_high_value = False
                            if use_aux and 'reasoning' in aux_labels:
                                if aux_labels['reasoning'][i].item() >= 5 or aux_labels['architecture'][i].item() >= 4:
                                    is_high_value = True

                            if is_hard or is_high_value or topic:
                                krb.add(x[i], y[i], topic, is_hard)

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_accumulation_step = (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dl)
                if is_accumulation_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                        scaler.step(optimizer)
                        scaler.update()
                        grad_scale = scaler.get_scale()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                        optimizer.step()
                        grad_scale = 1.0
                        
                    ema_grad_norm = 0.9 * ema_grad_norm + 0.1 * grad_norm if ema_grad_norm > 0 else grad_norm
                    lr = scheduler.step()
                    optimizer.zero_grad()

                ema_lm_loss = 0.9 * ema_lm_loss + 0.1 * raw_lm if ema_lm_loss > 0 else raw_lm
                if use_aux:
                    for k in hl_dict:
                        ema_head_losses[k] = 0.9 * ema_head_losses[k] + 0.1 * hl_dict[k] if ema_head_losses[k] > 0 else hl_dict[k]
                        ema_head_accs[k] = 0.9 * ema_head_accs[k] + 0.1 * ha_dict[k] if ema_head_accs[k] > 0 else ha_dict[k]
                        ema_head_confs[k] = 0.9 * ema_head_confs[k] + 0.1 * hc_dict[k] if ema_head_confs[k] > 0 else hc_dict[k]

                total_loss += raw_lm
                csv_writer.writerow([epoch+1, step+1, raw_lm, lr, curriculum.current_seq_len])
                krb_size = krb.total_size()
                
                progress.update(task, advance=1,
                    info=f"Loss: {raw_lm:.4f} | LR: {lr:.2e} | SL: {curriculum.current_seq_len} | KRB: {krb_size}")

                if (step + 1) % 500 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    inst_elapsed = current_time - last_500_time
                    
                    inst_tps = last_500_tokens / inst_elapsed if inst_elapsed > 0 else 0
                    avg_tps = total_tokens_processed / elapsed if elapsed > 0 else 0
                    
                    tokens_left = total_tokens_target - total_tokens_processed
                    eta_sec = tokens_left / avg_tps if avg_tps > 0 else 0
                    eta_str = f"{int(eta_sec // 86400)}d {int((eta_sec % 86400) // 3600)}h {int((eta_sec % 3600) // 60)}m"

                    gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024**3) if device.type == 'cuda' else 0
                    gpu_mem_res = torch.cuda.memory_reserved(device) / (1024**3) if device.type == 'cuda' else 0
                    try:
                        gpu_util = subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").split('\n')[0]
                        gpu_util = f"{gpu_util}%"
                    except:
                        gpu_util = "N/A"
                        
                    hard_hit_rate = (total_hard_replayed / total_sequences_seen) * 100 if total_sequences_seen > 0 else 0
                    topic_hit_rate = (total_topic_replayed / total_sequences_seen) * 100 if total_sequences_seen > 0 else 0
                    
                    ppl = math.exp(ema_lm_loss) if ema_lm_loss < 50 else float('inf')
                    
                    table = Table(show_header=False, box=None)
                    table.add_column("Key", style="cyan", justify="left")
                    table.add_column("Value", style="white", justify="left")
                    
                    table.add_row("Step", str(global_step))
                    table.add_row("LM Loss", f"{ema_lm_loss:.4f} (Perplexity: {ppl:.2f})")
                    
                    for k in ['syntax', 'concept', 'semantic', 'architecture', 'reasoning']:
                        if k in ema_head_losses:
                            acc = ema_head_accs[k]
                            bars = int(acc * 10) * '█' + (10 - int(acc * 10)) * '░'
                            table.add_row(f"Loss {k.capitalize()}", f"{ema_head_losses[k]:.4f} | Acc: {acc*100:.1f}% | Conf: {ema_head_confs[k]:.2f}")
                            table.add_row(f"Health {k.capitalize()}", f"[green]{bars}[/green] {acc*100:.1f}%")
                            
                    table.add_row("Gradient Norm", f"{ema_grad_norm:.4f} (Scale: {grad_scale})")
                    table.add_row("Learning Rate", f"{lr:.2e}")
                    table.add_row("KRB Size", str(krb_size))
                    table.add_row("Replay Hit Rate", f"Hard: {hard_hit_rate:.1f}% | Topic: {topic_hit_rate:.1f}%")
                    table.add_row("GPU", f"Util: {gpu_util} | Alloc: {gpu_mem_alloc:.1f}GB | Res: {gpu_mem_res:.1f}GB")
                    table.add_row("Tokens/sec", f"Instant: {inst_tps:.0f} | Avg: {avg_tps:.0f}")
                    table.add_row("ETA", eta_str)
                    
                    console.print(Panel(table, title=f"Training Status @ Step {global_step}", expand=False))
                    
                    json_data = {
                        "step": global_step,
                        "lm_loss": ema_lm_loss,
                        "perplexity": ppl,
                        "grad_norm": ema_grad_norm,
                        "lr": lr,
                        "krb_size": krb_size,
                        "hard_hit_rate": hard_hit_rate,
                        "topic_hit_rate": topic_hit_rate,
                        "inst_tps": inst_tps,
                        "avg_tps": avg_tps
                    }
                    for k in ema_head_losses:
                        json_data[f"{k}_loss"] = ema_head_losses[k]
                        json_data[f"{k}_acc"] = ema_head_accs[k]
                        json_data[f"{k}_conf"] = ema_head_confs[k]
                        
                    jsonl_file.write(json.dumps(json_data) + "\n")
                    jsonl_file.flush()
                    
                    writer.add_scalar("Training/PPL", ppl, global_step)
                    writer.add_scalar("Performance/TPS_Instant", inst_tps, global_step)
                    writer.add_scalar("Performance/GPU_Allocated_GB", gpu_mem_alloc, global_step)
                    for k in ema_head_losses:
                        writer.add_scalar(f"AuxLoss/{k}", ema_head_losses[k], global_step)
                        writer.add_scalar(f"AuxAcc/{k}", ema_head_accs[k], global_step)
                    
                    last_500_tokens = 0
                    last_500_time = time.time()

                    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                    torch.save({
                        'model': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None,
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'step_in_epoch': step,
                        'global_step': global_step,
                    }, os.path.join(exp_dir, "checkpoints", "latest.pt"))
                    csv_file.flush()

                    krb_stats = krb.stats()
                    if krb_stats:
                        for topic, count in krb_stats.items():
                            writer.add_scalar(f"KRB/{topic}", count, global_step)

        denominator = len(train_dl) - (start_step if epoch == start_epoch else 0)
        avg_loss = total_loss / denominator if denominator > 0 else 0.0
        start_step = 0

        val_loss = validate(model, val_dl, device)
        train_perplexity = math.exp(avg_loss) if avg_loss < 50 else float('inf')
        val_perplexity = math.exp(val_loss) if (val_loss and not math.isnan(val_loss) and val_loss < 50) else float('nan')
        gpu_mem = torch.cuda.memory_reserved(device) / (1024**3) if device.type == 'cuda' else 0.0

        if math.isnan(avg_loss) or math.isinf(avg_loss):
            logger.error(f"  FATAL: avg_loss={avg_loss:.4f} - training tidak stabil!")
            break

        logger.info(f"Epoch {epoch+1}/{config.max_epochs} Selesai")
        logger.info(f"  Train Loss: {avg_loss:.4f} | Train PPL: {train_perplexity:.2f}")
        if not math.isnan(val_loss):
            logger.info(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_perplexity:.2f}")
        else:
            logger.info(f"  Val Loss:   N/A")
        logger.info(f"  GPU Mem:    {gpu_mem:.2f} GB | LR: {lr:.2e} | SeqLen: {curriculum.current_seq_len}")

        writer.add_scalar("Loss/Train", avg_loss, epoch)
        if not math.isnan(val_loss):
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Perplexity/Validation", val_perplexity, epoch)
        writer.add_scalar("Perplexity/Train", train_perplexity, epoch)

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
            model.train()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if not math.isnan(val_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model': model_to_eval.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, os.path.join(exp_dir, "checkpoints", "best.pt"))
                logger.info("  ✅ New Best Checkpoint Saved!")
            else:
                patience_counter += 1
                logger.info(f"  ⚠️ Validation loss tidak membaik. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info("🛑 Early Stopping triggered!")
                    break
        else:
            logger.info("  ⏭️ Melewati Early Stopping (val dataset tidak tersedia)")

    csv_file.close()
    jsonl_file.close()
    writer.close()

    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    final_model_path = os.path.join(exp_dir, "checkpoints", "model_final.pt")
    torch.save(model_to_save.state_dict(), final_model_path)

    torch.save({
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(exp_dir, "checkpoints", "latest.pt"))

    with open(os.path.join(exp_dir, "notes.md"), "a") as f:
        f.write(f"\n\n## Hasil Akhir\n- Waktu Selesai: {datetime.now()}\n- Final Train Loss: {avg_loss:.4f}\n")
        f.write(f"- KRB Final Stats: {krb.stats()}\n")

    logger.info(f"Model final tersimpan di {final_model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/small.yaml")
    args = parser.parse_args()
    
    # We no longer need args.resume logic in argparse since it auto-detects from experiments/
    train(args.config, "")
