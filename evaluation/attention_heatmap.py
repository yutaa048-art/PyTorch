import os
import torch
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from model.model import SentinelLM
from tokenizer.encode import encode
from tokenizer.decode import decode
from rich.console import Console
from rich.panel import Panel

console = Console()

def generate_heatmap(prompt: str, config_path="config/small.yaml", layer_idx=-1):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_path = "checkpoints/latest.pt"
    if not os.path.exists(save_path):
        console.print(f"[red]Model checkpoint tidak ditemukan di {save_path}[/red]")
        return
        
    model = SentinelLM(config)
    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    input_ids = encode(prompt, config)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Dapatkan label token (Decode satu per satu agar bisa jadi label axes)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(config.tokenizer_path)
    
    labels = []
    for token_id in input_ids:
        piece = sp.id_to_piece(token_id).replace(' ', ' ')
        labels.append(piece)
        
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            # Jalankan dengan flag output_attentions=True
            _, all_attentions = model(x, output_attentions=True)
            
    # all_attentions adalah list berisikan attention tensor tiap layer.
    # Pilih layer tertentu (default: -1 atau layer terakhir)
    attn_matrix = all_attentions[layer_idx] # shape: [batch_size, num_heads, seq_len, seq_len]
    
    # Ambil rata-rata atensi dari semua heads
    attn_avg = attn_matrix[0].mean(dim=0).cpu().numpy() # shape: [seq_len, seq_len]
    
    console.print(Panel.fit(f"[bold magenta]Test 5: Token Heatmap (Layer {layer_idx})[/bold magenta]"))
    console.print(f"Menggambar matriks korelasi untuk kalimat: '{prompt}'")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_avg, xticklabels=labels, yticklabels=labels, cmap="viridis")
    plt.title(f"Attention Heatmap (Layer {layer_idx})")
    plt.xlabel("Key Tokens (Yang Dilihat)")
    plt.ylabel("Query Tokens (Yang Melihat)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    out_dir = "evaluation/results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "attention_heatmap.png")
    
    plt.savefig(out_file, dpi=300)
    console.print(f"[bold green]Heatmap berhasil disimpan di: {out_file}[/bold green]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Authorization: Bearer JWT")
    parser.add_argument("--layer", type=int, default=-1)
    args = parser.parse_args()
    
    generate_heatmap(args.prompt, layer_idx=args.layer)
