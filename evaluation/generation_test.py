import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from model.model import SentinelLM
from inference.generate import generate_text
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_generation_tests(config_path="config/small.yaml"):
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
    
    prompts = [
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.post(\"/chat\")\n",
        "def fibonacci(n):\n    if n <= 1:\n",
        "import requests\n\nresponse = requests.get(\"https://api.github.com"
    ]
    
    console.print(Panel.fit("[bold cyan]Test 3: Generation Test (General)[/bold cyan]"))
    
    for i, prompt in enumerate(prompts):
        console.print(f"\n[bold yellow]Prompt {i+1}:[/bold yellow]\n{prompt}")
        output = generate_text(prompt, config, model, device, max_new_tokens=20)
        
        # Highlight generated part (remove prompt from display to see completion)
        completion = output[len(prompt):]
        console.print(f"[bold green]Completion:[/bold green]\n{completion}")
        console.print("-" * 50)

if __name__ == "__main__":
    run_generation_tests()
