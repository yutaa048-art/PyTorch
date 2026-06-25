import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel

from evaluation.benchmark import evaluate_model
from evaluation.generation_test import run_generation_tests
from evaluation.security_test import run_security_tests
from evaluation.attention_heatmap import generate_heatmap

console = Console()

def run_suite(config_path="config/small.yaml"):
    console.print(Panel.fit("[bold cyan]SentinelLM Validation Suite[/bold cyan]"))
    
    console.print("\n[bold magenta]=== Test 1 & 2: Kuantitatif (Perplexity & Accuracy) ===[/bold magenta]")
    evaluate_model(config_path)
    
    console.print("\n[bold magenta]=== Test 3: Kualitatif (Generation) ===[/bold magenta]")
    run_generation_tests(config_path)
    
    console.print("\n[bold magenta]=== Test 4: Kualitatif (Security SDK) ===[/bold magenta]")
    run_security_tests(config_path)
    
    console.print("\n[bold magenta]=== Test 5: Visual Analytics (Attention) ===[/bold magenta]")
    generate_heatmap("Authorization: Bearer JWT", config_path=config_path, layer_idx=-1)
    
    console.print(Panel.fit("[bold green]Validation Suite Selesai Dijalankan![/bold green]"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/small.yaml")
    args = parser.parse_args()
    
    run_suite(args.config)
