import argparse
import sys
import os

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.language_stats import analyze_languages
from tools.deduplicate import detect_duplicates
from tools.quality_score import analyze_quality

console = Console()

def run_inspection(corpus_dir: str, config_path: str):
    console.print(Panel.fit("[bold cyan]Corpus Inspector[/bold cyan]", subtitle="SentinelLM v0.2"))
    
    with console.status("[bold green]Menganalisis Bahasa & Tipe File..."):
        lang_stats = analyze_languages(corpus_dir)
        
    table_lang = Table(title="1. Language Statistics")
    table_lang.add_column("Language", style="cyan")
    table_lang.add_column("Percentage", justify="right", style="green")
    
    for lang, data in lang_stats.items():
        table_lang.add_row(lang, f"{data['percentage']:.1f}%")
    console.print(table_lang)
    
    with console.status("[bold green]Mendeteksi Duplikasi..."):
        dup_stats = detect_duplicates(corpus_dir)
        
    table_dup = Table(title="2. Duplicate Detection")
    table_dup.add_column("Metric", style="cyan")
    table_dup.add_column("Count", justify="right", style="magenta")
    
    table_dup.add_row("Total Files", str(dup_stats["total"]))
    table_dup.add_row("Duplicate", str(dup_stats["duplicate"]))
    table_dup.add_row("Unique", str(dup_stats["unique"]))
    console.print(table_dup)
    
    with console.status("[bold green]Menghitung Statistik Token & Security Coverage (Bisa memakan waktu)..."):
        qual_stats = analyze_quality(corpus_dir, config_path)
        
    table_tok = Table(title="3. Token Statistics")
    table_tok.add_column("Metric", style="cyan")
    table_tok.add_column("Value", justify="right", style="yellow")
    
    table_tok.add_row("Total Characters", f"{qual_stats['total_chars']/1e6:.1f} MB")
    table_tok.add_row("Total Tokens", f"{qual_stats['total_tokens']/1e6:.1f} juta")
    table_tok.add_row("Avg Token Length", f"{qual_stats['avg_token_len']:.2f}")
    table_tok.add_row("Vocabulary Usage", f"{qual_stats['vocab_usage']:.1f}%")
    console.print(table_tok)
    
    table_ctx = Table(title="4. Context Distribution")
    table_ctx.add_column("Length", style="cyan")
    table_ctx.add_column("Percentage", justify="right", style="green")
    for k, v in qual_stats['context_dist'].items():
        table_ctx.add_row(f"<= {k} tokens", f"{v:.1f}%")
    console.print(table_ctx)
    
    table_sec = Table(title="5. Security Coverage")
    table_sec.add_column("Keyword", style="cyan")
    table_sec.add_column("Occurrences", justify="right", style="red")
    
    for kw, count in sorted(qual_stats['security_counts'].items(), key=lambda x: x[1], reverse=True):
        table_sec.add_row(kw, str(count))
    console.print(table_sec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="dataset/corpus")
    parser.add_argument("--config", type=str, default="config/small.yaml")
    args = parser.parse_args()
    
    run_inspection(args.corpus, args.config)
