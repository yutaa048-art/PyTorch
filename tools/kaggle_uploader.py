import os
import shutil
import subprocess
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

console = Console()

def ignore_files(dir_name, files):
    """Fungsi untuk mengabaikan file/folder yang tidak perlu diupload ke Kaggle."""
    ignored = []
    # Folder utama yang harus di-skip
    if "venv" in files: ignored.append("venv")
    if ".git" in files: ignored.append(".git")
    if "__pycache__" in files: ignored.append("__pycache__")
    
    # Jangan copy folder kaggle_upload itu sendiri agar tidak rekursif loop
    if "kaggle_upload" in files: ignored.append("kaggle_upload")
    
    # Skip raw_repos karena sangat besar dan tidak digunakan untuk training
    if "raw_repos" in files: ignored.append("raw_repos")
    
    # Skip dataset berformat tensor .pt besar jika ingin di-generate di Kaggle saja
    # Namun karena tujuan kita adalah memuluskan pretraining, kita akan upload corpus txt
    # dan biarkan Kaggle preprocess atau kita upload tensornya juga jika ada.
    
    return ignored

def prepare_kaggle_package():
    console.print("[bold cyan]Menyiapkan paket SentinelLM untuk Kaggle...[/bold cyan]")
    
    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    upload_dir = os.path.join(source_dir, "kaggle_upload")
    
    # Bersihkan folder upload lama jika ada
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
        
    os.makedirs(upload_dir)
    
    console.print("Menyalin Codebase & Dataset (Mengabaikan venv, .git, raw_repos)...")
    # Salin seluruh isi SentinelLM ke dalam kaggle_upload/SentinelLM
    project_dest = os.path.join(upload_dir, "SentinelLM")
    
    shutil.copytree(source_dir, project_dest, ignore=ignore_files)
    
    # Buat dataset-metadata.json
    console.print("Membuat dataset-metadata.json...")
    metadata = {
        "title": "SentinelLM V0.2 Codebase and Corpus",
        "id": "whuteringkuro/sentinellm-v02-project",
        "licenses": [
            {
                "name": "CC0-1.0"
            }
        ]
    }
    
    with open(os.path.join(upload_dir, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    console.print("[bold green]Paket berhasil disiapkan di folder kaggle_upload/[/bold green]")
    
    return upload_dir

def push_to_kaggle(upload_dir):
    console.print("[bold yellow]Memulai proses upload ke Kaggle...[/bold yellow]")
    console.print("Ini bisa memakan waktu bergantung pada ukuran corpus (ratusan MB).")
    
    # Kita panggil perintah CLI kaggle melalui subprocess
    try:
        # Cek apakah dataset sudah ada
        result = subprocess.run(["./venv/bin/kaggle", "datasets", "status", "whuteringkuro/sentinellm-v02-project"], 
                                capture_output=True, text=True)
        
        if "ready" in result.stdout.lower() or "not found" not in result.stderr.lower() and "404" not in result.stdout:
            # Dataset sudah ada, buat versi baru
            console.print("Dataset sudah ada di Kaggle. Membuat versi baru...")
            subprocess.run(["./venv/bin/kaggle", "datasets", "version", "-p", upload_dir, "-m", "Auto-update from SentinelLM Workspace", "--dir-mode", "zip"], check=True)
        else:
            # Dataset belum ada, buat baru
            console.print("Dataset baru. Melakukan inisialisasi...")
            subprocess.run(["./venv/bin/kaggle", "datasets", "create", "-p", upload_dir, "--dir-mode", "zip"], check=True)
            
        console.print("[bold green]SUKSES! SentinelLM kini tersedia di Kaggle Datasets.[/bold green]")
        console.print("Gunakan dataset ini di Kaggle Notebook Anda untuk Pretraining!")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Gagal melakukan upload ke Kaggle. Error: {e}[/bold red]")

if __name__ == "__main__":
    upload_path = prepare_kaggle_package()
    push_to_kaggle(upload_path)
