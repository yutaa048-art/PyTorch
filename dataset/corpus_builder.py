import os
import shutil
import subprocess
import hashlib
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("CorpusBuilder")

TARGET_SIZE_MB = 250

RATIOS = {
    "code": 0.60,      # 150 MB
    "docs": 0.20,      # 50 MB
    "security": 0.10,  # 25 MB
    "writeups": 0.10   # 25 MB
}

REPOSITORIES = {
    "chatbot": [
        "https://github.com/langchain-ai/langchain.git",
        "https://github.com/open-webui/open-webui.git",
        "https://github.com/tiangolo/fastapi.git",
        "https://github.com/run-llama/llama_index.git",
        "https://github.com/microsoft/autogen.git",
        "https://github.com/vllm-project/vllm.git",
        "https://github.com/huggingface/transformers.git"
    ],
    "security": [
        "https://github.com/corca-ai/Eval-LLM.git",
        "https://github.com/leondz/Awesome-LLM-Security.git",
        "https://github.com/tombkeeper/Awesome-LLM-Security.git",
        "https://github.com/greshake/llm-security.git",
        "https://github.com/OWASP/www-project-top-10-for-large-language-model-applications.git"
    ],
    "writeups": [
        "https://github.com/verazuo/jailbreak_llms.git",
        "https://github.com/JailbreakChat/jailbreakchat.git"
    ]
}

CODE_EXTS = {".py", ".ts", ".js", ".go", ".rs", ".cpp", ".c", ".h", ".hpp", ".java", ".yml", ".yaml", ".json"}
DOCS_EXTS = {".md", ".rst", ".txt", ".html"}

def clone_repo(url: str, dest_dir: str):
    logger.info(f"Cloning {url} into {dest_dir}...")
    try:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        subprocess.run(["git", "clone", "--depth", "1", url, dest_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Gagal melakukan clone pada {url}")
        return False

def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def extract_and_route(repo_dir: str, repo_category: str, sizes: dict, targets: dict, seen_hashes: set):
    for root, dirs, files in os.walk(repo_dir):
        # Hindari folder tidak penting
        if ".git" in dirs: dirs.remove(".git")
        if "node_modules" in dirs: dirs.remove("node_modules")
        if "__pycache__" in dirs: dirs.remove("__pycache__")
        if "venv" in dirs: dirs.remove("venv")
            
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in CODE_EXTS and ext not in DOCS_EXTS:
                continue
                
            # Tentukan kategori file ini
            file_cat = ""
            if repo_category == "chatbot":
                file_cat = "code" if ext in CODE_EXTS else "docs"
            elif repo_category == "security":
                file_cat = "security"
            elif repo_category == "writeups":
                file_cat = "writeups"
                
            if sizes[file_cat] >= targets[file_cat]:
                continue # Target tercapai untuk kategori ini
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    
                if len(content) < 50:
                    continue
                    
                content_hash = hash_content(content)
                if content_hash in seen_hashes:
                    continue # Deduplikasi
                seen_hashes.add(content_hash)
                    
                header = f"\n\n# --- FILE: {file} ---\n\n"
                full_text = header + content
                
                out_path = os.path.join("dataset", "corpus", file_cat, "data.txt")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                with open(out_path, "a", encoding="utf-8") as out_f:
                    out_f.write(full_text)
                    
                sizes[file_cat] += len(full_text.encode('utf-8'))
                
            except Exception:
                pass

def build_corpus():
    raw_dir = "dataset/raw_repos"
    os.makedirs(raw_dir, exist_ok=True)
    
    # Bersihkan corpus lama jika ada untuk membuat yang baru dengan rasio benar
    if os.path.exists("dataset/corpus"):
        shutil.rmtree("dataset/corpus")
        
    for cat in RATIOS.keys():
        os.makedirs(os.path.join("dataset", "corpus", cat), exist_ok=True)
        
    targets = {k: int(v * TARGET_SIZE_MB * 1024 * 1024) for k, v in RATIOS.items()}
    sizes = {k: 0 for k in RATIOS.keys()}
    seen_hashes = set()
    
    # Process Repos
    for repo_cat, urls in REPOSITORIES.items():
        for url in urls:
            # Cek apakah target untuk kategori ini sudah penuh
            if repo_cat == "chatbot" and sizes["code"] >= targets["code"] and sizes["docs"] >= targets["docs"]:
                continue
            if repo_cat == "security" and sizes["security"] >= targets["security"]:
                continue
            if repo_cat == "writeups" and sizes["writeups"] >= targets["writeups"]:
                continue
                
            repo_name = url.split("/")[-1].replace(".git", "")
            repo_dest = os.path.join(raw_dir, repo_name)
            
            if not os.path.exists(repo_dest):
                success = clone_repo(url, repo_dest)
                if not success: continue
                
            logger.info(f"Mengekstrak teks dari {repo_name}...")
            extract_and_route(repo_dest, repo_cat, sizes, targets, seen_hashes)
            
            logger.info(f"Progress: Code {sizes['code']/1e6:.1f}/{targets['code']/1e6:.1f}MB | "
                        f"Docs {sizes['docs']/1e6:.1f}/{targets['docs']/1e6:.1f}MB | "
                        f"Security {sizes['security']/1e6:.1f}/{targets['security']/1e6:.1f}MB | "
                        f"Writeups {sizes['writeups']/1e6:.1f}/{targets['writeups']/1e6:.1f}MB")
            
            # Hapus repo yang sudah diekstrak untuk hemat disk
            shutil.rmtree(repo_dest, ignore_errors=True)
            
    logger.info("Corpus V0.1 selesai dikompilasi dengan Curriculum Learning structure!")

if __name__ == "__main__":
    build_corpus()
