import os
import hashlib

def detect_duplicates(corpus_dir: str):
    seen_hashes = set()
    total_files = 0
    duplicate_files = 0
    
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue
                
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            # Pisahkan file berdasarkan header
            blocks = content.split("# --- FILE: ")
            for block in blocks:
                if not block.strip():
                    continue
                    
                total_files += 1
                # Hash isi blok (abaikan nama file di baris pertama)
                lines = block.split("\n")[1:]
                body = "\n".join(lines).strip()
                
                if not body:
                    continue
                    
                h = hashlib.sha256(body.encode('utf-8')).hexdigest()
                if h in seen_hashes:
                    duplicate_files += 1
                else:
                    seen_hashes.add(h)
                    
    return {
        "total": total_files,
        "duplicate": duplicate_files,
        "unique": total_files - duplicate_files
    }
