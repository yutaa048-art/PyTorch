import os
import sys
import json
import hashlib
import argparse
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

def filter_and_hash_content(content: str) -> tuple[bool, str, str]:
    """
    Menerapkan heuristik pembersihan:
    - Membuang baris kosong yang berlebihan
    - Membuang jika terlalu pendek
    - Menghitung SHA-256 hash untuk deteksi duplikasi persis
    """
    lines = content.split('\n')
    
    # 1. Buang baris yang terlalu panjang (minified JS, hex dump)
    cleaned_lines = []
    consecutive_empty = 0
    for line in lines:
        if len(line) > 1000:
            return False, "", ""  # Buang seluruh file jika ada baris terlalu panjang
        
        if not line.strip():
            consecutive_empty += 1
            if consecutive_empty > 2:
                continue  # Skip baris kosong jika lebih dari 2 berurutan
        else:
            consecutive_empty = 0
            
        cleaned_lines.append(line)
        
    cleaned_content = '\n'.join(cleaned_lines).strip()
    
    # 2. Heuristik Panjang Minimal
    if len(cleaned_content) < 100:
        return False, "", ""
        
    # 3. Hitung Hash (untuk deduplikasi logis, abaikan baris pertama jika itu nama file)
    body_for_hash = '\n'.join(cleaned_lines[1:]).strip() if len(cleaned_lines) > 1 else cleaned_content
    content_hash = hashlib.sha256(body_for_hash.encode('utf-8', errors='ignore')).hexdigest()
    
    return True, cleaned_content, content_hash

def process_file(filepath: str) -> dict:
    """
    Memproses satu file. Akan dipanggil oleh ProcessPoolExecutor.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        is_valid, cleaned_text, content_hash = filter_and_hash_content(content)
        
        if is_valid:
            return {
                "success": True,
                "filepath": str(filepath),
                "text": cleaned_text,
                "hash": content_hash,
                "size_bytes": len(cleaned_text.encode('utf-8'))
            }
        else:
            return {"success": False, "reason": "filtered"}
    except Exception as e:
        return {"success": False, "reason": f"error: {str(e)}"}

def chunk_writer(output_dir: str, chunk_size_mb: int = 500):
    """
    Generator/coroutine untuk mengelola penulisan chunk file JSONL.
    Menerima dictionary data dan memindahkannya ke file baru jika chunk penuh.
    """
    chunk_idx = 1
    current_size = 0
    chunk_limit = chunk_size_mb * 1024 * 1024
    
    os.makedirs(output_dir, exist_ok=True)
    current_file = os.path.join(output_dir, f"corpus_chunk_{chunk_idx:03d}.jsonl")
    f = open(current_file, 'w', encoding='utf-8')
    
    try:
        while True:
            data = yield
            if data is None:
                break
                
            json_str = json.dumps({"text": data["text"], "source": data["filepath"]}) + "\n"
            json_bytes = json_str.encode('utf-8')
            
            f.write(json_str)
            current_size += len(json_bytes)
            
            if current_size >= chunk_limit:
                f.close()
                chunk_idx += 1
                current_size = 0
                current_file = os.path.join(output_dir, f"corpus_chunk_{chunk_idx:03d}.jsonl")
                f = open(current_file, 'w', encoding='utf-8')
    finally:
        f.close()

def main(input_dir: str, output_dir: str, chunk_size_mb: int):
    print(f"Scanning files in: {input_dir}")
    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Sesuaikan dengan jenis file korpus Anda
            if file.endswith((".txt", ".md", ".py", ".java", ".c", ".cpp", ".json")):
                all_files.append(os.path.join(root, file))
                
    total_files = len(all_files)
    print(f"Found {total_files} files to process.")
    if total_files == 0:
        return
        
    seen_hashes = set()
    stats = {
        "processed": 0,
        "written": 0,
        "duplicates_dropped": 0,
        "filtered_dropped": 0,
        "errors": 0,
        "total_output_bytes": 0
    }
    
    # Initialize the writer coroutine
    writer = chunk_writer(output_dir, chunk_size_mb)
    next(writer) # Prime the generator
    
    num_cores = max(1, mp.cpu_count() - 1)
    print(f"Starting processing pool with {num_cores} workers...")
    
    pool = mp.Pool(processes=num_cores)
    
    try:
        with tqdm(total=total_files, desc="Cleaning Data", unit="file") as pbar:
            for result in pool.imap_unordered(process_file, all_files, chunksize=100):
                stats["processed"] += 1
                
                if result["success"]:
                    content_hash = result["hash"]
                    if content_hash in seen_hashes:
                        stats["duplicates_dropped"] += 1
                    else:
                        seen_hashes.add(content_hash)
                        writer.send(result)
                        stats["written"] += 1
                        stats["total_output_bytes"] += result["size_bytes"]
                else:
                    if result.get("reason", "").startswith("error"):
                        stats["errors"] += 1
                    else:
                        stats["filtered_dropped"] += 1
                        
                pbar.update(1)
                pbar.set_postfix({
                    "Valid": stats["written"], 
                    "Dup": stats["duplicates_dropped"],
                    "GB": f"{stats['total_output_bytes'] / (1024**3):.2f}"
                })
    finally:
        pool.close()
        pool.join()
        try:
            writer.send(None) # Close writer
        except StopIteration:
            pass
        
    print("\n" + "="*50)
    print("🧹 CLEANING REPORT")
    print("="*50)
    print(f"Total Files Scanned : {stats['processed']:,}")
    print(f"Files Kept (Valid)  : {stats['written']:,}")
    print(f"Duplicates Dropped  : {stats['duplicates_dropped']:,}")
    print(f"Low Quality Dropped : {stats['filtered_dropped']:,}")
    print(f"Errors              : {stats['errors']:,}")
    print(f"Final Output Size   : {stats['total_output_bytes'] / (1024**3):.2f} GB")
    print("="*50)
    print(f"Bersih! Chunk file telah disimpan di: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and deduplicate massive corpus")
    parser.add_argument("--input", type=str, required=True, help="Path ke root direktori corpus mentah")
    parser.add_argument("--output", type=str, default="dataset/clean_corpus", help="Path untuk menyimpan chunks .jsonl")
    parser.add_argument("--chunk_size_mb", type=int, default=1000, help="Ukuran tiap chunk output dalam MB (Default: 1000 MB / 1 GB)")
    
    args = parser.parse_args()
    main(args.input, args.output, args.chunk_size_mb)
