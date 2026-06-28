"""
dataset/build_memmap.py
=======================
Script untuk tokenisasi JSONL (126 batch) skala industri ke format Memory-Mapped (np.uint16).
Script ini harus dijalankan di server lokal Anda (bukan Kaggle) setelah tokenizer dilatih.

File output 'corpus.bin' ini nantinya yang akan di-upload ke Kaggle (Dataset) 
agar model bisa memuat data 22GB tanpa menggunakan RAM sama sekali.

Usage:
    python dataset/build_memmap.py \
        --input "/home/yuta/RIset Corpus/Sentinel_Corpus/Raw_Pretrain_Batches" \
        --output "/home/yuta/RIset Corpus/Sentinel_Corpus/corpus.bin" \
        --tokenizer "tokenizer/tokenizer_32k.model" \
        --workers 16
"""

import os
import glob
import json
import argparse
import multiprocessing as mp
from tqdm import tqdm
import array
import gc
import numpy as np
import sentencepiece as spm

def process_file(file_path: str, tokenizer_path: str, tmp_dir: str) -> tuple[str, int]:
    """Membaca satu file JSONL, tokenisasi, tulis ke file .bin sementara secara streaming agar RAM super aman."""
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    tmp_path = os.path.join(tmp_dir, os.path.basename(file_path) + ".bin")
    total_tokens = 0
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f, \
         open(tmp_path, 'wb') as out_f:
         
        # Gunakan array tipe H (unsigned short 2-byte) agar jauh lebih hemat RAM dibanding Python list
        chunk = array.array('H')
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Proteksi terhadap 1 baris kode minified yang luar biasa besar (misal: 1 file json/js ukuran 50MB)
            # yang bisa membuat json.loads() dan sp.encode() kehabisan RAM. Skip file > 10MB.
            if len(line) > 10_000_000:
                continue
            
            try:
                data = json.loads(line)
                text = data.get("content", data.get("text", ""))
            except json.JSONDecodeError:
                text = line
            
            if text:
                ids = sp.encode(text)
                if ids:
                    chunk.extend(ids)
                    chunk.append(sp.eos_id())
            
            # Streaming ke disk setiap 1.000.000 token (~2MB)
            if len(chunk) >= 1000000:
                arr = np.array(chunk, dtype=np.uint16)
                out_f.write(arr.tobytes())
                total_tokens += len(chunk)
                # Reset array
                chunk = array.array('H')
                
        # Sisa
        if len(chunk) > 0:
            arr = np.array(chunk, dtype=np.uint16)
            out_f.write(arr.tobytes())
            total_tokens += len(chunk)
            
    # Hapus sp dari memory secara paksa
    del sp
    gc.collect()
            
    return tmp_path, total_tokens

def build_memmap(args):
    jsonl_files = sorted(glob.glob(os.path.join(args.input, "*.jsonl")))
    if not jsonl_files:
        print(f"Tidak ada file .jsonl ditemukan di {args.input}")
        return
        
    print(f"Memulai tokenisasi {len(jsonl_files)} file JSONL ke memmap uint16...")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Output: {args.output}")
    print(f"Workers: {args.workers}")
    
    # 1. Tentukan ukuran total token
    # Kita tidak tahu pasti sebelum memproses semuanya, jadi kita akan menumpuk (append)
    # array ke file menggunakan mode 'ab' (append binary) jika file sudah ada,
    # tapi lebih aman kita kumpulkan chunk lalu tulis.
    
    # Hapus file output jika sudah ada
    if os.path.exists(args.output):
        os.remove(args.output)
        
    total_tokens = 0
    
    # Buat folder sementara untuk chunk
    tmp_dir = os.path.join(os.path.dirname(args.output), "_tmp_chunks")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Buka pool multiprocessing
    pool = mp.Pool(args.workers)
    
    jobs = []
    for fpath in jsonl_files:
        jobs.append(pool.apply_async(process_file, (fpath, args.tokenizer, tmp_dir)))
        
    pool.close()
    
    with open(args.output, 'wb') as out_f:
        for i, job in enumerate(jobs):
            tmp_path, tokens_count = job.get()
            total_tokens += tokens_count
            
            # Pindahkan data dari file sementara ke file akhir
            with open(tmp_path, 'rb') as tmp_f:
                # Copy in chunks of 10MB to be extremely safe with RAM
                while True:
                    chunk = tmp_f.read(10 * 1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    
            # Hapus file sementara
            os.remove(tmp_path)
            
            print(f"[{i+1}/{len(jsonl_files)}] Ditulis {tokens_count:,} token | Total: {total_tokens:,} token")
            
    pool.join()
    os.rmdir(tmp_dir)
    
    print("\n✅ SELESAI!")
    print(f"Total Token: {total_tokens:,}")
    print(f"Ukuran File: {os.path.getsize(args.output) / (1024**3):.2f} GB")
    print(f"Silakan upload file {args.output} ini ke Kaggle!")
    
    # Buat file meta kecil untuk loader di Kaggle agar tahu total tokennya
    meta_path = args.output + ".meta"
    with open(meta_path, 'w') as mf:
        json.dump({"total_tokens": total_tokens, "dtype": "uint16"}, mf)
    print(f"Meta file tersimpan di {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Folder berisi .jsonl batch")
    parser.add_argument("--output", type=str, required=True, help="Path output file .bin")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer_32k.model")
    parser.add_argument("--workers", type=int, default=8, help="Jumlah proses paralel")
    
    args = parser.parse_args()
    build_memmap(args)
