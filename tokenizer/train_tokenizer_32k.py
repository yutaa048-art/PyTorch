"""
SentinelLM — Train Tokenizer 32K dari Corpus JSONL
===================================================
Melatih SentencePiece BPE tokenizer dengan vocab 32K
dari file .jsonl hasil cleaner.py.

Usage:
    python tokenizer/train_tokenizer_32k.py \
        --input dataset/clean_corpus_10B \
        --output tokenizer/tokenizer_32k \
        --vocab_size 32000
"""
import os
import json
import glob
import argparse
import tempfile
import sentencepiece as spm


def extract_text_from_jsonl(input_dir: str, max_chars: int = 500_000_000) -> str:
    """
    Ekstrak teks dari semua file .jsonl di direktori.
    SentencePiece membutuhkan plain text file sebagai input.
    
    Kita batasi max_chars agar RAM tidak meledak saat training tokenizer.
    5 Miliar karakter ≈ 5GB RAM, cukup representatif untuk 32K vocab.
    """
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    
    if not jsonl_files:
        # Fallback: coba baca .txt files langsung
        txt_files = sorted(glob.glob(os.path.join(input_dir, "**/*.txt"), recursive=True))
        if not txt_files:
            raise FileNotFoundError(f"Tidak ditemukan file .jsonl atau .txt di {input_dir}")
        jsonl_files = txt_files
    
    print(f"Ditemukan {len(jsonl_files)} file untuk training tokenizer")
    
    # Buat temporary file untuk SentencePiece
    tmp_path = os.path.join(os.path.dirname(input_dir), "_tokenizer_training_data.txt")
    total_chars = 0
    
    with open(tmp_path, 'w', encoding='utf-8') as out:
        for fpath in jsonl_files:
            print(f"  Membaca: {os.path.basename(fpath)}")
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Coba parse sebagai JSON
                    try:
                        data = json.loads(line)
                        text = data.get("content", data.get("text", ""))
                    except json.JSONDecodeError:
                        text = line  # Fallback: treat as plain text
                        
                    # Filter null characters yang dibenci SentencePiece
                    if text:
                        text = text.replace('\x00', '')

                    
                    if text:
                        out.write(text + "\n")
                        total_chars += len(text)
                        
                    if total_chars >= max_chars:
                        print(f"  Mencapai batas {max_chars/1e9:.1f}GB, berhenti membaca.")
                        break
                
                if total_chars >= max_chars:
                    break
    
    print(f"Total karakter untuk training: {total_chars/1e9:.2f} GB")
    return tmp_path


def train_tokenizer(input_path: str, output_prefix: str, vocab_size: int):
    """Latih SentencePiece BPE tokenizer."""
    
    print(f"\nMemulai training SentencePiece BPE...")
    print(f"  Vocab size  : {vocab_size}")
    print(f"  Input       : {input_path}")
    print(f"  Output      : {output_prefix}.model")
    
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        
        # Optimisasi untuk kode dan teks teknis
        character_coverage=0.9995,       # Tinggi untuk karakter Unicode
        byte_fallback=True,              # Fallback ke byte untuk karakter langka
        split_by_unicode_script=True,    # Pisahkan CJK, Latin, dll
        split_by_whitespace=True,
        split_digits=True,               # Setiap digit jadi token terpisah (penting untuk angka)
        max_sentence_length=16384,       # Dokumen panjang
        num_threads=os.cpu_count(),      # Gunakan semua CPU
        train_extremely_large_corpus=True,  # Optimisasi RAM untuk corpus besar
    )
    
    print(f"\n✅ Tokenizer berhasil dilatih!")
    print(f"   Model : {output_prefix}.model")
    print(f"   Vocab : {output_prefix}.vocab")
    
    # Quick test
    sp = spm.SentencePieceProcessor()
    sp.load(f"{output_prefix}.model")
    
    test_texts = [
        "def scan_port(host, port):",
        "SELECT * FROM users WHERE id = 1;",
        "import torch.nn.functional as F",
        "The vulnerability CVE-2025-1234 allows remote code execution.",
    ]
    
    print(f"\n📝 Test Tokenisasi (vocab={sp.get_piece_size()}):")
    for text in test_texts:
        tokens = sp.encode(text, out_type=str)
        ids = sp.encode(text)
        print(f"  Input  : {text}")
        print(f"  Tokens : {tokens}")
        print(f"  IDs    : {ids}")
        print(f"  Count  : {len(ids)} tokens")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentencePiece 32K tokenizer")
    parser.add_argument("--input", type=str, required=True,
                        help="Path ke direktori berisi .jsonl atau .txt")
    parser.add_argument("--output", type=str, default="tokenizer/tokenizer_32k",
                        help="Prefix output (tanpa .model)")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Ukuran vocabulary (default: 32000)")
    parser.add_argument("--max_chars", type=int, default=500_000_000,
                        help="Maksimum karakter yang dibaca (default: 500MB)")
    
    args = parser.parse_args()
    
    # Ekstrak teks dari JSONL
    tmp_text_path = extract_text_from_jsonl(args.input, args.max_chars)
    
    try:
        # Latih tokenizer
        train_tokenizer(tmp_text_path, args.output, args.vocab_size)
    finally:
        # Bersihkan temporary file
        if os.path.exists(tmp_text_path):
            os.remove(tmp_text_path)
            print(f"🧹 Temporary file dihapus: {tmp_text_path}")
