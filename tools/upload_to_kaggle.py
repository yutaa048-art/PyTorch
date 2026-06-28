import os
import glob
import json
import shutil
import subprocess

def prepare_and_upload():
    print("🚀 Memulai persiapan upload dataset ke Kaggle...")
    
    # 1. Siapkan direktori staging
    stage_dir = "kaggle_dataset_stage"
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)
    
    # 2. Setup metadata Kaggle Dataset
    dataset_name = "sentinellm-500m-corpus"
    # Dapatkan username dari kaggle config (menggunakan subprocess)
    try:
        username = subprocess.check_output(["./venv/bin/kaggle", "config", "view"], text=True)
        # Ambil username dari output config
        user_id = [line for line in username.split('\n') if 'username' in line][0].split(':')[1].strip().strip("'")
    except Exception as e:
        print("Gagal membaca username dari kaggle cli, memakai default 'yuta'")
        user_id = "yuta"

    metadata = {
        "title": "SentinelLM 500M Pretrain Corpus",
        "id": f"{user_id}/{dataset_name}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    with open(os.path.join(stage_dir, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # 3. Kumpulkan file dataset besar (menggunakan Hard Link agar tidak makan disk 16GB lagi)
    print("📦 Mengumpulkan corpus.bin (16GB)...")
    corpus_path = "/home/yuta/RIset Corpus/Sentinel_Corpus/corpus.bin"
    meta_path = "/home/yuta/RIset Corpus/Sentinel_Corpus/corpus.bin.meta"
    
    try:
        os.link(corpus_path, os.path.join(stage_dir, "corpus.bin"))
        os.link(meta_path, os.path.join(stage_dir, "corpus.bin.meta"))
    except Exception:
        print("Hard link gagal, menyalin file (bisa memakan waktu lama)...")
        shutil.copy2(corpus_path, stage_dir)
        shutil.copy2(meta_path, stage_dir)
        
    # 4. Kumpulkan tokenizer
    print("📦 Mengumpulkan Tokenizer...")
    shutil.copy2("tokenizer/tokenizer_32k.model", stage_dir)
    shutil.copy2("tokenizer/tokenizer_32k.vocab", stage_dir)
    
    # 5. Zip Source Code (Mengabaikan venv dan .git)
    print("📦 Melakukan zip pada source code SentinelLM...")
    code_zip = os.path.join(stage_dir, "sentinellm_code.zip")
    try:
        subprocess.run([
            "zip", "-r", "-q", code_zip, ".", 
            "-x", "venv/*", "-x", ".git/*", "-x", "checkpoints/*", "-x", "kaggle_dataset_stage/*"
        ], check=True)
    except Exception as e:
        print(f"Gagal melakukan zip: {e}")
    
    print("\n✅ Semua file siap di folder 'kaggle_dataset_stage'!")
    
    # 6. Upload menggunakan Kaggle CLI
    print(f"🚀 Memulai upload ke Kaggle (ID: {metadata['id']})...")
    
    try:
        # Cek apakah dataset sudah ada
        subprocess.run(
            ["./venv/bin/kaggle", "datasets", "create", "-p", stage_dir, "-r", "tar"], 
            check=True
        )
        print("\n🎉 BINGO! Dataset berhasil dibuat di Kaggle!")
    except subprocess.CalledProcessError:
        print("\nDataset sepertinya sudah ada, mencoba mengupdate versi baru...")
        try:
            subprocess.run(
                ["./venv/bin/kaggle", "datasets", "version", "-p", stage_dir, "-m", "Update corpus v2", "-r", "tar"], 
                check=True
            )
            print("\n🎉 BINGO! Dataset berhasil diupdate di Kaggle!")
        except Exception as e:
            print(f"❌ Gagal mengupload: {e}")

if __name__ == "__main__":
    prepare_and_upload()
