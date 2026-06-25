# SentinelLM v0.2

Proyek riset untuk mempelajari implementasi Transformer decoder-only dari nol menggunakan PyTorch murni. Tidak menggunakan framework training siap pakai seperti HuggingFace Trainer, PyTorch Lightning, DeepSpeed, atau Megatron.

## Arsitektur
- Positional Embedding: **RoPE (Rotary Positional Embedding)**
- Feed Forward: **SwiGLU**
- Normalization: **RMSNorm**
- Tokenizer: **SentencePiece (BPE)**

## Struktur Proyek
- `config/`: Konfigurasi arsitektur YAML (contoh: `tiny.yaml`).
- `dataset/`: Modul dataset multi-kategori (Curriculum Learning) dan skrip preprocessing.
- `evaluation/`: The Validation Suite (Perplexity, Next-Token Accuracy, Generation, Attention Heatmap).
- `model/`: Implementasi komponen Transformer murni secara modular.
- `tokenizer/`: Membangun *vocab* menggunakan SentencePiece.
- `training/`: Implementasi Kaggle-Resilient Trainer, AdamW, loss, dan Cosine LR scheduler.
- `inference/`: Proses autoregresif untuk text generation.
- `tools/`: The Corpus Inspector (Auditing data, Deduplikasi, Analisis Security Coverage).
- `utils/`: Logging dan konfigurasi environment.

## Cara Penggunaan

1. **Persiapan Dependensi**
   Buat virtual environment dan install requirements.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Membuat Dataset**
   ```bash
   python dataset/builder.py
   ```
   *Script ini akan men-generate sampel dummy (HTTP & Python) di `dataset/samples/dummy.txt`.*

3. **Training Tokenizer**
   ```bash
   python tokenizer/train_tokenizer.py
   ```
   *Akan menggunakan SentencePiece untuk membentuk `.model` dan `.vocab`.*

4. **Preprocessing Dataset**
   ```bash
   python dataset/preprocess.py
   ```
   *Mengubah teks menjadi tensor `train.pt` dan `val.pt`.*

5. **Melatih Model (Training)**
   ```bash
   python training/trainer.py
   ```
   *Target: melihat metrik loss (CrossEntropy) berkurang dari epoch ke epoch.*

6. **Menjalankan Inference**
   ```bash
   python inference/generate.py --prompt "GET /"
   ```
   *Menghasilkan prediksi token berdasarkan model yang sudah dilatih.*
