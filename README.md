# SentinelLM v0.1

Proyek riset untuk mempelajari implementasi Transformer decoder-only dari nol menggunakan PyTorch murni. Tidak menggunakan framework training siap pakai seperti HuggingFace Trainer, PyTorch Lightning, DeepSpeed, atau Megatron.

## Arsitektur
- Positional Embedding: **RoPE (Rotary Positional Embedding)**
- Feed Forward: **SwiGLU**
- Normalization: **RMSNorm**
- Tokenizer: **SentencePiece (BPE)**

## Struktur Proyek
- `config/`: Berisi `configs.py` untuk parameter model.
- `dataset/`: 
  - `builder.py`: Menghasilkan dataset dummy (HTTP logs & Python code).
  - `preprocess.py`: Mengubah teks ke tensor.
  - `loader.py`: Menghasilkan `{"input_ids": ..., "target_ids": ...}`.
- `model/`: Implementasi komponen Transformer murni secara modular.
- `tokenizer/`: Membangun *vocab* menggunakan SentencePiece.
- `training/`: Implementasi loop PyTorch murni, AdamW, loss, dan LR scheduler.
- `inference/`: Proses autoregresif untuk text generation.
- `utils/`: Logging dan seed.

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
