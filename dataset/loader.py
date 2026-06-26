import os
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("Loader")

class SentinelDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, seq_len: int):
        self.data = data_tensor
        self.seq_len = seq_len
        
    def __len__(self):
        # Gunakan non-overlapping chunks (stride = seq_len)
        return len(self.data) // self.seq_len
        
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        x = self.data[start_idx : start_idx + self.seq_len]
        y = self.data[start_idx + 1 : start_idx + self.seq_len + 1]
        
        return {
            "input_ids": x,
            "target_ids": y
        }

def get_dataloaders(config):
    base_dir = os.path.dirname(config.data_path)
    categories = getattr(config, "data_category", "all")
    
    if categories == "all":
        categories_list = ["code", "docs", "security", "writeups"]
    else:
        categories_list = [categories]
        
    train_tensors = []
    val_tensors = []
    
    for cat in categories_list:
        train_path = os.path.join(base_dir, cat, "train.pt")
        val_path = os.path.join(base_dir, cat, "val.pt")
        
        if os.path.exists(train_path):
            train_tensors.append(torch.load(train_path, weights_only=True))
        if os.path.exists(val_path):
            val_tensors.append(torch.load(val_path, weights_only=True))
            
    # Fallback: Jika train.pt ada di base_dir (dataset tunggal)
    if os.path.exists(os.path.join(base_dir, "train.pt")) and not train_tensors:
        train_tensors.append(torch.load(os.path.join(base_dir, "train.pt"), weights_only=True))
        if os.path.exists(os.path.join(base_dir, "val.pt")):
            val_tensors.append(torch.load(os.path.join(base_dir, "val.pt"), weights_only=True))
            
    if not train_tensors:
        raise FileNotFoundError("Dataset train.pt tidak ditemukan di kategori yang diminta maupun di base directory. Jalankan preprocess.py terlebih dahulu.")
        
    # Gabungkan tensor
    train_data = torch.cat(train_tensors)
    
    train_ds = SentinelDataset(train_data, config.max_position_embeddings)
    
    if len(train_ds) == 0:
        raise ValueError(
            f"Dataset training kosong! Data hanya {len(train_data)} token, "
            f"tetapi seq_len={config.max_position_embeddings}. "
            f"Pastikan sudah menjalankan preprocess.py dengan dataset yang cukup besar."
        )
    
    # drop_last=False agar dataset kecil (lokal/dummy) tidak menghasilkan 0 samples
    # pin_memory=True  : Mengunci halaman RAM agar transfer ke GPU lebih cepat (zero-copy DMA)
    # num_workers=4    : Prefetch data di 4 proses CPU paralel agar GPU tidak menunggu
    is_main_process = True  # Bisa diubah ke False jika menggunakan multi-processing trainer
    num_workers = 4 if is_main_process else 0
    train_dl = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)  # Hindari spawn ulang worker tiap epoch
    )
    
    if val_tensors:
        val_data = torch.cat(val_tensors)
        val_ds = SentinelDataset(val_data, config.max_position_embeddings)
        if len(val_ds) > 0:
            val_dl = DataLoader(
                val_ds, 
                batch_size=config.batch_size, 
                shuffle=False, 
                drop_last=False,
                pin_memory=True,
                num_workers=num_workers,
                persistent_workers=(num_workers > 0)
            )
        else:
            logger.warning(f"Val dataset hanya {len(val_data)} token (terlalu kecil untuk seq_len={config.max_position_embeddings}). Validation dinonaktifkan.")
            val_dl = None
    else:
        val_dl = None
    
    return train_dl, val_dl

if __name__ == "__main__":
    config = load_config("config/tiny.yaml")
    train_dl, val_dl = get_dataloaders(config)
    for batch in train_dl:
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Target shape: {batch['target_ids'].shape}")
        break
