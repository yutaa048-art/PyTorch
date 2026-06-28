import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("Loader")


class SentinelMemmapDataset(Dataset):
    """
    Dataset berbasis numpy memmap dengan dukungan Curriculum Context.
    seq_len bisa diubah secara dinamis oleh trainer tanpa membuat ulang objek.
    """
    def __init__(self, memmap_path: str, seq_len: int, total_tokens: int):
        self.memmap_path = memmap_path
        self.seq_len = seq_len
        self.total_tokens = total_tokens

        self.data = np.memmap(self.memmap_path, dtype=np.uint16, mode='r',
                              shape=(self.total_tokens,))

    def set_seq_len(self, new_seq_len: int):
        """Dipanggil oleh CurriculumScheduler saat transisi fase."""
        self.seq_len = new_seq_len

    def __len__(self):
        return self.total_tokens // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        chunk = self.data[start_idx : start_idx + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return {"input_ids": x, "target_ids": y}


def get_dataloaders(config, seq_len_override: int = None):
    """
    Args:
        config: SentinelConfig
        seq_len_override: Jika diberikan, gunakan ini sebagai seq_len awal
                         (dipakai oleh CurriculumScheduler).
    """
    data_path = config.data_path
    seq_len = seq_len_override or config.max_position_embeddings

    if data_path.endswith('.bin'):
        meta_path = data_path + ".meta"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"File meta {meta_path} tidak ditemukan! "
                "Pastikan Anda sudah menjalankan build_memmap.py"
            )

        with open(meta_path, 'r') as f:
            meta = json.load(f)
            total_tokens = meta['total_tokens']

        logger.info(f"Memmap Dataset: {data_path} ({total_tokens:,} tokens, seq_len={seq_len})")

        train_ds = SentinelMemmapDataset(data_path, seq_len, total_tokens)

        num_workers = 4
        train_dl = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0)
        )
        return train_dl, None

    else:
        # Fallback ke sistem lama (.pt)
        logger.warning("Memuat menggunakan sistem lama (torch.load). TIDAK DISARANKAN untuk data > 1GB.")

        base_dir = os.path.dirname(config.data_path)
        train_tensors = []
        for cat in ["code", "docs", "security", "writeups"]:
            train_path = os.path.join(base_dir, cat, "train.pt")
            if os.path.exists(train_path):
                train_tensors.append(torch.load(train_path, map_location='cpu', weights_only=True))

        if os.path.exists(os.path.join(base_dir, "train.pt")) and not train_tensors:
            train_tensors.append(torch.load(os.path.join(base_dir, "train.pt"),
                                            map_location='cpu', weights_only=True))

        if not train_tensors:
            raise FileNotFoundError("Dataset tidak ditemukan.")

        train_data = torch.cat(train_tensors)

        class LegacyDataset(Dataset):
            def __init__(self, data_tensor, sl):
                self.data = data_tensor
                self.seq_len = sl
            def __len__(self):
                return len(self.data) // self.seq_len
            def __getitem__(self, idx):
                s = idx * self.seq_len
                x = self.data[s : s + self.seq_len].clone().detach().long()
                y = self.data[s + 1 : s + self.seq_len + 1].clone().detach().long()
                return {"input_ids": x, "target_ids": y}

        train_ds = LegacyDataset(train_data, seq_len)
        train_dl = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, drop_last=False, pin_memory=True, num_workers=2)
        return train_dl, None


if __name__ == "__main__":
    config = load_config("config/500m_deep.yaml")
    train_dl, _ = get_dataloaders(config, seq_len_override=512)
    for batch in train_dl:
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Target shape: {batch['target_ids'].shape}")
        break
