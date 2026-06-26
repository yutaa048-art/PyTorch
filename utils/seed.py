import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    """
    Mengatur seed untuk semua pustaka agar hasil eksperimen dapat direproduksi.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # FIX: cudnn.deterministic=True konflik dengan FlashAttention (non-deterministic kernel)
    # dan mematikan cudnn.benchmark yang penting untuk optimasi kecepatan kernel.
    # Untuk training LLM, reprodusibilitas cukup dijamin oleh seed Torch saja.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # Biarkan CUDA memilih kernel tercepat otomatis
