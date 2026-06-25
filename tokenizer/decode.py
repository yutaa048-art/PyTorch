import os
import sentencepiece as spm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("Decode")

def get_tokenizer(config):
    if not os.path.exists(config.tokenizer_path):
        logger.error(f"Tokenizer tidak ditemukan di {config.tokenizer_path}.")
        raise FileNotFoundError(f"Missing {config.tokenizer_path}")
    
    sp = spm.SentencePieceProcessor()
    sp.load(config.tokenizer_path)
    return sp

def decode(ids: list[int], config=None) -> str:
    """
    Mengubah daftar token IDs menjadi teks.
    """
    if config is None: config = load_config("config/small.yaml")
    sp = get_tokenizer(config)
    return sp.decode_ids(ids)

if __name__ == "__main__":
    from encode import encode
    sample = "def hello_world_1():"
    ids = encode(sample)
    print(f"IDs: {ids}")
    print(f"Decoded: {decode(ids)}")
