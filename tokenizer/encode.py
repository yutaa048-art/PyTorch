import os
import sentencepiece as spm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("Encode")

def get_tokenizer(config):
    if not os.path.exists(config.tokenizer_path):
        logger.error(f"Tokenizer tidak ditemukan di {config.tokenizer_path}.")
        raise FileNotFoundError(f"Missing {config.tokenizer_path}")
    
    sp = spm.SentencePieceProcessor()
    sp.load(config.tokenizer_path)
    return sp

def encode(text: str, config=None) -> list[int]:
    """
    Mengubah teks menjadi daftar token IDs.
    """
    if config is None: config = load_config("config/small.yaml")
    sp = get_tokenizer(config)
    return sp.encode_as_ids(text)

if __name__ == "__main__":
    sample = "GET /login HTTP/1.1"
    print(f"Text: {sample}")
    print(f"Encoded: {encode(sample)}")
