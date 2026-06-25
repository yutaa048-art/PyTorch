import os
import sentencepiece as spm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("TrainTokenizer")

def train(config_path="config/small.yaml"):
    config = load_config(config_path)
    if not os.path.exists(config.data_path):
        logger.error(f"Dataset tidak ditemukan di {config.data_path}. Jalankan dataset/builder.py terlebih dahulu.")
        return
        
    os.makedirs(os.path.dirname(config.tokenizer_path), exist_ok=True)
    
    # prefix untuk file model dan vocab
    model_prefix = os.path.splitext(config.tokenizer_path)[0]
    
    logger.info(f"Melatih tokenizer SentencePiece (BPE) dengan ukuran vocab {config.vocab_size}...")
    
    spm.SentencePieceTrainer.train(
        input=config.data_path,
        model_prefix=model_prefix,
        vocab_size=config.vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>"
    )
    
    logger.info(f"Tokenizer berhasil dilatih.")
    logger.info(f"Output: {model_prefix}.model dan {model_prefix}.vocab")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/small.yaml")
    args = parser.parse_args()
    train(args.config)
