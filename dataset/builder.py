import os
import random
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from dataset.http_generator import generate_http_dataset
from dataset.python_generator import generate_python_dataset
from dataset.chatbot_generator import generate_chatbot_dataset

logger = get_logger("DatasetBuilder")

def load_config(config_path="config/small.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_dummy_dataset(config_path="config/small.yaml"):
    config = load_config(config_path)
    data_path = config.get("data_path", "dataset/samples/dummy.txt")
    
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    logger.info("Men-generate dataset dummy besar...")
    
    # Generate 50.000 kombinasi agar vocab 16k sentencepiece dapat bekerja maksimal
    samples = []
    samples.extend(generate_http_dataset(50000))
    samples.extend(generate_python_dataset(50000))
    samples.extend(generate_chatbot_dataset(50000))
        
    random.shuffle(samples)
    
    with open(data_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(samples))
        
    logger.info(f"Berhasil men-generate {len(samples)} sampel dummy ke {data_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/small.yaml")
    args = parser.parse_args()
    
    build_dummy_dataset(args.config)
