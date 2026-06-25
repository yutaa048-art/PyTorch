import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from tokenizer.encode import encode
from tokenizer.decode import decode

@pytest.fixture
def config():
    return load_config("config/tiny.yaml")

def test_tokenizer_encode_decode(config):
    # Skip test if tokenizer hasn't been built
    if not os.path.exists(config.tokenizer_path):
        pytest.skip(f"Tokenizer not found at {config.tokenizer_path}. Run dataset builder and tokenizer training first.")
        
    original_text = "GET /api/users HTTP/1.1\nHost: localhost"
    
    # Test encode
    ids = encode(original_text, config)
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
    
    # Test decode
    decoded_text = decode(ids, config)
    assert isinstance(decoded_text, str)
    assert decoded_text == original_text
