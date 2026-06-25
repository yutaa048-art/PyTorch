import pytest
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.loader import SentinelDataset

def test_dataset_item():
    seq_len = 5
    total_tokens = 20
    dummy_data = torch.arange(total_tokens)
    
    dataset = SentinelDataset(dummy_data, seq_len)
    
    assert len(dataset) == total_tokens - seq_len
    
    item = dataset[0]
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "target_ids" in item
    
    # X: [0, 1, 2, 3, 4]
    assert torch.all(item["input_ids"] == torch.tensor([0, 1, 2, 3, 4]))
    # Y: [1, 2, 3, 4, 5] (Bergeser 1)
    assert torch.all(item["target_ids"] == torch.tensor([1, 2, 3, 4, 5]))
