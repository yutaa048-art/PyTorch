import torch
import torch.nn as nn
from utils.config import load_config
from model.model import SentinelLM

config = load_config("config/tiny.yaml")
model = SentinelLM(config).to('cuda')
model = nn.DataParallel(model)

x = torch.randint(0, 1000, (16, 10)).to('cuda')
try:
    out = model(x)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
