import yaml

class SentinelConfig:
    def __init__(self, config_dict=None):
        if config_dict:
            for k, v in config_dict.items():
                setattr(self, k, v)
                
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)

def load_config(path: str = "config/small.yaml") -> SentinelConfig:
    return SentinelConfig.from_yaml(path)
