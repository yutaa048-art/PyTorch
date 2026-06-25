import torch

def create_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    """
    Mengonfigurasi AdamW optimizer. Memisahkan parameter yang butuh weight decay (weight matrix di Linear)
    dari parameter yang tidak butuh weight decay (bias, LayerNorm/RMSNorm weights, embeddings).
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if param.dim() < 2 or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
