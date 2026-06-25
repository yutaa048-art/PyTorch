import math
import torch

class CosineLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int, min_lr: float = 1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr * self.current_step / self.warmup_steps
        elif self.current_step > self.max_steps:
            lr = self.min_lr
        else:
            # Cosine decay
            decay_ratio = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.initial_lr - self.min_lr)
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
        
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'initial_lr': self.initial_lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps
        }
        
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.initial_lr = state_dict['initial_lr']
        self.min_lr = state_dict['min_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.max_steps = state_dict['max_steps']
