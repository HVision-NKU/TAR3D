import math
import torch
from torch.optim.lr_scheduler import LambdaLR


class LambdaWarmUpCosineFactorScheduler:
    def __init__(self, max_decay_steps, lr_max, warm_up_steps=5000, f_start=1e-6, f_min=1e-3, f_max=1.0):
        self.max_decay_steps = max_decay_steps
        self.lr_max = lr_max
        self.warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max

    def schedule(self, step):
        if step < self.warm_up_steps:
            return self.f_start + (self.f_max - self.f_start) * (step / self.warm_up_steps)
        else:
            progress = (step - self.warm_up_steps) / (self.max_decay_steps - self.warm_up_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.f_min + (self.f_max - self.f_min) * cosine_decay



class WarmupCosineAnnealingLR(LambdaLR):
    def __init__(self, optimizer, warmup_steps, max_steps, base_lr, max_lr, last_epoch=-1):
        # Lambda function for learning rate scheduling
        def lr_lambda(step):
            step_tensor = torch.tensor(step, dtype=torch.float32)
            warmup_steps_tensor = torch.tensor(warmup_steps, dtype=torch.float32)
            max_steps_tensor = torch.tensor(max_steps, dtype=torch.float32)
            pi = torch.pi
            
            if step < warmup_steps:
                return (step_tensor / warmup_steps_tensor) * (max_lr / base_lr)
            else:
                progress = (step_tensor - warmup_steps_tensor) / (max_steps_tensor - warmup_steps_tensor)
                return 0.5 * (1 + torch.cos(progress * pi)) * (max_lr / base_lr)
        
        super().__init__(optimizer, lr_lambda, last_epoch)

