import torch


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']

def reset_base_lrs(optimizer: torch.optim.Optimizer, lr: float, scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
    optimizer.defaults["lr"] = lr
    for param_group in optimizer.param_groups:
        param_group['inital_lr'] = lr
    scheduler.base_lrs = [lr] * len(optimizer.param_groups)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
