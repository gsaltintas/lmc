from dataclasses import dataclass
from pathlib import Path

from torch import nn, optim
from torch.utils.data import DataLoader


@dataclass
class TrainingElement:
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    train_loader: DataLoader
    train_eval_loader: DataLoader
    test_loader: DataLoader
    seed: int = 42
    loader_seed: int = 42
    aug_seed: int = 42
    optimal_acc: float = -1
    optimal_path: Path = None
    permutation = None
    prev_perm_wm = None
    prev_perm_am = None
    max_steps: int = None
    