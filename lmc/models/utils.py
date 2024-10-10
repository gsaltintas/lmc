import numpy as np
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Counts the total number of trainable parameters of a model"""
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def outer_product(a, b) -> np.array:
    return a @ b.T
