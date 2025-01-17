import random
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch

def seed_worker(loader_seed):
    def seed_worker_(worker_id):
        worker_seed = loader_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return seed_worker_


def seed_worker(loader_seed):
    def seed_worker_(worker_id):
        worker_seed = loader_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return seed_worker_


@contextmanager
def temp_seed(seed: Optional[int]):
    """
    Context manager to temporarily set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed (int): The temporary seed to set.

    Usage:
        with temp_seed(42):
            # Code that requires the temporary seed
    """
    if seed is None:
        return
    # Save current RNG states
    state_python = random.getstate()
    state_numpy = np.random.get_state()
    state_torch = torch.random.get_rng_state()
    if torch.cuda.is_available():
        state_cuda = torch.cuda.get_rng_state_all()

    try:
        # Set the temporary seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Yield control back to the caller
        yield
    finally:
        # Restore original RNG states
        random.setstate(state_python)
        np.random.set_state(state_numpy)
        torch.random.set_rng_state(state_torch)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state_cuda)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
