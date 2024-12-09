import random
from contextlib import contextmanager

import numpy as np
import torch
import os

@contextmanager
def temp_seed(seed: int):
    """
    Context manager to temporarily set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed (int): The temporary seed to set.

    Usage:
        with temp_seed(42):
            # Code that requires the temporary seed
    """
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


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # set env variable for use_deterministic_algorithms
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
