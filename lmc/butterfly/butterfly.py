import math
from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from lmc.utils.seeds import temp_seed


@torch.no_grad()
def get_batch_noise(model: "BaseModel", dataloader: DataLoader, noise_seed: int=None, loss_fn: callable = None) -> Dict[str, torch.Tensor]:
    """
    Get a random (based on the seed) batch from the dataloader and compute noise as the gradient 
    with respect to the provided loss.

    Args:
        model (BaseModel): Model to perturb
        dataloader (DataLoader): Dataloader to get the batch from
        noise_seed (int, optional): If provided, samples the batch from the dataloader with this seed. Defaults to None.
        loss_fn (callable, optional): Loss function to use. Defaults to cross entropy loss.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing the noise for each parameter
    """
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    if noise_seed is not None:
        with temp_seed(noise_seed):
            # Get a random batch from the dataloader
            batch = next(iter(dataloader))
    else:
        # Get a random batch from the dataloader without setting a seed
        batch = next(iter(dataloader))
    
    # Unpack inputs and targets (adjust if your dataloader provides a different structure)
    inputs, targets = batch

    # Move inputs and targets to the device of the model
    inputs, targets = inputs.to(model.device), targets.to(model.device)

    # Set model to evaluation mode
    model.eval()

    # Perform a forward pass
    outputs = model(inputs)

    # Compute the cross-entropy loss
    loss = loss_fn(outputs, targets)

    # Compute gradients w.r.t the model's parameters
    noise = {}
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            grad = torch.zeros_like(param)
        noise[name] = grad.clone()
        # if param.requires_grad:
        #     noise[name] = torch.autograd.grad(loss, param, retain_graph=True, create_graph=False)[0]
    model.zero_grad()
    return noise

@torch.no_grad()
def get_gaussian_noise(model: "BaseModel", noise_seed: int = None) -> Dict[str, torch.Tensor]:
    """_summary_

    Args:
        model (BaseModel): _description_
        config (_type_): _description_

    Returns:
        Dict[str, torch.Tensor]: _description_
    """
    def _get_noise_dct():
        model.zero_grad()
        noise_dct = dict()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                noise_dct[name] = torch.zeros_like(param)
                continue

            fan_in = 1.
            if param.ndim >= 2 :
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(param)
            noise = torch.empty_like(param)
            std = 1. / math.sqrt(fan_in)
            with torch.no_grad():
                torch.nn.init.normal_(noise, 0, std)
            noise_dct[name] = noise
        model.zero_grad()
        return noise_dct

    if noise_seed is not None:
        with temp_seed(noise_seed):
            noise_dct = _get_noise_dct()
    else:
        noise_dct = _get_noise_dct()
    return noise_dct
    


@torch.no_grad()
def perturb_model(model: nn.Module, noise_dct: OrderedDict, noise_multiplier: float = 1., inplace: bool = True) -> nn.Module:
    perturb_params = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            perturb_params[n] = p
            continue
        with torch.no_grad():
            param_noise = noise_dct[n]
            perturb_params[n] = p + param_noise*noise_multiplier
    if inplace: 
        model.load_state_dict(perturb_params)
    else:
        model = deepcopy(model)
        model.load_state_dict(perturb_params)
    
