import math
from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import torch
from torch import nn


@torch.no_grad()
def get_gaussian_noise(model: "BaseModel") -> Dict[str, torch.Tensor]:
    """_summary_

    Args:
        model (BaseModel): _description_
        config (_type_): _description_

    Returns:
        Dict[str, torch.Tensor]: _description_
    """
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
    
