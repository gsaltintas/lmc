import logging
import math
import re
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

from lmc.utils.seeds import temp_seed

logger = logging.getLogger(__name__)  # Add this line to define the logger


def _should_perturb_parameter(param_name: str, dont_perturb_module_patterns: Optional[List[str]] = None) -> bool:
    """
    Check if a parameter should be perturbed based on its name and the dont_perturb_module_patterns.
    
    Args:
        param_name (str): Name of the parameter
        
    Returns:
        bool: True if the parameter should be perturbed, False otherwise
    """
    if dont_perturb_module_patterns is None or len(dont_perturb_module_patterns) == 0:
        return True
    for pattern in dont_perturb_module_patterns:
        if re.match(pattern, param_name):
            return False
    return True

def get_batch_noise(model: "BaseModel", dataloader: DataLoader, noise_seed: int=None, loss_fn: callable = None, dont_perturb_patterns: List[str] = None) -> Dict[str, torch.Tensor]:
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
    model.zero_grad()
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    batch = next(iter(dataloader))
    if model.is_language_model:
        # Handle language model batch
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass and loss calculation
        outputs = model(**batch)
        
        # Use model's built-in loss if available
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        elif hasattr(outputs, "logits") and "labels" in batch:
            if loss_fn is not None:
                loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                batch["labels"].view(-1))
            else:
                loss = nn.CrossEntropyLoss()(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                            batch["labels"].view(-1))
        else:
            raise ValueError("Could not compute loss for language model")
            
    else:
        # Unpack inputs and targets (adjust if your dataloader provides a different structure)
        inputs, targets = batch

        # Move inputs and targets to the device of the model
        inputs, targets = inputs.to(model.device), targets.to(model.device)

        # Perform a forward pass
        outputs = model(inputs)

        # Compute the cross-entropy loss
        loss = loss_fn(outputs, targets)
    loss.backward()

    # Compute gradients w.r.t the model's parameters
    noise = {}
    for name, param in model.named_parameters():
        grad = param.grad
        if not _should_perturb_parameter(name, dont_perturb_patterns):
            logger.info(f"Not generating any noise for parameter ({name}).")
            noise[name] = torch.zeros_like(param)
        elif grad is None:
            grad = torch.zeros_like(param)
        else:
            noise[name] = grad.clone()

    model.zero_grad()
    return noise

@torch.no_grad()
def get_gaussian_noise(model: "BaseModel", noise_seed: int = None, dont_perturb_patterns:Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
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
            if not param.requires_grad or not _should_perturb_parameter(name, dont_perturb_patterns):
                logger.info(f"Not generating any noise for parameter ({name}).")
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
    return model

def get_noise_l2(noise_dct: OrderedDict) -> float:
    noise_l2 = 0.
    for n, p in noise_dct.items():
        noise_l2 += p.pow(2).sum().item()
    return noise_l2

def get_average_grad_norm(
    model: "BaseModel", 
    dataloader: DataLoader, 
    loss_fn: callable = None, 
    num_datapoints: int = 1
) -> Tuple[float, int]:
    """
    Calculate the gradient norm of a model over specified number of datapoints.
    Handles both vision and language models.
    
    Args:
        model: The neural network model
        dataloader: DataLoader containing the input data
        loss_fn: Loss function (defaults to CrossEntropyLoss if None)
        num_datapoints: Number of batches to process (default=1), pass -1 to iterate through all points
    
    Returns:
        tuple[float, int]: A tuple containing:
            - Average L2 norm of the gradients
            - Total number of parameters with a gradient in the model
    """
    model.zero_grad()
    
    if loss_fn is None and not model.is_language_model:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    processed_batches = 0
    
    for i, batch in enumerate(dataloader):
        if i != -1 and i >= num_datapoints:
            break
            
        if model.is_language_model:
            # Handle language model batch
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass and loss calculation
            outputs = model(**batch)
            
            # Use model's built-in loss if available
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            elif hasattr(outputs, "logits") and "labels" in batch:
                if loss_fn is not None:
                    loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                 batch["labels"].view(-1))
                else:
                    loss = nn.CrossEntropyLoss()(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                                batch["labels"].view(-1))
            else:
                raise ValueError("Could not compute loss for language model")
                
        else:
            # Handle vision model batch
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
            else:
                raise ValueError(f"Unexpected batch format for vision model: {type(batch)}")
                
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        
        loss.backward()
        processed_batches += 1
    
    # Calculate gradient norm
    total_norm = 0.
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += param.numel()
    
    # Divide by both number of parameters and number of processed batches
    if processed_batches == 0 or param_count == 0:
        avg_grad_norm = 0.0
    else:
        avg_grad_norm = torch.sqrt(torch.tensor(total_norm)) / (param_count * processed_batches)
    
    model.zero_grad()
    return avg_grad_norm, param_count

def normalize_noise(noise_dct: OrderedDict[str, torch.Tensor], l2: float):
    """ Normalize the noise dict to have a total l2 length """
    total_norm = torch.linalg.norm(parameters_to_vector(noise_dct.values()))
    norm_factor = l2 / total_norm
    for name, n in noise_dct.items():
        noise_dct[name] = n * norm_factor

    act_norm = torch.linalg.norm(parameters_to_vector(noise_dct.values()))
    assert torch.allclose(act_norm, torch.tensor(l2)), f"Noise norm ({act_norm}) and desired {l2}."
    return noise_dct