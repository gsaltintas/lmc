import logging
import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lmc.experiment_config import PerturbedTrainer
from lmc.utils.seeds import temp_seed
from lmc.utils.setup_training import setup_loader
from lmc.utils.training_element import params_l2

logger = logging.getLogger(__name__)  # Add this line to define the logger


def _should_perturb_parameter(
    param_name: str, dont_perturb_module_patterns: Optional[List[str]]
) -> bool:
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


def get_perturbed_layers(
    model, dont_perturb_module_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Returns a set of parameter names that should be perturbed
    """
    layers = []
    for name, param in model.named_parameters():
        if param.requires_grad and _should_perturb_parameter(
            name, dont_perturb_module_patterns
        ):
            layers.append(name)
    return layers


def get_batch_noise(
    model: "BaseModel", dataloader: DataLoader, loss_fn: nn.Module, layers: List[str]
) -> Dict[str, torch.Tensor]:
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
        batch = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward pass and loss calculation
        outputs = model(**batch)

        # Use model's built-in loss if available
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        elif hasattr(outputs, "logits") and "labels" in batch:
            if loss_fn is not None:
                loss = loss_fn(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    batch["labels"].view(-1),
                )
            else:
                loss = nn.CrossEntropyLoss()(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    batch["labels"].view(-1),
                )
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
        if name not in layers:
            logger.info(f"Not generating any noise for parameter ({name}).")
            noise[name] = torch.zeros_like(param)
        elif grad is None:
            grad = torch.zeros_like(param)
        else:
            noise[name] = grad.clone()
    model.zero_grad()
    return noise


@torch.no_grad()
def get_gaussian_noise(
    model: "BaseModel", layers: List[str]
) -> Dict[str, torch.Tensor]:
    """_summary_

    Args:
        model (BaseModel): _description_
        config (_type_): _description_

    Returns:
        Dict[str, torch.Tensor]: _description_
    """
    model.zero_grad()
    std = model.get_init_stds(include_constant_params=True)
    # we want to be able to perturb the norm weights and biases, so use the next layer's fan_in std
    # reasoning: noise should be scaled according to distribution, each neuron is the sum of fan_in products of form w*x
    # x is the output of a normalization layer with 0 mean, 1 std at init
    # w is scaled proportional to sqrt(fan_in), so that variance of the sum is 1
    # we want the noise effect to be the same in x as in w
    # when adding noise to w, (w+n)*x = wx + xn has scale proportional to w
    # when adding noise to x, w*(x+n) = wx + wn should also have the same scale
    noise_dct = dict()
    for name, param in model.named_parameters():
        if name not in layers:
            logger.info(f"Not generating any noise for parameter ({name}).")
            noise_dct[name] = torch.zeros_like(param)
            continue
        noise = torch.empty_like(param)
        with torch.no_grad():
            torch.nn.init.normal_(noise, 0, std[name])
        noise_dct[name] = noise
    model.zero_grad()
    return noise_dct


@torch.no_grad()
def perturb_model(
    model: nn.Module, noise_dct: Dict[str, torch.Tensor], inplace: bool = True
) -> nn.Module:
    perturb_params = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            perturb_params[n] = p
            continue
        with torch.no_grad():
            perturb_params[n] = p + noise_dct[n]
    if inplace:
        model.load_state_dict(perturb_params, strict=False)
    else:
        model = deepcopy(model)
        model.load_state_dict(perturb_params)
    return model


def get_all_init_l2s(model, layers) -> Tuple[float, Dict[str, float]]:
    """Return standard deviation of specified layers at init,
    i.e. the expected L2 norm of the parameters minus their mean at init"""
    init_l2s = {}
    total_sqsum = 0
    stds = model.get_init_stds(include_constant_params=True)
    for k, v in model.named_parameters():
        if k in layers:
            sqsum = stds[k] ** 2 * v.nelement()
            init_l2s[k] = sqsum**0.5
            total_sqsum += sqsum
        else:
            init_l2s[k] = 0
    return total_sqsum**0.5, init_l2s


def get_average_grad_norm(
    model: "BaseModel",
    dataloader: DataLoader,
    loss_fn: callable = None,
    num_datapoints: int = 1,
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
            batch = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass and loss calculation
            outputs = model(**batch)

            # Use model's built-in loss if available
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            elif hasattr(outputs, "logits") and "labels" in batch:
                if loss_fn is not None:
                    loss = loss_fn(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        batch["labels"].view(-1),
                    )
                else:
                    loss = nn.CrossEntropyLoss()(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        batch["labels"].view(-1),
                    )
            else:
                raise ValueError("Could not compute loss for language model")

        else:
            # Handle vision model batch
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
            else:
                raise ValueError(
                    f"Unexpected batch format for vision model: {type(batch)}"
                )

            inputs = inputs.to(model.device)
            targets = targets.to(model.device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        loss.backward()
        processed_batches += 1

    # Calculate gradient norm
    total_norm = 0.0
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
        avg_grad_norm = torch.sqrt(torch.tensor(total_norm)) / (
            param_count * processed_batches
        )

    model.zero_grad()
    return avg_grad_norm, param_count


def scale_noise(
    noise_dct: Dict[str, torch.Tensor],
    model,
    layers,
    scale: float,
    normalize: bool,
    scale_to_init_if_normalized: bool,
):
    """Normalize the noise dict to have a fixed total l2 length"""
    # log expected l2 (std) and per-layer std
    expected_l2, _ = get_all_init_l2s(model, layers)
    # log l2 of noise before scaling
    actual_norm = params_l2(noise_dct.values())
    # do the scaling
    if normalize:
        scale /= actual_norm
        if scale_to_init_if_normalized:
            scale *= expected_l2
    for name, n in noise_dct.items():
        noise_dct[name] = n * scale
    return noise_dct


def sample_noise_and_perturb(
    config: PerturbedTrainer,
    model,
    perturb_seed: Optional[int],
    loss_fn: nn.Module,
    ind: int,
    tokenizer: AutoTokenizer = None,
):
    layers = get_perturbed_layers(model, config.dont_perturb_module_patterns)
    if config.perturb_mode == "batch":
        dl = setup_loader(
            config.data,
            train=True,
            evaluate=False,
            loader_seed=perturb_seed,
            tokenizer=tokenizer,
        )
        noise = get_batch_noise(model, dataloader=dl, loss_fn=loss_fn, layers=layers)
    elif config.perturb_mode == "gaussian":
        with temp_seed(perturb_seed):
            noise = get_gaussian_noise(model, layers=layers)
    else:
        raise ValueError(f"Invalid noise mode {config.perturb_mode}.")
    noise = scale_noise(
        noise,
        model,
        layers,
        config.perturb_scale,
        config.normalize_perturb,
        config.scale_to_init_if_normalized,
    )
    # log l2 and per-layer l2 of scaled noise
    log_dct = {f"static/noise_l2/{ind}/total": params_l2(noise.values())}
    if config.log_per_layer_l2:
        for k, v in noise.items():
            log_dct[f"static/noise_l2/{ind}/layer/{k}"] = torch.linalg.norm(v.flatten())
    perturb_model(model, noise)
    return log_dct
    return log_dct
