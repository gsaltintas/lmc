import logging
import math
from typing import Dict, Generator, List, Literal, Union

import numpy as np
import ot
import torch
import torch.utils
import torch.utils.data
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .alignment import solve_for_perms
from .utils import PermSpec, PermType, get_permutation_sizes, outer_product

logger = logging.getLogger("act-align")

@torch.no_grad()
def get_activations_yield(
    models: nn.Module, dataloader: torch.utils.data.DataLoader
) -> Generator[torch.Tensor, None, None]:
    is_language_model = models[0].is_language_model
    device = models[0].device
    for batch in dataloader:
        if is_language_model:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
            yield [model(**batch) for model in models]
        else:
            x = x.to(device)
            yield [model(x) for model in models]


def register_hooks(
    model: nn.Module,
    activations: Dict[str, torch.Tensor],
    exclude_names: List[str] = ["relu", "shortcut"],
) -> List:
    def _get_activation(activations: Dict[str, torch.Tensor], name: str):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    forward_hooks = []
    for name, module in model.model.named_modules():
        if any([e in name for e in exclude_names]):
            continue
        hook = module.register_forward_hook(_get_activation(activations, name))
        forward_hooks.append(hook)

    return forward_hooks

def remove_hooks(hooks: List[torch.utils.hooks.RemovableHandle]) -> None:
    logger.debug("Removing hooks")
    for hook in hooks:
        hook.remove()

def register_activation_hooks(
    model: nn.Module,
    ps: PermSpec,
    activations: Dict[str, torch.Tensor],
    exclude_names: List[str] = [],
) -> List[torch.utils.hooks.RemovableHandle]:
    def _get_activation(activations: Dict[str, torch.Tensor], name: str):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    forward_hooks = []
    modules = []
    if ps.acts_to_perms:
        modules = list(ps.acts_to_perms.keys())
    for name, module in model.model.named_modules():
        if any([e in name for e in exclude_names]) or name not in modules:
            continue
        hook = module.register_forward_hook(_get_activation(activations, name))
        forward_hooks.append(hook)

    return forward_hooks

@torch.no_grad()
def activation_matching(
    ps: PermSpec,
    model_a: nn.Module,
    model_b: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    verbose: bool = False,
    kernel_func: callable = outer_product,
    num_samples: int = -1,
    perm_method: Literal["lsa", "sinkhorn", "sinkhorn_lsa"] = "lsa",
    sinkhorn_regularizer: Union[None, float, Dict[str, float]] = None
) -> PermType:
    """ Matches model_b's parameters to that of model_b based on their activations """
    log_fn = logger.info if verbose else logger.debug
    log_fn("Starting activation alignment")
    cost_matrix, act_shapes = get_activations_cost_matrix(ps, model_a, model_b, dataloader, kernel_func, num_samples)

    if sinkhorn_regularizer is None:
        sinkhorn_regularizer = dict((p, 1./math.sqrt(act_shapes[p])) for p in act_shapes)
    perms = solve_for_perms(cost_matrix, perm_method, verbose,  sinkhorn_regularizer)
    return perms

def get_activations_cost_matrix(ps:PermSpec, model_a: nn.Module, model_b: nn.Module, dataloader: torch.utils.data.DataLoader, 
    kernel_func: callable = outer_product,
    num_samples: int = -1):
    perm_sizes = get_permutation_sizes( model_a.model.state_dict(), ps)
    cost_matrix = {p: np.zeros((n, n)) for p, n in perm_sizes.items()}

    activations_a = dict()
    activations_b = dict()

    forward_hooks = register_activation_hooks(model_a, ps, activations_a, exclude_names=[]) + register_activation_hooks(model_b, ps, activations_b, exclude_names=[])

    cnt = 0
    act_shapes = dict()
    for i, (act_a, act_b) in enumerate(get_activations_yield([model_a, model_b], dataloader)):
        if num_samples != -1 and cnt > num_samples:
            break
        for batch_ind in range(len(act_a)):
            if num_samples != -1 and cnt > num_samples:
                break
            for module, p in ps.acts_to_perms.items():
                sample_a = activations_a[module][batch_ind].flatten(1).detach().cpu().numpy()
                sample_b = activations_b[module][batch_ind].flatten(1).detach().cpu().numpy()
                c = kernel_func(sample_a, sample_b,)
                cost_matrix[p] += c
                act_shapes[p] = sample_a.shape[1]

            cnt += 1
        activations_a.clear()
        activations_b.clear()
    for p in cost_matrix:
        cost_matrix[p] /= cnt
    remove_hooks(forward_hooks)
    return cost_matrix,act_shapes

def activation_matching_old(
    ps: PermSpec,
    model_a,
    model_b,
    dataloader: torch.utils.data.DataLoader,
    verbose: bool = False,
    kernel_func: callable = outer_product,
    num_samples: int = 2,
    align_bias: bool = True,
    exclude_lst: List[str] = ["relu", "shortcut", "fc"],
) -> Dict[str, np.array]:
    log_fn = logger.info if verbose else logger.debug
    log_fn("Starting activation alignment")
    # TODO: maybe add here an init perm?
    params_a = model_a.model.state_dict()
    # Assuming params_a and ps are defined elsewhere
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]]
        for p, axes in ps.perms_to_names.items()
    }
    perms = {p: np.arange(n) for p, n in perm_sizes.items()}
    cost_matrix = {p: np.zeros((n, n)) for p, n in perm_sizes.items()}
    perm_names = list(perms.keys())
    activations_a = (
        dict()
    )  # get_activations(model_a, dataloader, num_samples=num_samples)
    activations_b = (
        dict()
    )  # get_activations(model_b, dataloader, num_samples=num_samples)

    forward_hooks = register_hooks(
        model_a, activations_a, exclude_lst
    ) + register_hooks(model_b, activations_b, exclude_lst)

    cnt = 0
    for i, (act_a, act_b) in enumerate(
        get_activations_yield([model_a, model_b], dataloader)
    ):
        act_keys = []
        import gc

        gc.collect()
        if num_samples != -1 and cnt > num_samples:
            break
        cnt += act_a.shape[0]
        for p in perm_names:
            n = perm_sizes[p]
            for wk, axis in ps.perms_to_names[p]:
                if axis != 0 or any([e in wk for e in exclude_lst]):
                    continue
                # elif align_bias and ".bias" in wk:
                #     act_key = wk[:-5]
                elif wk.endswith(".weight"):
                    act_key = wk[:-7]
                else:
                    continue
                if act_key in act_keys:
                    logger.debug(f"{act_key} already in ackt_keys")
                act_keys.append(act_key)
                # import code; code.interact(local=locals()|globals())
                act_a = activations_a[act_key]  # .mean(axis=0)
                act_b = activations_b[act_key]  # .mean(axis=0)
                norm_factor = 1.0 / (num_samples * act_a.shape[0])
                act_a = torch.moveaxis(act_a, 0, -1).reshape((n, -1))  # .cpu().numpy()
                # act_b = torch.index_select(act_b, 1, torch.tensor(perm[p], device=act_b.device))
                act_b = torch.moveaxis(act_b, 0, -1).reshape((n, -1))  # .cpu().numpy()
                cost_matrix[p] += norm_factor * kernel_func(act_a, act_b).cpu().numpy()
                activations_a[act_key] = None
                activations_b[act_key] = None
    log_fn("Removing hooks")
    for hook in forward_hooks:
        hook.remove()
    # now solve the linear sum assignment
    for p in perm_names:
        ri, ci = linear_sum_assignment(cost_matrix[p], maximize=True)
        assert (ri == np.arange(len(ri))).all()

        # x[perms[p]] -> x[perms[p]][ci]
        perms[p] = ci  # perms[p][ci]

    return perms


def activation_matching_sinkhorn(
    ps: PermSpec,
    model_a,
    model_b,
    dataloader: torch.utils.data.DataLoader,
    verbose: bool = False,
    kernel_func: callable = outer_product,
    num_samples: int = 2,
    align_bias: bool = True,
    exclude_lst: List[str] = ["relu", "shortcut", "fc"],
    reg: float = 0.1,
    post_act: bool = True,
    return_perms: bool = True,
) -> Dict[str, np.array]:
    log_fn = logger.info if verbose else logger.debug
    log_fn("Starting activation alignment")
    # TODO: maybe add here an init perm?
    params_a = model_a.model.state_dict()
    # Assuming params_a and ps are defined elsewhere
    perm_sizes = get_permutation_sizes(model_a.model.state_dict(), ps)
    perms = {p: np.arange(n) for p, n in perm_sizes.items()}
    cost_matrix = {p: np.zeros((n, n)) for p, n in perm_sizes.items()}
    perm_names = list(perms.keys())
    activations_a = (
        dict()
    )  # get_activations(model_a, dataloader, num_samples=num_samples)
    activations_b = (
        dict()
    )  # get_activations(model_b, dataloader, num_samples=num_samples)

    forward_hooks = register_hooks(
        model_a, activations_a, exclude_lst
    ) + register_hooks(model_b, activations_b, exclude_lst)

    cnt = 0
    cost_cnts = dict()
    for i, (act_a, act_b) in enumerate(
        get_activations_yield([model_a, model_b], dataloader)
    ):
        act_keys = []
        import gc

        gc.collect()
        if num_samples != -1 and cnt > num_samples:
            break
        cnt += act_a.shape[0]
        for p in perm_names:
            n = perm_sizes[p]
            for wk, axis in ps.perms_to_names[p]:
                if axis != 0 or any([e in wk for e in exclude_lst]):
                    continue
                # elif align_bias and ".bias" in wk:
                #     act_key = wk[:-5]
                if "norm" not in wk:
                    continue
                elif wk.endswith(".weight"):
                    act_key = wk[:-7]
                else:
                    continue
                if act_key in act_keys:
                    logger.debug(f"{act_key} already in ackt_keys")
                act_keys.append(act_key)
                # import code; code.interact(local=locals()|globals())
                act_a = activations_a[act_key]  # .mean(axis=0)
                act_b = activations_b[act_key]  # .mean(axis=0)
                if post_act:
                    act_a = F.relu(act_a)
                    act_b = F.relu(act_b)
                norm_factor = 1.0 / (num_samples * act_a.shape[0])
                act_a = (
                    torch.moveaxis(act_a, 0, -1).reshape((n, -1)) / norm_factor
                )  # .cpu().numpy()
                # act_b = torch.index_select(act_b, 1, torch.tensor(perm[p], device=act_b.device))
                act_b = (
                    torch.moveaxis(act_b, 0, -1).reshape((n, -1)) / norm_factor
                )  # .cpu().numpy()
                cost_matrix[p] += kernel_func(act_a, act_b).cpu().numpy()
                if p not in cost_cnts:
                    cost_cnts[p] = 0
                cost_cnts[p] += 1
                activations_a[act_key] = None
                activations_b[act_key] = None
    log_fn("Removing hooks")
    for hook in forward_hooks:
        hook.remove()
    # now solve the linear sum assignment
    cis = dict()
    for p in perm_names:
        m, n = cost_matrix[p].shape
        a = np.ones(m) / m
        b = np.ones(n) / n
        P = ot.sinkhorn(a, b, cost_matrix[p] / cost_cnts[p], reg=reg, verbose=verbose)
        # ci = np.argmax(P, 1)
        ri, ci = linear_sum_assignment(-P)
        # ri, ci =  linear_sum_assignment(cost_matrix[p], maximize=True)
        # assert (ri == np.arange(len(ri))).all()
        cis[p] = ci
        # x[perms[p]] -> x[perms[p]][ci]
        perms[p] = P  # ci #perms[p][ci]
    if return_perms:
        return ci
    return perms