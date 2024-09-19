"""_summary_

Defines all functionality pertaining to permutations and alignments

"""

import inspect
import logging
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy import typing as npt
from rich.console import Console
from rich.table import Table
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger("Permutations")

PermType = Dict[str, npt.ArrayLike]


@dataclass
class PermSpec:
    """ex:
    names_to_perms: {"conv": ["P_0", None]}
    perms_to_names: {"P_0": [("conv", 0)]}
    """

    names_to_perms: Dict[str, List[str]]
    perms_to_names: Dict[str, List[Tuple[str, int]]] = None
    acts_to_perms: Dict[str, str] = None
    perms_to_acts: Dict[str, List[str]] = None
    model_name: Optional[str] = None

    def __post_init__(self):
        if self.perms_to_names is None:
            perms_to_names = OrderedDict()
            for param_name, perm_names in self.names_to_perms.items():
                for axis, perm in enumerate(perm_names):
                    if perm is not None:
                        if perm not in perms_to_names:
                            perms_to_names[perm] = []
                        perms_to_names[perm].append((param_name, axis))
            self.perms_to_names = perms_to_names
        if self.acts_to_perms is not None and self.perms_to_acts is None:
            perms_to_acts = OrderedDict()
            for layer_name, perm_name in self.acts_to_perms.items():
                if perm_name not in perms_to_acts:
                    perms_to_acts[perm_name] = []
                perms_to_acts[perm_name].append(layer_name)
            self.perms_to_acts = perms_to_acts

    @property
    def perm_names(self):
        return list(self.perms_to_names.keys())
    
    @staticmethod
    def from_names_to_perms(names_to_perms: Dict[str, List[str]]) -> "PermSpec":
        perms_to_names = OrderedDict()
        for param_name, perm_names in names_to_perms.items():
            for axis, perm in enumerate(perm_names):
                if perm is not None:
                    if perm not in perms_to_names:
                        perms_to_names[perm] = []
                    perms_to_names[perm].append((param_name, axis))
        return PermSpec(names_to_perms=names_to_perms, perms_to_names=perms_to_names)

    def __str__(self):
        console = Console()

        # Create a rich Table
        table = Table(title=f"{self.model_name} Permutations")

        # Add columns for the table
        table.add_column("P_in", style="cyan", no_wrap=True)
        table.add_column("Param", style="magenta")
        table.add_column("P_out", style="green")
        for n, p in self.names_to_perms.items():
            pin = None
            if len(p) > 1:
                pin = p[1]
            pout = p[0]
            table.add_row(pin, n, pout)

        if self.acts_to_perms is not None:
            table2 = Table(title=f"{self.model_name} Activations to Permutations")
            table2.add_column("Module", style="magenta", no_wrap=True)
            table2.add_column("P_out", style="green")
            for n, pout in self.acts_to_perms.items():
                table2.add_row(n, pout)
        with console.capture() as capture:
            console.print(table)
            if self.acts_to_perms is not None:
                console.print(table2)
        return capture.get()


def permute_param(
    perm_spec: PermSpec,
    perms: PermType,
    param_name: str,
    param: nn.Parameter,
    except_axis: int = None,
):
    """Permute a parameter according to the permutation spec except for the except axis, useful for when solving the soblap."""
    for axis, perm_name in enumerate(perm_spec.names_to_perms[param_name]):
        if axis == except_axis:
            continue
        if perm_name is not None:
            perm = perms[perm_name]
            ## TODO: handle mismatch in dimensions
            if len(param.shape) <= axis:
                raise ValueError(
                    f"Parameter ({param_name}) has {len(param.shape)} axes while requested {axis}."
                )
            if len(perm) > param.shape[axis]:
                raise ValueError(
                    f"Perms ({perm_name} with {len(perm)}) have larger shape at axis {axis} for {param_name} of shape {param.shape}."
                )
            with torch.no_grad():
                param = torch.index_select(
                    param, dim=axis, index=torch.from_numpy(perm).to(param.device)
                )
    return param


def permute_model(
    model: "BaseModel", perm_spec: PermSpec, perms: PermType, inplace: bool = False
) -> Union["BaseModel", nn.Module]:
    """given a pytorch model, permutes it with respect to the given permutations
    Returns a new model if inplace is false
    """
    permuted = model
    if not inplace:
        permuted = deepcopy(model)
    permuted_dct = OrderedDict()
    for name, param in model.state_dict().items():
        permuted_dct[name] = permute_param(perm_spec, perms, name, param)
    permuted.load_state_dict(permuted_dct)
    return permuted


def permute_state_dct(
    model_state_dct: Dict[str, torch.Tensor],
    perm_spec: PermSpec,
    perms: PermType,
    inplace: bool = False,
) -> Union["BaseModel", nn.Module]:
    """given a pytorch model, permutes it with respect to the given permutations
    Returns a new model if inplace is false
    """
    permuted_dct = OrderedDict()
    for name, param in model_state_dct.items():
        permuted_dct[name] = permute_param(perm_spec, perms, name, param)
    return permuted_dct


def get_permutation_sizes(
    model_dct: Dict[str, torch.Tensor], perm_spec: PermSpec
) -> Dict[str, int]:
    """
    calculates the permutation sizes for a given state_dict
    e.g. {P_0: size of the parameter along axis 0, ...}"
    """
    pn = perm_spec.perms_to_names
    perm_sizes = {
        perm_name: model_dct[axes[0][0]].shape[axes[0][1]]
        for perm_name, axes in pn.items()
    }
    return perm_sizes


def get_random_permutation_with_fixed_points(
    n: int, fixed_points_fraction: float
) -> npt.ArrayLike:
    # draws a random permutation with at least fixed_points_fraction fixed points
    # TODO may want to also be able to mask certain indices as never permuted
    n_permuted = int(np.round(n * (1 - fixed_points_fraction))) - 1
    # expected number of fixed points in random perm is 1, so subtract 1 from guaranteed fixed points
    assert n_permuted > 1
    p = np.random.permutation(n_permuted)
    idx = np.arange(n)
    # randomly distribute permuted elements among fixed points
    idx_to_permute = np.random.permutation(n)[:n_permuted]
    idx[idx_to_permute] = idx_to_permute[p]
    return idx


def outer_product(a, b) -> np.array:
    return a @ b.T


def weight_matching(
    ps: PermSpec,
    params_a,
    params_b,
    max_iter=100,
    init_perm=None,
    verbose: bool = False,
    kernel_func: callable = outer_product,
) -> Dict[str, np.array]:
    """Find a permutation of `params_b` to make them match `params_a`."""
    # inspect_func_args(kernel_func, 2)
    log_fn = logger.info if verbose else logger.debug
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]]
        for p, axes in ps.perms_to_names.items()
    }

    perm = (
        # {p: np.random.permutation(n) for p, n in perm_sizes.items()}
        {p: np.arange(n) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in np.random.permutation(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = np.zeros((n, n))
            for wk, axis in ps.perms_to_names[p]:
                w_a = params_a[wk]
                w_b = permute_param(ps, perm, wk, params_b[wk], except_axis=axis)
                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).cpu().numpy()
                w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1)).cpu().numpy()
                A += kernel_func(w_a, w_b)  # w_a @ w_b.T

            ri, ci = linear_sum_assignment(A, maximize=True)
            assert (ri == np.arange(len(ri))).all()

            oldL = np.vdot(A, np.eye(n)[perm[p]])
            newL = np.vdot(A, np.eye(n)[ci, :])
            if verbose:
                log_fn(
                    "Iter %3d/%4s: %.2f, %.2f prog: %.2f"
                    % (iteration, p, newL, oldL, newL - oldL)
                )
            progress = progress or newL > oldL + 1e-12

            perm[p] = np.array(ci)

        if not progress:
            break

    return perm


@torch.no_grad()
def get_activations_yield(
    models: nn.Module, dataloader: torch.utils.data.DataLoader
) -> Generator[torch.Tensor, None, None]:
    for x, _ in dataloader:
        if x.device != models[0].device:
            x = x.to(models[0].device)
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


def register_activation_hooks(
    model: nn.Module,
    ps: PermSpec,
    activations: Dict[str, torch.Tensor],
    exclude_names: List[str] = [],
) -> List:
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


def activation_matching(
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


import ot


# ot.sinkhorn(
#     a,
#     b,
#     M,
#     reg,
#     method='sinkhorn',
#     numItermax=1000,
#     stopThr=1e-09,
#     verbose=False,
#     log=False,
#     warn=True,
#     warmstart=None,
#     **kwargs,
# )
def reg_entropy(G):
    return np.sum(G * np.log(G + 1e-16)) - np.sum(G)


# import logging
# from lmc.utils.permutations import register_hooks, get_activations_yield, get_permutation_sizes
# from scipy.optimize import linear_sum_assignment
# from torch.nn import functional as F

# ps = el1.model.permutation_spec()
# model_a = el1.model
# model_b = el2.model
# dataloader: torch.utils.data.DataLoader = el1.train_eval_loader
# verbose: bool = False
# kernel_func: callable = outer_product
# num_samples: int = 2
# align_bias: bool = True
# exclude_lst = []#"relu","shortcut", "fc"]
# reg:float=0.1
# logger = logging.getLogger("notebook")
# log_fn = logger.info if verbose else logger.debug
# log_fn("Starting activation alignment")
# #TODO: maybe add here an init perm?
# params_a = model_a.model.state_dict()
# # Assuming params_a and ps are defined elsewhere
# perm_sizes = get_permutation_sizes(model_a.model.state_dict(), ps)
# perms = {p: np.arange(n) for p, n in perm_sizes.items()}
# cost_matrix = {p: np.zeros((n, n)) for p, n in perm_sizes.items()}
# perm_names = list(perms.keys())
# activations_a = dict() #get_activations(model_a, dataloader, num_samples=num_samples)
# activations_b = dict() #get_activations(model_b, dataloader, num_samples=num_samples)

# forward_hooks = register_hooks(model_a, activations_a, exclude_lst) + register_hooks(model_b, activations_b, exclude_lst)

# cnt = 0
# cost_cnts  = dict()
# for i, (act_a, act_b) in enumerate(get_activations_yield([model_a, model_b], dataloader)):
#     act_keys = []
#     import gc; gc.collect()
#     if num_samples != -1 and cnt > num_samples:
#         break
#     cnt += act_a.shape[0]
#     for p in perm_names:
#         n = perm_sizes[p]
#         for wk, axis in ps.perms_to_names[p]:
#             print(wk)
#             if axis != 0 or any([e in wk for e in exclude_lst]):
#                 continue
#             # elif align_bias and ".bias" in wk:
#             #     act_key = wk[:-5]
#             if "norm" not in wk:
#                 # pass
#                 continue
#             if wk.endswith(".weight"):
#                 act_key = wk[:-7]
#             else:
#                 continue
#             if act_key in act_keys:
#                 logger.debug(f"{act_key} already in ackt_keys")
#             act_keys.append(act_key)
#             act_a = F.relu(activations_a[act_key])#.mean(axis=0)
#             act_b = F.relu(activations_b[act_key])#.mean(axis=0)
#             norm_factor = 1. / math.sqrt(num_samples * act_a.shape[0])
#             act_a = torch.moveaxis(act_a, 0, -1).reshape((n, -1))  #.cpu().numpy()
#             m = act_a.shape[1]
#             # m = 1
#             act_a = act_a/m
#             # act_b = torch.index_select(act_b, 1, torch.tensor(perm[p], device=act_b.device))
#             act_b = torch.moveaxis(act_b, 0, -1).reshape((n, -1)) / m #.cpu().numpy()
#             cost_matrix[p] += kernel_func(act_a, act_b).cpu().numpy()
#             # cost_matrix[p] += F.cosine_similarity(act_a, act_b).cpu().numpy()
#             if p not in cost_cnts:
#                 cost_cnts[p] = 0
#             cost_cnts[p] += 1
#             activations_a[act_key] = None
#             activations_b[act_key] = None
# log_fn("Removing hooks")
# for hook in forward_hooks:
#     hook.remove()
# # now solve the linear sum assignment
# cis = dict()
# for p in perm_names:
#     m, n = cost_matrix[p].shape
#     a = np.ones(m) / m
#     b = np.ones(n) / n
#     P = ot.bregman.sinkhorn_log(a, b, -cost_matrix[p], reg=reg, verbose=verbose)
#     ri, ci =  linear_sum_assignment(cost_matrix[p], maximize=True)
#     ri, ci = linear_sum_assignment(-P)
#     assert (ri == np.arange(len(ri))).all()
#     cis[p] = ci
#     # x[perms[p]] -> x[perms[p]][ci]
#     perms[p] = P #ci #perms[p][ci]
