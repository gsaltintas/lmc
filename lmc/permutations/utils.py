"""_summary_

Defines all functionality pertaining to permutations and alignments

"""

import logging
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy import typing as npt
from rich.console import Console
from rich.table import Table
from torch import nn

logger = logging.getLogger("Permutations")

@dataclass
class OnlineMean:
    sum: np.ndarray
    count: int

    @staticmethod
    def init(num_features: int):
        return OnlineMean(sum=np.zeros(num_features), count=0)

    def update(self, batch: np.ndarray):
        self.sum += np.sum(batch, axis=0)
        self.count += batch.shape[0]
        # return OnlineMean(self.sum + np.sum(batch, axis=0), self.count + batch.shape[0])

    def mean(self):
        return self.sum / self.count


@dataclass
class OnlineCovariance:
    a_mean: np.ndarray  # (d, )
    b_mean: np.ndarray  # (d, )
    cov: np.ndarray  # (d, d)
    var_a: np.ndarray  # (d, )
    var_b: np.ndarray  # (d, )
    count: int

    @staticmethod
    def init(a_mean: np.ndarray, b_mean: np.ndarray):
        assert a_mean.shape == b_mean.shape
        assert len(a_mean.shape) == 1
        d = a_mean.shape[0]
        return OnlineCovariance(
            a_mean,
            b_mean,
            cov=np.zeros((d, d)),
            var_a=np.zeros((d,)),
            var_b=np.zeros((d,)),
            count=0,
        )

    def update(self, a_batch, b_batch):
        assert a_batch.shape == b_batch.shape
        batch_size, _ = a_batch.shape
        a_res = a_batch - self.a_mean
        b_res = b_batch - self.b_mean
        self.cov += a_res.T @ b_res
        self.var_a += np.sum(a_res**2, axis=0)
        self.var_b += np.sum(b_res**2, axis=0)
        self.count += batch_size

    def covariance(self):
        return self.cov / (self.count - 1)

    def a_variance(self):
        return self.var_a / (self.count - 1)

    def b_variance(self):
        return self.var_b / (self.count - 1)

    def a_stddev(self):
        return np.sqrt(self.a_variance())

    def b_stddev(self):
        return np.sqrt(self.b_variance())

    def E_ab(self):
        return self.covariance() + np.outer(self.a_mean, self.b_mean)

    def pearson_correlation(self):
        # Note that the 1/(n-1) normalization terms cancel out nicely here.
        # TODO: clip?
        eps = 0
        # Dead units will have zero variance, which produces NaNs. Convert those to
        # zeros with nan_to_num.
        return np.nan_to_num(
            self.cov
            / (np.sqrt(self.var_a[:, np.newaxis]) + eps)
            / (np.sqrt(self.var_b) + eps)
        )



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

    @property
    def group_to_axes(self):
        return self.perms_to_names
    @property
    def axes_to_group(self):
        return self.names_to_perms

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

    def get_sizes(self, model_dct: Dict[str, torch.Tensor]):
        pn = self.perms_to_names
        perm_sizes = {
            perm_name: model_dct[axes[0][0]].shape[axes[0][1]]
            for perm_name, axes in pn.items()
        }
        return perm_sizes
    
    def get_identity_permutation(self, model_dct: Dict[str, torch.Tensor]):
        sizes = self.get_sizes(model_dct)
        return {p: np.arange(s) for p, s in sizes.items()}

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
            perm = torch.from_numpy(perms[perm_name]).to(param.device)
            ## TODO: handle mismatch in dimensions
            if len(param.shape) <= axis:
                raise ValueError(
                    f"Parameter ({param_name}) has {len(param.shape)} axes while requested {axis}."
                )
            permute_size = param.shape[axis]
            if len(perm) > permute_size:
                raise ValueError(
                    f"Perms ({perm_name} with {len(perm)}) have larger shape at axis {axis} for {param_name} of shape {param.shape}."
                )
            if perm.ndim == 1:
                with torch.no_grad():
                    param = torch.index_select(
                        param, dim=axis, index=perm
                    )
            else:
                # 2d perm matrix or transport map
                perm = perm.to(param.dtype)
                with torch.no_grad():
                    paramcp = param.clone()
                    paramcp = torch.moveaxis(param, axis, 0)
                    other_size = paramcp.shape[1:]
                    paramcp = paramcp.reshape(permute_size, -1)
                    paramcp = torch.matmul(perm.T, paramcp)
                    paramcp = paramcp.view(permute_size, *other_size)
                    paramcp = torch.moveaxis(paramcp, 0, axis)
                    param = paramcp

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


def get_non_permuted_sizes(
    model_dct: Dict[str, torch.Tensor], perm_spec: PermSpec
) -> Dict[str, int]:
    non_permuted_sizes = {}
    """for each permutation in perm_spec, sum up the non-permuted axes of the related parameters """
    for perm_name, params in perm_spec.perms_to_names.items():
        m = 0
        for param_name, permuted_dim in params:
            shape = model_dct[param_name].shape
            # find size over all dimensions except the permuted one
            m += np.product(shape) // shape[permuted_dim]
        non_permuted_sizes[perm_name] = m
    return non_permuted_sizes

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


################################################################################
############################### kernel functions ###############################
################################################################################

def outer_product(a, b) -> np.array:
    return a @ b.T


def cosine_similarity2d(A, B):
    # Step 1: Compute the dot product between all pairs of rows
    dot_product = A @ B.T
    
    # Step 2: Compute the L2 norms of each row in A and B
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)  # Shape (m, 1)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)  # Shape (n, 1)
    
    # Step 3: Normalize the dot products by the norms of A and B
    similarity = dot_product / (norm_A * norm_B.T + 1e-8)
    
    return similarity

def get_kernel_function(f: Union[str, callable]) -> callable:
    if isinstance(f, str):
        if f.lower() in ["linear", "outer", "outer_product"]:
            return outer_product
        if "cos" in f.lower():
            return cosine_similarity2d
    return f
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



def fixed_points(perm: np.array) -> int:
    """ counts the number of fixed points in a given permutation """
    return (perm == np.arange(perm.shape[0])).sum()

def get_fixed_points_ratio(perms: PermType) -> float:
    total = 0.
    fixed_points_cnt = 0
    for _, p in perms.items():
        fixed_points_cnt += fixed_points(p)
        total += len(p)
    return fixed_points_cnt / total

def is_identity_perm(perm:  np.array) -> bool:
    """ checks if the permutation is identiy permutation, i.e. (0, 1, ..., n-1) """
    return fixed_points(perm) == len(perm)

def all_perms_are_identity(perms:  PermType) -> bool:
    """ check if all permutations are identity permutations """
    for _, p in perms.items():
        if not is_identity_perm(p):
            return False
    return True

def get_fixed_points_count(perms: PermType) -> int:
    fixed_points_cnt = sum([fixed_points(p) for _, p in perms.items()])
    return fixed_points_cnt