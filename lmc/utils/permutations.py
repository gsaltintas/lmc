"""_summary_

Defines all functionality pertaining to permutations and alignments

"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from numpy import typing as npt
from rich.console import Console
from rich.table import Table
from torch import nn

PermType = Dict[str, npt.ArrayLike]

@dataclass
class PermSpec:
    """ex: 
    names_to_perms: {"conv": ["P_0", None]}
    perms_to_names: {"P_0": [("conv", 0)]}
    """
    names_to_perms: Dict[str, List[str]]
    perms_to_names: Dict[str, List[str]] = None
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

    def __str__(self):
        console = Console()

        # Create a rich Table
        table = Table(title=self.model_name)

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
        with console.capture() as capture:
            console.print(table)
        return capture.get()


def permute_param(perm_spec: PermSpec, perms: PermType, param_name: str, param: nn.Parameter, except_axis: int = None):
    """Permute a parameter according to the permutation spec except for the except axis, useful for when solving the soblap."""
    for axis, perm_name in enumerate(perm_spec.names_to_perms[param_name]):
        if axis == except_axis:
            continue
        if perm_name is not None:
            perm = perms[perm_name]
            ## TODO: handle mismatch in dimensions
            if len(param.shape) <= axis:
                raise ValueError(f"Parameter ({param_name}) has {len(param.shape)} axes while requested {axis}.")
            if len(perm) > param.shape[axis]:
                raise ValueError(
                    f"Perms ({perm_name} with {len(perm)}) have larger shape at axis {axis} for {param_name} of shape {param.shape}."
                )
            with torch.no_grad():
                param = torch.index_select(
                    param, dim=axis, index=torch.from_numpy(perm).to(param.device)
                )
    return param

def permute_model(model: 'BaseModel', perm_spec: PermSpec, perms: PermType, inplace: bool = False) -> Union['BaseModel', nn.Module]:
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

def get_permutation_sizes(model_dct: Dict[str, torch.Tensor], perm_spec: PermSpec) -> Dict[str, int]:
    """ 
    calculates the permutation sizes for a given state_dict
    e.g. {P_0: size of the parameter along axis 0, ...}"
    """
    pn = perm_spec.perms_to_names
    perm_sizes = {perm_name: model_dct[axes[0][0]].shape[axes[0][1]] for perm_name, axes in pn.items()}
    return perm_sizes

def get_random_permutation_with_fixed_points(n: int, fixed_points_fraction: float) -> npt.ArrayLike:
    # draws a random permutation with at least fixed_points_fraction fixed points
    #TODO may want to also be able to mask certain indices as never permuted
    n_permuted = int(np.round(n * (1 - fixed_points_fraction))) - 1
    # expected number of fixed points in random perm is 1, so subtract 1 from guaranteed fixed points
    assert n_permuted > 1
    p = np.random.permutation(n_permuted)
    idx = np.arange(n)
    # randomly distribute permuted elements among fixed points
    idx_to_permute = np.random.permutation(n)[:n_permuted]
    idx[idx_to_permute] = idx_to_permute[p]
    return idx


