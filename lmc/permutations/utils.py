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

from .attention_perms import AttentionLayerPermType

logger = logging.getLogger("Permutations")

PermType = Dict[str, Union[AttentionLayerPermType, npt.ArrayLike]]

@dataclass
class PermSpec:
    """
    Examples:
    names_to_perms: {
        "conv": ["P_0", None],
        "attention.query": [("P_head_0", "P_dhead_0"), None]
    }
    perms_to_names: {
        "P_0": [("conv", 0)],
        "P_head_0": [("attention.query", 0)],
        "P_dhead_0": [("attention.query", 0)]
    }
    """

    names_to_perms: Dict[str, List[Union[str, Tuple[str, str], None]]]
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
                        if isinstance(perm, tuple):
                            # Handle head permutations
                            perm_head, perm_dhead = perm
                            if perm_head not in perms_to_names:
                                perms_to_names[perm_head] = []
                            if perm_dhead not in perms_to_names:
                                perms_to_names[perm_dhead] = []
                            perms_to_names[perm_head].append((param_name, axis, "head"))  # Add type
                            perms_to_names[perm_dhead].append((param_name, axis, "d_head"))  # Add type
                        else:
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
    
    def _format_perm_for_display(self, perm):
        """Format permutation for display in table"""
        if perm is None:
            return ""
        elif isinstance(perm, tuple):
            return f"{perm[0]}, {perm[1]}"  # Removed parentheses for better rendering
        return str(perm)
    
    def set_head_info(self, num_heads: int, d_head: int):
        """Set head dimension information"""
        self.head_info = {"num_heads": num_heads, "d_head": d_head}

    @staticmethod
    def from_names_to_perms(names_to_perms: Dict[str, List[Union[str, Tuple[str, str], None]]]) -> "PermSpec":
        perms_to_names = OrderedDict()
        for param_name, perm_names in names_to_perms.items():
            for axis, perm in enumerate(perm_names):
                if perm is not None:
                    if isinstance(perm, tuple):
                        # Handle head permutations
                        perm_head, perm_dhead = perm
                        if perm_head not in perms_to_names:
                            perms_to_names[perm_head] = []
                        if perm_dhead not in perms_to_names:
                            perms_to_names[perm_dhead] = []
                        perms_to_names[perm_head].append((param_name, axis))
                        perms_to_names[perm_dhead].append((param_name, axis))
                    else:
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
        
        try:
            for n, p in self.names_to_perms.items():
                pin = None
                if len(p) > 1:
                    pin = p[1]
                pout = p[0]
                
                # Convert permutations to strings
                pin_str = self._format_perm_for_display(pin)
                pout_str = self._format_perm_for_display(pout)
                
                table.add_row(str(pin_str), str(n), str(pout_str))

            if self.acts_to_perms is not None:
                table2 = Table(title=f"{self.model_name} Activations to Permutations")
                table2.add_column("Module", style="magenta", no_wrap=True)
                table2.add_column("P_out", style="green")
                for n, pout in self.acts_to_perms.items():
                    table2.add_row(str(n), str(self._format_perm_for_display(pout)))

            with console.capture() as capture:
                console.print(table)
                if self.acts_to_perms is not None:
                    console.print(table2)
            return capture.get()
        except Exception as e:
            return f"Error rendering table: {str(e)}\nPermutation spec: {self.names_to_perms}"
    
    
def apply_head_permutation(param: torch.Tensor, perm_head: np.ndarray, perm_dhead: np.ndarray, 
                          num_heads: int, d_head: int, axis: int) -> torch.Tensor:
    """Apply permutations to head dimensions of parameter"""
    if axis == 0:
        # Reshape to [num_heads, d_head, ...]
        shape = param.shape
        param = param.view(num_heads, d_head, *shape[1:])
        # Permute heads
        param = torch.index_select(param, 0, torch.from_numpy(perm_head).to(param.device))
        # Permute within heads
        param = torch.index_select(param, 1, torch.from_numpy(perm_dhead).to(param.device))
        # Reshape back
        param = param.reshape(shape)
    else:
        # Reshape to [..., num_heads, d_head]
        shape = param.shape
        param = param.view(*shape[:-1], num_heads, d_head)
        # Permute heads
        param = torch.index_select(param, -2, torch.from_numpy(perm_head).to(param.device))
        # Permute within heads
        param = torch.index_select(param, -1, torch.from_numpy(perm_dhead).to(param.device))
        # Reshape back
        param = param.reshape(shape)
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
    num_heads: int = 12,
    d_head: int = 64,
) -> Union["BaseModel", nn.Module]:
    """Given a pytorch model, permutes it with respect to the given permutations
    Returns a new model if inplace is false
    """
    permuted_dct = OrderedDict()
    if inplace:
        permuted_dct = model_state_dct
    for name, param in model_state_dct.items():
        if name in perm_spec.names_to_perms:
            permuted_dct[name] = permute_param(perm_spec, perms, name, param, num_heads=num_heads, d_head=d_head)
        else:
            permuted_dct[name] = param
    return permuted_dct

def permute_param(
    perm_spec: PermSpec,
    perms: PermType,
    param_name: str,
    param: torch.Tensor,
    except_axis: int = None,
    num_heads: int = 12,
    d_head: int = 64,
):
    """Permute a parameter according to the permutation spec except for the except axis"""
    if param_name not in perm_spec.names_to_perms:
        return param

    permuted_param = param
    for axis, perm_name in enumerate(perm_spec.names_to_perms[param_name]):
        if axis == except_axis:
            continue
            
        if perm_name is not None:
            if isinstance(perm_name, tuple):
                # Handle head-level permutations
                perm_head_name, perm_dhead_name = perm_name
                if perm_head_name is not None and perm_dhead_name is not None:
                    shape = permuted_param.shape
                    # Reshape to separate head dimensions
                    if axis == 0:
                        # For output dimension
                        permuted_param = permuted_param.view(num_heads, d_head, *shape[1:])
                        # Apply head permutation
                        perm_head = torch.from_numpy(perms[perm_head_name]).to(permuted_param.device)
                        permuted_param = torch.index_select(permuted_param, 0, perm_head)
                        # Apply d_head permutation
                        perm_dhead = torch.from_numpy(perms[perm_dhead_name]).to(permuted_param.device)
                        permuted_param = torch.index_select(permuted_param, 1, perm_dhead)
                        # Reshape back
                        permuted_param = permuted_param.reshape(shape)
                    else:
                        # For input dimension
                        permuted_param = permuted_param.view(*shape[:-1], num_heads, d_head)
                        # Apply head permutation
                        perm_head = torch.from_numpy(perms[perm_head_name]).to(permuted_param.device)
                        permuted_param = torch.index_select(permuted_param, -2, perm_head)
                        # Apply d_head permutation
                        perm_dhead = torch.from_numpy(perms[perm_dhead_name]).to(permuted_param.device)
                        permuted_param = torch.index_select(permuted_param, -1, perm_dhead)
                        # Reshape back
                        permuted_param = permuted_param.reshape(shape)
            else:
                # Regular permutation
                perm = torch.from_numpy(perms[perm_name]).to(permuted_param.device)
                if len(permuted_param.shape) <= axis:
                    raise ValueError(
                        f"Parameter ({param_name}) has {len(permuted_param.shape)} axes while requested {axis}."
                    )
                permuted_param = torch.index_select(permuted_param, axis, perm)

    return permuted_param

def get_permutation_sizes(
    model_dct: Dict[str, torch.Tensor], 
    perm_spec: PermSpec,
    num_heads: int = 12,
    d_head: int = 64,
) -> Dict[str, int]:
    """
    calculates the permutation sizes for a given state_dict
    e.g. {P_0: size of the parameter along axis 0, ...}"
    """
    perm_sizes = {}
    perm_sizes = {}
    for perm_name, params in perm_spec.perms_to_names.items():
        for param_name, axis, *ptype in params:
            if ptype:  # This is a head permutation
                if ptype[0] == "head":
                    perm_sizes[perm_name] = num_heads #perm_spec.head_info["num_heads"]
                elif ptype[0] == "d_head":
                    perm_sizes[perm_name] = d_head #perm_spec.head_info["d_head"]
            else:  # Regular permutation
                perm_sizes[perm_name] = model_dct[param_name].shape[axis]
            break  # Only need one parameter to determine size
    return perm_sizes


def generate_random_permutations(perm_spec: PermSpec, model_dct: Dict[str, torch.Tensor] = None, 
                               fixed_points_fraction: float = 0.0,
                               num_heads: int = 12, d_head: int = 64) -> Dict[str, np.ndarray]:
    """Generate random permutations according to the permutation spec"""
    if model_dct is not None:
        perm_sizes = get_permutation_sizes(model_dct, perm_spec, num_heads, d_head)
    else:
        # If no model_dct provided, use default sizes for attention
        perm_sizes = {}
        for perm_name in perm_spec.perm_names:
            if isinstance(perm_name, tuple):
                perm_sizes[perm_name[0]] = num_heads
                perm_sizes[perm_name[1]] = d_head
            else:
                # Default size for other permutations
                perm_sizes[perm_name] = 768  # BERT hidden size

    perms = {}
    for perm_name, size in perm_sizes.items():
        if fixed_points_fraction > 0:
            perms[perm_name] = get_random_permutation_with_fixed_points(size, fixed_points_fraction)
        else:
            perms[perm_name] = np.random.permutation(size)
            
    return perms

def permute_attention(attn_layer, perm):
    d_model = attn_layer.d_model
    num_heads = attn_layer.num_heads
    head_dim = attn_layer.head_dim

    # Permute Q, K, V weights
    attn_layer.q.weight.data = attn_layer.q.weight.data.view(num_heads, head_dim, d_model).index_select(0, perm).view(d_model, d_model)
    attn_layer.k.weight.data = attn_layer.k.weight.data.view(num_heads, head_dim, d_model).index_select(0, perm).view(d_model, d_model)
    attn_layer.v.weight.data = attn_layer.v.weight.data.view(num_heads, head_dim, d_model).index_select(0, perm).view(d_model, d_model)

    # Permute out weights
    attn_layer.o.weight.data = attn_layer.o.weight.data.view(d_model, num_heads, head_dim).index_select(1, perm).view(d_model, d_model)


def get_permutation_sizes_old(
    model_dct: Dict[str, torch.Tensor], perm_spec: PermSpec
) -> Dict[str, int]:
    """
    calculates the permutation sizes for a given state_dict
    e.g. {P_0: size of the parameter along axis 0, ...}"
    """
    pn = perm_spec.perms_to_names
    perm_sizes = dict()
    for perm_name, axes in pn.items():
        # axes: [(param_name, axis), (in_head, name, ax, AttentionLayerPermType)])
        first_ax = axes[0]
        
        if first_ax[0] == "across_head":
            perm_sizes[perm_name] = first_ax[-1].n_head
        elif first_ax[0] == "in_head":
            perm_sizes[perm_name] = first_ax[-1].d_head
        else:
            perm_sizes[perm_name] = model_dct[first_ax[0]].shape[first_ax[1]]
    return perm_sizes

def get_non_permuted_sizes(
    model_dct: Dict[str, torch.Tensor], 
    perm_spec: PermSpec,
    num_heads: int = 12,
    d_head: int = 64
) -> Dict[str, int]:
    """
    For each permutation in perm_spec, sum up the non-permuted axes of the related parameters.
    Handles both regular and head-level (tuple) permutations.
    
    Args:
        model_dct: Dictionary of model parameters
        perm_spec: Permutation specification
        num_heads: Number of attention heads
        d_head: Size of each attention head
        
    Returns:
        Dictionary mapping permutation names to their non-permuted sizes
    """
    non_permuted_sizes = {}
    
    def compute_non_permuted_size(param_shape: Tuple[int, ...], permuted_dim: int, 
                                is_head_dim: bool = False) -> int:
        """Helper to compute non-permuted size for a single parameter"""
        if is_head_dim:
            # For head dimensions, we need to consider the combined head dimension
            head_dim_size = num_heads * d_head
            other_dims = list(param_shape)
            head_dim = other_dims[permuted_dim]
            other_dims[permuted_dim] = head_dim_size
            return np.product(other_dims) // head_dim_size
        else:
            # Regular case
            return np.product(param_shape) // param_shape[permuted_dim]

    for perm_name, params in perm_spec.perms_to_names.items():
        total_size = 0
        for param_name, permuted_dim in params:
            shape = model_dct[param_name].shape
            
            # Check if this parameter uses head permutations
            uses_head_perm = False
            for perms in perm_spec.names_to_perms[param_name]:
                if isinstance(perms, tuple) and any(p == perm_name for p in perms):
                    uses_head_perm = True
                    break
            
            # Compute size
            size = compute_non_permuted_size(shape, permuted_dim, uses_head_perm)
            total_size += size
            
        non_permuted_sizes[perm_name] = total_size

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