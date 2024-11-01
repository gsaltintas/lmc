import logging
import math
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import ot
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .alignment import solve_for_a_perm
from .utils import (PermSpec, PermType, get_kernel_function,
                    get_permutation_sizes, outer_product, permute_param)

logger = logging.getLogger("w-matching")
def weight_matching_cost(
    ps: PermSpec,
    params_a: Union[Dict[str, torch.Tensor], nn.Module],
    params_b: Union[Dict[str, torch.Tensor], nn.Module],
    init_perm=None,
    kernel_func: Optional[Union[str, callable]] = outer_product,
    perm_order: Optional[List[int]] = None,
    align_bias: bool = False,
    perm: Optional[PermType] = None
) -> PermType:
    """Find a permutation of `params_b` to make them match `params_a`."""
    # inspect_func_args(kernel_func, 2)
    kernel_func = get_kernel_function(kernel_func)
    if isinstance(params_a, nn.Module):
        params_a = params_a.state_dict()
    if isinstance(params_b, nn.Module):
        params_b = params_b.state_dict()
    perm_sizes = get_permutation_sizes(params_a, ps)
    
    if perm is None:
        perm = (
        {p: np.arange(n) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )
    perm_names = np.array(list(perm.keys()))
    if perm_order is None:
        perm_order = perm_names[np.random.permutation(len(perm_names))]
    elif isinstance(perm_order[0], int):
        perm_order = perm_names[perm_order]

    costs = dict()
    for p in perm_order:
        n = perm_sizes[p]
        A = np.zeros((n, n))
        cnt = 0
        for wk, axis in ps.perms_to_names[p]:
            if not align_bias and "bias" in wk:
                continue
            w_a = params_a[wk]
            w_b = permute_param(ps, perm, wk, params_b[wk], except_axis=axis)
            w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).cpu().numpy()
            w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1)).cpu().numpy()
            A += kernel_func(w_a, w_b)
            cnt += 1
        costs[p] = A

    return costs


def weight_matching(
    ps: PermSpec,
    params_a,
    params_b,
    max_iter=100,
    init_perm=None,
    verbose: bool = False,
    kernel_func: callable = outer_product,
    perm_method: Literal["lsa", "sinkhorn", "sinkhorn_lsa"] = "lsa",
    sinkhorn_regularizer: Union[None, float, Dict[str, float]] = None,
) -> PermType:
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
            cnt = 0
            for wk, axis in ps.perms_to_names[p]:
                if "bias" in wk:
                    continue
                w_a = params_a[wk]
                w_b = permute_param(ps, perm, wk, params_b[wk], except_axis=axis)
                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).cpu().numpy()
                w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1)).cpu().numpy()
                rest_size = math.sqrt(w_a.shape[1])
                std = 1.0  # math.sqrt(2./rest_size)
                c = kernel_func(w_a / std, w_b / std) / 1.0  # rest_size
                A += c  # w_a @ w_b.T
                # print("cost", wk, c.mean(), c.std())
                # print((kernel_func(w_a / std, w_b / std) /std ).std())
                cnt += 1
            # A /= cnt
            # print(n, A.mean(), A.std(), np.diag(A).sum(), np.triu(A, 1).sum()+ np.tril(A, -1).sum())

            reg = None
            if isinstance(sinkhorn_regularizer, dict):
                reg = sinkhorn_regularizer.get(p, None)
            elif isinstance(sinkhorn_regularizer, float):
                reg = sinkhorn_regularizer
            ci = solve_for_a_perm(
                A, perm_method=perm_method, verbose=False, sinkhorn_regularizer=reg
            )

            oldL = np.vdot(A, np.eye(n)[perm[p]])
            if ci.ndim == 2:
                # TODO: not sure about this
                newL = np.dot(ci, A).sum()
            else:
                newL = np.vdot(A, np.eye(n)[ci, :])
            if verbose:
                print(
                    "Iter %3d/%4s: %.2f, %.2f prog: %.2f"
                    % (iteration, p, newL, oldL, newL - oldL)
                )
            progress = progress or newL > oldL + 1e-12

            perm[p] = np.array(ci)

        if not progress:
            break

    return perm
