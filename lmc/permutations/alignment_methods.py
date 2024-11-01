from typing import Dict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from lmc.permutations.permutations import PermSpec, logger, permute_param


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