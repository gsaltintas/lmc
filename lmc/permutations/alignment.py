from typing import Dict, Literal, Union

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

from .utils import PermType


def solve_for_perms(cost_matrix: Dict[str, np.array],
    perm_method: Literal["lsa", "sinkhorn", "sinkhorn_lsa"] = "lsa",
    verbose: bool = False,
    sinkhorn_regularizer: Union[None, float, Dict[str, float]] = None,
    max_iters: int = 1000) -> PermType:
    perms = dict()
    for p in cost_matrix:
        c = cost_matrix[p]
        ## sinkhorn
        n = c.shape[0]
        probs = []#np.ones(n) / n
        # todo: design choice, should we still normalize it?
        if isinstance(sinkhorn_regularizer, dict):
            reg = sinkhorn_regularizer.get(p, 1.)
        elif isinstance(sinkhorn_regularizer, float):
            reg = sinkhorn_regularizer
        else:
            reg = 1.
        # print(reg)
        if perm_method == "lsa":
            ri, ci = linear_sum_assignment(c, maximize=True)
            assert np.allclose(ri, np.arange(len(ri)))
            perms[p] = ci
        elif perm_method == "sinkhorn":
            ## returns a matrix, the soft alignment matrix
            perms[p] = ot.bregman.sinkhorn_log(probs, probs, -c, reg=reg, verbose=verbose, numItermax=max_iters, stopThr=1e-4, )
        elif perm_method == "sinkhorn_lsa":
            perms[p] = ot.bregman.sinkhorn_log(probs, probs, -c, reg=reg, verbose=verbose, numItermax=max_iters)
            # minimize the 
            ri, ci = linear_sum_assignment(perms[p], maximize=True)
            assert np.allclose(ri, np.arange(len(ri)))
            perms[p] = ci
    return perms

def solve_for_a_perm(cost_matrix: np.array,
    perm_method: Literal["lsa", "sinkhorn", "sinkhorn_lsa"] = "lsa",
    verbose: bool = False,
    sinkhorn_regularizer: Union[None, float, Dict[str, float]] = None,
    max_iters: int = 1000) -> np.array:
    c = cost_matrix
    perm = None
    ## sinkhorn
    n = c.shape[0]
    probs = np.ones(n) / n
    # todo: design choice, should we still normalize it?
    reg = sinkhorn_regularizer if sinkhorn_regularizer else 1.
    if perm_method == "lsa":
        ri, ci = linear_sum_assignment(c, maximize=True)
        assert np.allclose(ri, np.arange(len(ri)))
        perm = ci
    elif perm_method == "sinkhorn":
        ## returns a matrix, the soft alignment matrix
        perm = ot.bregman.sinkhorn_log(probs, probs, -c, reg=reg, verbose=verbose, numItermax=max_iters)
    elif perm_method == "sinkhorn_lsa":
        perm = ot.bregman.sinkhorn_log(probs, probs, -c, reg=reg, verbose=verbose, numItermax=max_iters)
        # minimize the 
        ri, ci = linear_sum_assignment(perm, maximize=True)
        assert np.allclose(ri, np.arange(len(ri)))
        perm = ci
    return perm