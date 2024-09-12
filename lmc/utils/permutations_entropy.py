import numpy as np
from scipy.optimize import linear_sum_assignment

"""
numerical instability issues
the scale of the cost matrix needs to be controlled and not too big
1. we need a clever but fixed choice of \lambda that reflects weight norm
2. we need \lambda to depend on n as that indirectly indicates weight norm

proposed solution: at init, the expected norm is the variance, so rescale by this variance 1 / (m * Std(x))
Let \lambda = E[||X||]E[||Y||]
then -\lambda * M = -\lambda * X Y^T = -(X / E[||X||]) (Y / E[||Y||])^T
e.g. He initialization has std \sqrt(2/n_{in})
    suppose we are aligning vectors of size m
    then divide weights by \sqrt(2m/n_{in}), or equivalently, divide cost matrix by 2m/n_{in}, or equivalently, set \lambda=n_{in}/(2m)
"""

def l2_cost(X, Y):
    return -1 * (X @ Y.T)  # minimizing distance is equivalent to maximizing dot product

def logsum(x, axis=0):
    # computes \log sum_{i=1}^n exp(x_i)
    m = np.max(x, axis=axis)  # shift by max
    return m + np.log(np.sum(np.exp(x - m), axis=axis))

def sinkhorn_fp_logspace(P, iters=30):
    for _ in range(iters):
        P = (P - logsum(P)).T  # rows
        P = (P - logsum(P)).T  # columns
    return np.exp(P)

def sinkhorn_ot(X, Y, l=1, cost_fn=l2_cost):
    M = cost_fn(X, Y)
    # do this in log space to reduce numerical precision issues
    P = sinkhorn_fp_logspace(-l * M)
    return P.T  # turn into Y->X matrix so that trace(PM) gives cost

def linear_programming_ot(X, Y, cost_fn=l2_cost):
    M = cost_fn(X, Y)
    _, c = linear_sum_assignment(M)
    return np.eye(len(c))[:, c]  # turn into matrix

# compute empirical P_\sigma
def empirical_P(X, Y, sigma=1, n_samples=1000, add_noise_to_both=False):
    n = len(X)
    P_sigma = np.zeros((n, n))
    for _ in range(n_samples):
        Z = np.random.randn(*X.shape) * sigma
        Y_noise = Y + np.random.randn(*Y.shape) * sigma if add_noise_to_both else Y
        P = linear_programming_ot(X + Z, Y_noise)
        P_sigma += P
    return P_sigma / n_samples

def entropy(P):
    P = P / np.sum(P)  # normalize to sum to 1
    h = P * np.log(P)
    h[np.isnan(h)] = 0
    return -1 * np.sum(h)