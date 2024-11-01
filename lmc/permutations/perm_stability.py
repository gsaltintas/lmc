import numpy as np


def reg_entropy(G: np.array) -> float:
    return np.sum(G * np.log(G + 1e-16)) - np.sum(G)

def entropy(G: np.array) -> float:
    G = 1. * G / np.sum(G)
    return -np.sum(G * np.log(G + 1e-16))

def normalized_kl_stability(P: np.array) -> float:
    n = len(P)
    # if P.ndim == 2:
    #     print("2D, dont knwo")

    P = P / n  # normalize to sum to 1
    P = P / np.sum(P)  # normalize to sum to 1
    P[P == 0] = 1  # by convention when computing entropy, 0 log(0) = 1 log(1) = 0
    h = -P * np.log(P)
    return 1 - (2*np.log(n) - np.sum(h)) / np.log(n)

from lmc.permutations import PermType


def normalized_entropy(perms: PermType):
    entropies = dict()
    for p, perm in perms.items():
        n = len(perm)
        perm = perm / perm.sum()
        perm[perm == 0] = 1
        h = -perm * np.log(perm)
        entropies[p] = 1- ( 2*np.log(n) - np.sum(h)) / np.log(n)
        continue
        unif = np.full_like(perm, 1./(n*m))
        kl = np.sum(perm * (np.log(perm + 1e-16) - np.log(unif)))
        kls[p] = kl

    return entropies


def sinkhorn_kl(perms: PermType):
    kls = dict()
    for p, perm in perms.items():
        unif = np.full_like(perm, 1./perm.size)
        kl = np.sum(perm * (np.log(perm + 1e-16) - np.log(unif)))
        kls[p] = kl

    return kls