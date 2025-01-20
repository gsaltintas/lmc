"""
Defines functionality to extract statistics from permutations
"""

from typing import Dict

import numpy as np


def fixed_points(perm: np.array) -> int:
    """counts the number of fixed points in a given permutation"""
    return (perm == np.arange(perm.shape[0])).sum()


def get_fixed_points_ratio(perms: Dict[str, np.array]) -> float:
    total = 0.0
    fixed_points_cnt = 0
    for _, p in perms.items():
        fixed_points_cnt += fixed_points(p)
        total += len(p)
    return fixed_points_cnt / total


def is_identity_perm(perm: np.array) -> bool:
    """checks if the permutation is identiy permutation, i.e. (0, 1, ..., n-1)"""
    return fixed_points(perm) == len(perm)


def all_perms_are_identity(perms: Dict[str, np.array]) -> bool:
    """check if all permutations are identity permutations"""
    for _, p in perms.items():
        if not is_identity_perm(p):
            return False
    return True


def get_fixed_points_count(perms: Dict[str, np.array]) -> int:
    fixed_points_cnt = sum([fixed_points(p) for _, p in perms.items()])
    return fixed_points_cnt


def get_fixed_point_changes(
    permsa: Dict[str, np.array], permsb: Dict[str, np.array]
) -> int:
    """returns the number of changes between the two permutations"""
    changes = 0
    total = 0
    for (n, perma), (_, permb) in zip(permsa.items(), permsb.items()):
        changes += ((perma - permb) != 0).sum()
        total += len(perma)
    return changes / total
