import re
from typing import Any, Dict, List

import numpy as np


def pattern_matched(name:str, ref_patterns: List[re.Pattern]):
    """ given a string, checks if it matches any patterns in the ref_patterns """
    compiled_patterns = [re.compile(pattern) for pattern in ref_patterns]
    return any([pattern.fullmatch(name) for pattern in compiled_patterns])

def match_pattern(name:str, ref_patterns: List[re.Pattern]):
    """ given a string, find the first pattern it matches """
    compiled_patterns = [re.compile(pattern) for pattern in ref_patterns]
    matches = np.array([pattern.fullmatch(name) for pattern in compiled_patterns])
    matched_ind = matches.astype(bool)
    return matches[matched_ind][0]



def flatten_dict(d: Dict[str, Any], preserve_prefix: bool = False) -> Dict[str, Any]:
    """Flatten a nested dictionary. If preserve prefix, nested keys will be prepended to the keys
    d: {"opt": {"lr": 0.1, "optimizer": "sgd"}, "training_steps": "2ep"}
    will return 
        - preserve_prefix=False:
            {"lr": 0.1, "optimizer": "sgd", "training_steps": "2ep"}
        - preserve_prefix=True:
            {"opt.lr": 0.1, "opt.optimizer": "sgd", "training_steps": "2ep"}
    """
    items = []
    def format_key(key, prefix):
        if preserve_prefix:
            return f"{prefix}.{key}"
        return key
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend([(format_key(k2, k), v2) for k2, v2 in flatten_dict(v, preserve_prefix).items()])
        else:
            items.append((k, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], parent_key="", sep=".") -> Dict[str, Any]:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(unflatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = str(
                v
            )  # Convert list to string to avoid issues in DataFrame
        else:
            items[new_key] = v
    return items
