import re
from typing import List

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