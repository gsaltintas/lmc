import re
from typing import List


def match_pattern(name:str, ref_patterns: List[re.Pattern]):
    """ given a string, checks if it matches any patterns in the ref_patterns """
    compiled_patterns = [re.compile(pattern) for pattern in ref_patterns]
    correct_name = any(
        pattern.fullmatch(name) for pattern in compiled_patterns
    )
    return correct_name