
import unittest

import numpy as np
import torch
import torch.nn as nn

from lmc.permutations import (PermSpec, PermType, get_permutation_sizes,
                              get_random_permutation_with_fixed_points,
                              permute_param)


class TestPermuteParam(unittest.TestCase):
    def test_perm_size(self):
        perm_spec = PermSpec({"weight": [None, "perm1"]})
        param = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        perm_sizes = get_permutation_sizes({"weight": param}, perm_spec)
        self.assertDictEqual(perm_sizes, {"perm1": 2})

    def test_permute_param(self):
        # Define the permutation specifications
        perm_spec = PermSpec({"weight": [None, "perm1"]})

        # Define the permutations (for example, swap indices 0 and 1 in the second axis)
        perms = {"perm1": np.array([1, 0], dtype=np.int64)}

        # Create a parameter tensor (2x2 matrix)
        param = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        # Call the permute_param function
        permuted_param = permute_param(perm_spec, perms, "weight", param)

        # Expected output: the second axis should be permuted
        expected_output = torch.tensor([[2.0, 1.0], [4.0, 3.0]])

        # Check if the permuted_param matches the expected output
        self.assertTrue(
            torch.equal(permuted_param, expected_output),
            "The permuted parameter does not match the expected output.",
        )

    def test_permute_3x4_matrix(self):
        # Define permutation specifications
        # here permutations are picked indices not the index we put them to
        perm_spec = PermSpec({"matrix": ["row_perm", "col_perm"]}, model_name="matrix")
        print()
        print(perm_spec)

        # Define permutations for rows and columns
        perms = {
            "row_perm": np.array([2, 0, 1], dtype=np.int64),  # Permute rows
            "col_perm": np.array([1, 0, 3, 2], dtype=np.int64),  # Permute columns
        }

        # Create a 3x4 parameter tensor
        param = nn.Parameter(
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
            )
        )
        # [[2, 1, 4, 3]]
        # index_select: [[2, 1, 4, 3]] -> goes to row-1
        ##  [[10, 9, 12, 11]
        ##  [2,  1,  4,  3]
        ##  [6,  5,  8,  7]]

        # Call the permute_param function
        permuted_param = permute_param(perm_spec, perms, "matrix", param)

        # Expected output after row and column permutations
        expected_output = torch.tensor(
            [[10.0, 9.0, 12.0, 11.0], [2.0, 1.0, 4.0, 3.0], [6.0, 5.0, 8.0, 7.0]]
        )

        # Check if the permuted_param matches the expected output
        self.assertTrue(
            torch.allclose(permuted_param, expected_output),
            "The permuted parameter does not match the expected output.",
        )

    def test_except_axis(self):
        # Define the permutation specifications with multiple axes
        perm_spec = PermSpec({"weight": ["perm1", "perm2"]})

        # Define permutations for both axes
        perms = {
            "perm1": np.array([1, 0], dtype=np.int64),
            "perm2": np.array([1, 0], dtype=np.int64),
        }

        # Create a 2x2 parameter tensor
        param = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        # Call permute_param, excepting the first axis (so only the second axis should be permuted)
        permuted_param = permute_param(perm_spec, perms, "weight", param, except_axis=0)

        # Expected output: only the second axis should be permuted
        expected_output = torch.tensor([[2.0, 1.0], [4.0, 3.0]])

        # Check if the permuted_param matches the expected output
        self.assertTrue(
            torch.equal(permuted_param, expected_output),
            "The permuted parameter does not match the expected output when except_axis is used.",
        )

    def test_mismatched_dimensions(self):
        # Define the permutation specifications
        perm_spec = PermSpec({"weight": ["perm1"]})

        # Define a permutation that exceeds the parameter dimensions
        perms = {
            "perm1": np.array(
                [0, 1, 2], dtype=np.int64
            )  # 3 elements, but param only has 2
        }

        # Create a 2x2 parameter tensor
        param = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        # Expect ValueError due to mismatch in dimensions
        with self.assertRaises(ValueError):
            permute_param(perm_spec, perms, "weight", param)

    def test_get_permutation_with_fixed_points(self):
        n = 1000; fixed_points_fraction = 0.2
        perm = get_random_permutation_with_fixed_points(n, fixed_points_fraction)
        fixeds = np.round(n*fixed_points_fraction)
        self.assertGreaterEqual((perm == np.arange(n)).sum(), fixeds)

if __name__ == "__main__":
    unittest.main()
