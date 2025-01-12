import unittest

import numpy as np
import torch
import torch.nn as nn

from lmc.permutations import (PermSpec, get_permutation_sizes,
                              get_random_permutation_with_fixed_points,
                              permute_param)
from lmc.permutations.perm_stability import (normalized_entropy,
                                             normalized_kl_stability,
                                             sinkhorn_kl)
from lmc.permutations.utils import (PermSpec, apply_head_permutation,
                                    get_permutation_sizes, permute_param,
                                    permute_state_dct)
from lmc.permutations.weight_alignment import (handle_head_param,
                                               weight_matching,
                                               weight_matching_cost)


class TestHeadPermutations(unittest.TestCase):
    def setUp(self):
        self.num_heads = 4
        self.d_head = 8
        self.hidden_size = self.num_heads * self.d_head
        
        # Create sample weights
        self.query_weight = torch.randn(self.hidden_size, self.hidden_size)
        self.key_weight = torch.randn(self.hidden_size, self.hidden_size)
        self.value_weight = torch.randn(self.hidden_size, self.hidden_size)
        self.output_weight = torch.randn(self.hidden_size, self.hidden_size)
        
        self.model_state = {
            "attention.query.weight": self.query_weight,
            "attention.key.weight": self.key_weight,
            "attention.value.weight": self.value_weight,
            "output.dense.weight": self.output_weight,
        }
        # Create PermSpec
        self.names_to_perms = {
            "attention.query.weight": [("P_head_0", "P_dhead_0"), "P_in"],
            "attention.key.weight": [("P_head_0", "P_dhead_0"), "P_in"],
            "attention.value.weight": [("P_v_0", "P_dhead_0"), "P_in"],
            "output.dense.weight": ["P_out", ("P_v_0", "P_dhead_0")],
        }
        self.perm_spec = PermSpec(names_to_perms=self.names_to_perms, num_heads=self.num_heads, d_head=self.d_head)

    def test_handle_head_param(self):
        """Test head dimension handling"""
        weight = self.query_weight
        
        # Test output dimension (axis=0)
        reshaped = handle_head_param(weight, 0, self.num_heads, self.d_head)
        self.assertEqual(reshaped.shape[:2], (self.num_heads, self.d_head))
        
        # Test input dimension (axis=1)
        reshaped = handle_head_param(weight, 1, self.num_heads, self.d_head)
        self.assertEqual(reshaped.shape[-2:], (self.num_heads, self.d_head))

    def test_permute_param(self):
        """Test parameter permutation with head dimensions"""
        # Create random permutations
        perms = {
            "P_head_0": np.random.permutation(self.num_heads),
            "P_dhead_0": np.random.permutation(self.d_head),
            "P_v_0": np.random.permutation(self.num_heads),
            "P_in": np.random.permutation(self.hidden_size),
        }
        
        # Test permutation
        permuted = permute_param(
            self.perm_spec, perms, 
            "attention.query.weight", 
            self.model_state["attention.query.weight"],
        )
        
        self.assertEqual(permuted.shape, self.query_weight.shape)

    def test_get_permutation_sizes(self):
        """Test getting correct permutation sizes for head dimensions"""
        sizes = get_permutation_sizes(
            self.model_state, self.perm_spec,
        )
        
        self.assertEqual(sizes["P_head_0"], self.num_heads)
        self.assertEqual(sizes["P_dhead_0"], self.d_head)

class TestWeightMatching(unittest.TestCase):
    def setUp(self):
        self.num_heads = 4
        self.d_head = 8
        self.hidden_size = self.num_heads * self.d_head
        
        # Create two sets of weights
        self.weights_a = {
            "attention.query.weight": torch.randn(self.hidden_size, self.hidden_size),
            "attention.key.weight": torch.randn(self.hidden_size, self.hidden_size),
            "attention.value.weight": torch.randn(self.hidden_size, self.hidden_size),
        }
        self.weights_b = {
            "attention.query.weight": torch.randn(self.hidden_size, self.hidden_size),
            "attention.key.weight": torch.randn(self.hidden_size, self.hidden_size),
            "attention.value.weight": torch.randn(self.hidden_size, self.hidden_size),
        }
        
        # Create PermSpec
        self.names_to_perms = {
            "attention.query.weight": [("P_head_0", "P_dhead_0"), "P_in"],
            "attention.key.weight": [("P_head_0", "P_dhead_0"), "P_in"],
            "attention.value.weight": [("P_v_0", "P_dhead_0"), "P_in"],
        }
        self.perm_spec = PermSpec(names_to_perms=self.names_to_perms, num_heads=self.num_heads, d_head=self.d_head)

    def test_weight_matching_cost(self):
        """Test cost computation for weight matching"""
        costs = weight_matching_cost(
            self.perm_spec,
            self.weights_a,
            self.weights_b,
            align_bias=False
        )
        
        self.assertIn("P_head_0", costs)
        self.assertIn("P_dhead_0", costs)
        self.assertIn("P_v_0", costs)
        
        # Check cost matrix shapes
        self.assertEqual(costs["P_head_0"].shape, (self.num_heads, self.num_heads))
        self.assertEqual(costs["P_dhead_0"].shape, (self.d_head, self.d_head))

    def test_weight_matching(self):
        """Test full weight matching"""
        perms = weight_matching(
            self.perm_spec,
            self.weights_a,
            self.weights_b,
            max_iter=10,
            verbose=False
        )
        
        # Check permutation sizes
        self.assertEqual(len(perms["P_head_0"]), self.num_heads)
        self.assertEqual(len(perms["P_dhead_0"]), self.d_head)
        
        # Check permutations are valid
        for p_name, perm in perms.items():
            self.assertTrue(np.all(np.sort(perm) == np.arange(len(perm))))

## TODO: perm stability metrics
# class TestPermStability(unittest.TestCase):
#     def setUp(self):
#         self.num_heads = 4
#         self.perms = {
#             "P_head_0": np.random.permutation(self.num_heads),
#             "P_dhead_0": np.random.permutation(8),
#             "P_v_0": np.random.permutation(self.num_heads),
#         }

#     def test_normalized_entropy(self):
#         """Test entropy calculation for permutations"""
#         entropies = normalized_entropy(self.perms)
        
#         for p_name in self.perms:
#             self.assertIn(p_name, entropies)
#             self.assertGreaterEqual(entropies[p_name], 0)
#             self.assertLessEqual(entropies[p_name], 1)

#     def test_sinkhorn_kl(self):
#         """Test KL divergence calculation"""
#         kls = sinkhorn_kl(self.perms)
        
#         for p_name in self.perms:
#             self.assertIn(p_name, kls)
#             self.assertGreaterEqual(kls[p_name], 0)  # KL divergence is non-negative


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

        # Expect IndexError due to mismatch in dimensions
        with self.assertRaises(IndexError):
            permute_param(perm_spec, perms, "weight", param)

    def test_get_permutation_with_fixed_points(self):
        n = 1000
        fixed_points_fraction = 0.2
        perm = get_random_permutation_with_fixed_points(n, fixed_points_fraction)
        fixeds = np.round(n * fixed_points_fraction)
        self.assertGreaterEqual((perm == np.arange(n)).sum(), fixeds)


if __name__ == "__main__":
    unittest.main()
