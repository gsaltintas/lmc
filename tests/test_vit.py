import unittest

import numpy as np
import torch

from lmc.models.vit import VIT

device = "cuda" if torch.cuda.is_available() else "cpu"


class TestVitPermutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Initialize model
        cls.model = VIT(
            model_name="google/vit-base-patch16-224",
            output_dim=10,
            initialization_strategy="pretrained",
            norm="layernorm",
        )
        print(cls.model.permutation_spec())

        # Convert to double precision for numerical stability
        cls.model = cls.model.to(torch.float64)
        cls.model.eval()
        cls.model.to(device)

    def generate_sample_input(self, batch_size=2):
        """Generate sample input for the ViT model"""
        x = torch.randn((batch_size, 3, 224, 224), device=device)
        return x

    def test_permutation_equivalence(self):
        """Test that permuted model produces same output as original model"""
        # Get permutation spec
        perm_spec = self.model.permutation_spec()

        # Generate random permutations
        perms = self.model.get_random_permutation()

        # Create permuted model
        model_ = self.model._permute(perms, inplace=False)
        model_.eval()

        # Generate sample input
        x = self.generate_sample_input()

        # Run both models
        with torch.no_grad():
            output = self.model(x)
            output_ = model_(x)

            # Check if outputs are equal
            self.assertTrue(
                torch.allclose(output, output_, rtol=1e-5, atol=1e-5),
                "Permuted model output differs from original model output",
            )

    def test_multiple_permutations(self):
        """Test that multiple successive permutations maintain equivalence"""
        # Get permutation spec
        perm_spec = self.model.permutation_spec()

        # Generate input once
        x = self.generate_sample_input()

        # Original output
        with torch.no_grad():
            original_output = self.model(x)

        # Apply multiple permutations
        current_model = self.model
        for i in range(3):  # Test 3 successive permutations
            # Generate new random permutations
            new_perm = current_model.get_random_permutation()

            # Create new permuted model
            current_model = current_model._permute(new_perm, inplace=False)
            current_model = current_model.to(torch.float64)
            current_model.eval()

            # Check output
            with torch.no_grad():
                current_output = current_model(x)

                self.assertTrue(
                    torch.allclose(
                        original_output, current_output, rtol=1e-5, atol=1e-5
                    ),
                    f"Model output differs after {i + 1} permutations",
                )


if __name__ == "__main__":
    unittest.main()
