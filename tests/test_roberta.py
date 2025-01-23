import unittest

import numpy as np
import torch
from transformers import RobertaTokenizer

from lmc.models.roberta import Roberta


class TestRobertaPermutationFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Initialize model
        cls.model = Roberta(
            model_name="roberta-base",
            output_dim=2,
            initialization_strategy="pretrained",
            norm="layernorm",
        )
        print(cls.model)
        cls.tokenizer = cls.model.tokenizer

        # Convert to double precision for numerical stability
        cls.model = cls.model.to(torch.float64)
        cls.model.eval()

    def generate_sample_input(self, batch_size=4, seq_length=128, tokenizer=None):
        """Generate sample input for RoBERTa model"""
        if tokenizer is None:
            tokenizer = self.tokenizer
        # Create random token IDs
        input_ids = torch.randint(
            low=0,
            high=tokenizer.vocab_size,
            size=(batch_size, seq_length),
            dtype=torch.long,
        )

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask

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
        input_ids, attention_mask = self.generate_sample_input()

        # Run both models
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            output_ = model_(input_ids=input_ids, attention_mask=attention_mask)

            # Check if outputs are equal
            self.assertTrue(
                torch.allclose(output.logits, output_.logits, rtol=1e-5, atol=1e-5),
                "Permuted model output differs from original model output",
            )

    def test_multiple_permutations(self):
        """Test that multiple successive permutations maintain equivalence"""
        # Get permutation spec
        perm_spec = self.model.permutation_spec()

        # Generate input once
        input_ids, attention_mask = self.generate_sample_input()

        # Original output
        with torch.no_grad():
            original_output = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

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
                current_output = current_model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

                self.assertTrue(
                    torch.allclose(
                        original_output, current_output, rtol=1e-5, atol=1e-5
                    ),
                    f"Model output differs after {i + 1} permutations",
                )

    def test_permutation_different_inputs(self):
        """Test permutation equivalence across different input lengths"""
        # Get permutation spec and generate permutations
        perm_spec = self.model.permutation_spec()
        perms = self.model.get_random_permutation()

        # Create permuted model
        model_ = self.model._permute(perms, inplace=False)
        model_.eval()

        # Test different sequence lengths
        for seq_length in [32, 64, 128]:
            input_ids, attention_mask = self.generate_sample_input(
                batch_size=2, seq_length=seq_length
            )

            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                output_ = model_(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

                self.assertTrue(
                    torch.allclose(output, output_, rtol=1e-5, atol=1e-5),
                    f"Outputs differ for sequence length {seq_length}",
                )

    def test_permutation_in_different_sized_robertas(self):
        """Test permutations with different RoBERTa model sizes"""

        model_name = "roberta-large"

        # Initialize model
        model = Roberta(
            model_name=model_name,
            output_dim=2,
            initialization_strategy="pretrained",
            norm="layernorm",
        )
        tokenizer = model.tokenizer

        # Convert to double precision for numerical stability
        model = model.to(torch.float64)
        model.eval()

        # Get permutation spec
        perm_spec = model.permutation_spec()

        # Generate random permutations
        perms = model.get_random_permutation()

        # Create permuted model
        model_ = model._permute(perms, inplace=False)
        model_.eval()

        # Generate sample input
        input_ids, attention_mask = self.generate_sample_input(tokenizer=tokenizer)

        # Run both models
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            output_ = model_(input_ids=input_ids, attention_mask=attention_mask)

            # Check if outputs are equal
            self.assertTrue(
                torch.allclose(output.logits, output_.logits, rtol=1e-5, atol=1e-5),
                "Permuted model output differs from original model output",
            )


if __name__ == "__main__":
    unittest.main()
