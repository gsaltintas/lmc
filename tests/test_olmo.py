import unittest

import numpy as np
import torch
from transformers import AutoTokenizer

from lmc.models.olmo import OLMo
from lmc.permutations.utils import permute_param

# salloc --cpus-per-gpu=8 --mem-per-cpu=8G --mem=64G --time=01:00:00 --gres=gpu:a100:1

# salloc  --cpus-per-task=8 --mem=72G --time=03:00:00 --gres:gpus:a100:1
# salloc  --cpus-per-task=8 --ntasks-per-node=1 --mem=48G -n 1 --time=03:00:00 --gpus-per-task=rtx8000:1
# salloc  --cpus-per-task=4 --ntasks-per-node=1 --mem=4G -n 1 --time=20:00 --gres=gpu:1


class TestOLMoPermutationFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        # cls.device = "cpu"
        cls.use_bfloat16 = False
        # cls.use_bfloat16 = True
        if cls.use_bfloat16:
            cls.rtol = 1e-2
            cls.atol = 1e-2
            cls.dtype = torch.bfloat16
        else:
            cls.rtol = 1e-5
            cls.atol = 1e-3
            cls.dtype = torch.float32

        # Initialize model
        cls.model = OLMo(
            # model_name="allenai/OLMo-1B",
            # model_name="allenai/OLMo-1B-hf",
            model_name="allenai/OLMo-2-1124-7B",
            output_dim=2,
            # task_type="classification",
            task_type="generation",
            initialization_strategy="pretrained",
            norm="layernorm",
        )
        print(cls.model)
        # exit(0)
        print(cls.model.permutation_spec())
        cls.tokenizer = cls.model.tokenizer
        cls.model.to(
            device=cls.device,
            dtype=cls.dtype,
            # device_map="auto",
            # max_memory={0: "35GB"},
        )
        print(f"Running on {cls.device}.")
        cls.model.eval()

    def generate_sample_input(self, batch_size=1, seq_length=1, tokenizer=None):
        """Generate sample input for OLMo model"""
        if tokenizer is None:
            tokenizer = self.tokenizer
        # Create random token IDs
        input_ids = torch.randint(
            low=0,
            high=tokenizer.vocab_size,
            size=(batch_size, seq_length),
            dtype=torch.long,
            device=self.device,
        )

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask

    def test_permutation_equivalenc1e(self):
        return
        perm_spec = self.model.permutation_spec()
        perms = self.model.get_random_permutation()
        model_ = self.model._permute(perms, inplace=False)
        model_.eval()

        layer_outputs = {}

        def get_activation(name):
            def hook(model, input, output):
                layer_outputs[name] = output

            return hook

        # Register hooks for both models
        for i, layer_ in enumerate((self.model.model.model.layers)):
            self.model.model.model.layers[i].self_attn.register_forward_hook(
                get_activation(f"attn1_{i}")
            )
            self.model.model.model.layers[i].mlp.register_forward_hook(
                get_activation(f"mlp1_{i}")
            )
            model_.model.model.layers[i].self_attn.register_forward_hook(
                get_activation(f"attn2_{i}")
            )
            model_.model.model.layers[i].mlp.register_forward_hook(
                get_activation(f"mlp2_{i}")
            )

        input_ids, attention_mask = self.generate_sample_input()

        with torch.no_grad():
            out1 = self.model(input_ids=input_ids, attention_mask=attention_mask)
            out2 = model_(input_ids=input_ids, attention_mask=attention_mask)

            # Compare activations with permutations
            for i in range(len(self.model.model.model.layers)):
                attn1 = layer_outputs[f"attn1_{i}"]
                attn2 = layer_outputs[f"attn2_{i}"]
                p_head = perms[f"P_head_{i}"]
                p_dhead = perms[f"P_dhead_{i}"]
                print(f"Layer {i} attention P_head_{i}:", perms[f"P_head_{i}"])
                # attn1_ = permute_param(perm_spec, perms, attn1, )
                perm = perms["P_0"]
                print(
                    f"Layer {i} attention match:",
                    torch.allclose(
                        attn1[0][:, :, perm], attn2[0], rtol=self.rtol, atol=self.atol
                    ),
                )

                mlp1 = layer_outputs[f"mlp1_{i}"]
                mlp2 = layer_outputs[f"mlp2_{i}"]
                print(f"Layer {i} MLP P_ff_{i}:", perms[f"P_ff_{i}"])
                print(
                    f"Layer {i} MLP match:",
                    torch.allclose(
                        mlp1[:, :, perm], mlp2, rtol=self.rtol, atol=self.atol
                    ),
                )

    def test_permutation_equivalence(self):
        input_ids = torch.randint(
            0, self.tokenizer.vocab_size, (1, 1), device=self.device
        )
        attention_mask = torch.ones_like(input_ids)

        # Full precision on A100
        self.model.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            # First forward pass
            output1 = self.model(input_ids=input_ids, attention_mask=attention_mask)

            perms = self.model.get_random_permutation()
            model_permuted = self.model._permute(perms, inplace=False)
            model_permuted.to(self.device, dtype=torch.float32)

            output2 = model_permuted(input_ids=input_ids, attention_mask=attention_mask)

            is_equal = torch.allclose(
                output1.logits, output2.logits, rtol=1e-5, atol=1e-5
            )
            if not is_equal:
                print(f"Max diff: {(output1.logits - output2.logits).abs().max()}")

            del model_permuted
            torch.cuda.empty_cache()

    def test_permutation_equivalenceold(self):
        return
        # return
        """Test that permuted model produces same output as original model"""
        # Get permutation spec
        perm_spec = self.model.permutation_spec()
        missing_keys = set(perm_spec.names_to_perms.keys()) - set(
            dict(self.model.model.named_parameters()).keys()
        )
        print("Missing", missing_keys)
        otherway_keys = set(dict(self.model.model.named_parameters()).keys()) - set(
            perm_spec.names_to_perms.keys()
        )
        print("Other", otherway_keys)

        # Generate sample input
        input_ids, attention_mask = self.generate_sample_input()
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            output = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        torch.cuda.empty_cache()

        # Generate random permutations
        perms = self.model.get_random_permutation()
        # Create permuted model
        model_ = self.model._permute(perms, inplace=False)
        # del self.model
        model_.eval()

        # Run both models
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            # output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            output_ = model_(input_ids=input_ids, attention_mask=attention_mask).logits
            x = output - output_
            print(x.abs().mean(), x.sum())
            # Check if outputs are equal
            self.assertTrue(
                torch.allclose(output, output_, rtol=1e-4, atol=1e-3),
                "Permuted model output differs from original model output",
            )

    def test_multiple_permutations(self):
        return
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
                        original_output, current_output, rtol=self.rtol, atol=self.atol
                    ),
                    f"Model output differs after {i + 1} permutations",
                )

    def test_permutation_different_inputs(self):
        return
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
                    torch.allclose(output, output_, rtol=self.rtol, atol=self.atol),
                    f"Outputs differ for sequence length {seq_length}",
                )

    def test_permutation_in_different_sized_OLMos(self):
        """Test permutations with different OLMo model sizes"""
        return

        model_name = "allenai/OLMo-7B"

        # Initialize model
        model = OLMo(
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
                torch.allclose(
                    output.logits, output_.logits, rtol=self.rtol, atol=self.atol
                ),
                "Permuted model output differs from original model output",
            )


if __name__ == "__main__":
    unittest.main()
    unittest.main()
