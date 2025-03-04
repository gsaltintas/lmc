from copy import deepcopy
from collections import defaultdict
import unittest

import torch
import torch.nn as nn

from lmc.butterfly.butterfly import (
    get_all_init_l2s,
    get_perturbed_layers,
    sample_noise_and_perturb,
    scale_noise,
)
from lmc.utils.training_element import params_l2
from lmc.models.bert import Bert
from lmc.utils.setup_training import configure_model
from tests.base import BaseTest


class TestNoise(BaseTest):
    def setUp(self):
        super().setUp()
        self.model, _ = configure_model(
            self.get_test_config(
                model_name="resnet20-32", initialization_strategy="kaiming_normal"
            ),
            device="cpu",
            seed=41,
        )
        layers = [k for k, _ in self.model.named_parameters()]
        non_constant_layers = [k for k in layers if "conv" in k]
        self.expected_l2_all, self.expected_l2_per_layer = get_all_init_l2s(
            self.model, layers
        )
        self.expected_l2, _ = get_all_init_l2s(self.model, non_constant_layers)

    def test_scale_noise(self):
        class DummyModel(nn.Module):
            def __init__(self, weight_scale, bias_scale):
                super().__init__()
                self.weight_scale = weight_scale
                self.bias_scale = bias_scale
                self.param = nn.Linear(10, 1)

            def get_init_stds(self, include_constant_params):
                return {
                    "param.weight": self.weight_scale,
                    "param.bias": self.bias_scale if include_constant_params else 0,
                }

        def make_param_dict(scale):
            return {
                "param.weight": torch.arange(-5, 5, dtype=torch.float).reshape(1, 10)
                * scale,
                "param.bias": torch.tensor([2], dtype=float) * scale,
            }

        model = DummyModel(1, 1)
        noise_dict = make_param_dict(1)
        layers = list(noise_dict.keys())
        expected_l2, _ = get_all_init_l2s(model, layers)
        self.assertEqual(expected_l2, 11**0.5)
        actual_norm = params_l2(noise_dict.values())
        self.assertEqual(actual_norm, 89**0.5)

        no_scale = scale_noise(noise_dict, model, layers, 1, False, False)
        self.assertAlmostEqual(params_l2(no_scale.values()), actual_norm, places=6)
        self.assertTrue(self.state_dicts_equal(no_scale, make_param_dict(1)))

        normalized = scale_noise(
            noise_dict, model, ["param.weight", "param.bias"], 1, True, False
        )
        self.assertAlmostEqual(params_l2(normalized.values()), 1, places=6)
        self.assertTrue(
            self.state_dicts_equal(normalized, make_param_dict(1 / actual_norm))
        )

        normalized = scale_noise(
            noise_dict, model, ["param.weight", "param.bias"], 1, True, True
        )
        self.assertAlmostEqual(params_l2(normalized.values()), expected_l2, places=6)
        self.assertTrue(
            self.state_dicts_equal(
                normalized, make_param_dict(expected_l2 / actual_norm)
            )
        )

    def _check_init_std(self, name, model_iterator):
        with self.subTest(name):
            # sample a lot of models
            params = defaultdict(list)
            l2 = []
            for n, model in enumerate(model_iterator):
                sqsum = 0
                norm_sqsum = 0
                for k, v in model.state_dict().items():
                    if "norm" in k or "Norm" in k:
                        norm_sqsum += torch.sum(v**2).item()
                    else:
                        sqsum += torch.sum(v**2).item()
                        params[k].append(v)
                l2.append(sqsum**0.5)

            # check that random init is close in standard deviation to the normalization constants
            scales = model.get_init_stds()
            layers = [k for k, v in scales.items() if v > 0]
            total, per_layer = get_all_init_l2s(model, layers)
            for k, v in params.items():
                params = torch.stack(v, dim=0).reshape(len(v), -1)
                self.assertAlmostEqual(scales[k], torch.std(params).item(), places=2)
                self.assertAlmostEqual(
                    per_layer[k],
                    torch.mean(torch.linalg.norm(params, dim=-1)).item(),
                    places=0,
                )
                self.assertAlmostEqual(torch.mean(params).item(), 0, places=2)

            # check that model L2 is close to stdev L2
            self.assertAlmostEqual(torch.mean(torch.tensor(l2)).item(), total, places=1)

    def _yield_vision_models(self, model_name, init_strategy):
        config = self.get_test_config(
            experiment="train",
            model_name=model_name,
            initialization_strategy=init_strategy,
        )
        for i in range(200):
            model, _ = configure_model(config, device="cpu", seed=i, print_output=False)
            yield model

    def _yield_multibert_models(self):
        for i in range(5):
            model = Bert(model_name=f"google/multiberts-seed_{i}-step_0k", output_dim=2)
            yield model

    def test_normalization_constants(self):
        self._check_init_std("multibert", self._yield_multibert_models())
        self._check_init_std(
            "mlp normal", self._yield_vision_models("mlp/128x3", "kaiming_normal")
        )
        self._check_init_std(
            "resnet uniform", self._yield_vision_models("resnet20-8", "kaiming_uniform")
        )
        self._check_init_std(
            "resnet normal", self._yield_vision_models("resnet20-16", "kaiming_normal")
        )

    def test_perturbed_layers(self):
        for n_unperturbed, dont_perturb in [
            (0, []),
            (43, [".*norm.*|.*bias.*"]),
            (23, ["^((?!norm).)*$"]),
            (64, ["^((?!model.fc.weight).)*$"]),
            (64, ["^((?!model.conv.weight).)*$"]),
        ]:
            with self.subTest(dont_perturb):
                layers = get_perturbed_layers(self.model, dont_perturb)
                self.assertEqual(len(layers), 65 - n_unperturbed)

    def _get_noise(
        self,
        perturb_scale=1,
        mode="batch",
        normalize_perturb=True,
        scale_to_init_if_normalized=True,
        dont_perturb_module_patterns=[],
        perturb_fraction=1,
    ):
        model = deepcopy(self.model)
        sd = self.model.state_dict()
        config = self.get_test_config(
            model_name="resnet20-32",
            initialization_strategy="kaiming_normal",
            dataset="cifar10",
            perturb_scale=perturb_scale,
            perturb_mode=mode,
            normalize_perturb=normalize_perturb,
            scale_to_init_if_normalized=scale_to_init_if_normalized,
            dont_perturb_module_patterns=dont_perturb_module_patterns,
            perturb_fraction=perturb_fraction,
        )
        config.perturb_fraction = perturb_fraction
        _ = sample_noise_and_perturb(
            config, model, perturb_seed=42, loss_fn=None, ind=0
        )
        perturbed_sd = model.state_dict()
        per_layer_l2 = {}
        total_sqsum = 0
        for k, v in perturbed_sd.items():
            sqdiff = torch.sum((v - sd[k]) ** 2).item()
            per_layer_l2[k] = sqdiff**0.5
            total_sqsum += sqdiff
        return total_sqsum**0.5, per_layer_l2

    def test_get_noise(self):
        def assert_noise_not_zero(per_layer_l2s, dont_perturb, mode, scale):
            for k, v in per_layer_l2s.items():
                if ("norm" in k or "fc.bias" in k) and len(dont_perturb) > 0:
                    self.assertEqual(v, 0)
                # without norm layers we expect gaussian noise to be distributed identically to init
                elif mode == "gaussian" and len(dont_perturb) > 0:
                    self.assertAlmostEqual(
                        v, self.expected_l2_per_layer[k] * scale, places=0
                    )
                else:  # with norm layers every layer should be perturbed
                    self.assertGreater(v, 0)

        for dont_perturb, expected_l2 in [
            ([], self.expected_l2_all),
            ([".*norm.*|.*bias.*"], self.expected_l2),
        ]:
            for mode in ["batch", "gaussian"]:
                with self.subTest(f"{mode} normalized only {dont_perturb}"):
                    noise_l2, per_layer_l2 = self._get_noise(
                        mode=mode,
                        normalize_perturb=True,
                        scale_to_init_if_normalized=False,
                        dont_perturb_module_patterns=dont_perturb,
                    )
                    self.assertAlmostEqual(noise_l2, 1, places=0)
                    assert_noise_not_zero(
                        per_layer_l2, dont_perturb, mode, 1 / expected_l2
                    )
                with self.subTest(f"{mode} normalized and rescaled {dont_perturb}"):
                    noise_l2, per_layer_l2 = self._get_noise(
                        mode=mode,
                        perturb_scale=0.5,
                        normalize_perturb=True,
                        scale_to_init_if_normalized=True,
                        dont_perturb_module_patterns=dont_perturb,
                    )
                    self.assertAlmostEqual(noise_l2, expected_l2 * 0.5, places=0)
                    assert_noise_not_zero(per_layer_l2, dont_perturb, mode, 0.5)
        with self.subTest("gaussian unnormalized all layers"):
            noise_l2, per_layer_l2 = self._get_noise(
                mode="gaussian",
                normalize_perturb=False,
                dont_perturb_module_patterns=[],
            )
            self.assertAlmostEqual(noise_l2, self.expected_l2_all, places=0)
        with self.subTest("gaussian unnormalized no norm layers"):
            noise_l2, per_layer_l2 = self._get_noise(
                mode="gaussian",
                normalize_perturb=False,
                dont_perturb_module_patterns=[".*norm.*|.*bias.*"],
            )
            self.assertAlmostEqual(noise_l2, self.expected_l2, places=0)
            assert_noise_not_zero(
                per_layer_l2, [".*norm.*", "model.fc.bias"], "gaussian", 1
            )
        # these values are recorded for regression testing purposes
        with self.subTest("batch unnormalized all layers"):
            noise_l2, per_layer_l2 = self._get_noise(
                mode="batch", normalize_perturb=False, dont_perturb_module_patterns=[]
            )
            self.assertAlmostEqual(noise_l2, 31.71325871531403)
        with self.subTest("batch unnormalized no norm layers"):
            noise_l2, per_layer_l2 = self._get_noise(
                mode="batch",
                normalize_perturb=False,
                dont_perturb_module_patterns=[".*norm.*", "model.fc.bias"],
            )
            self.assertAlmostEqual(noise_l2, 31.608178063272423)
        with self.subTest("perturb fraction"):
            noise_l2, per_layer_l2 = self._get_noise(
                mode="batch",
                normalize_perturb=False,
                dont_perturb_module_patterns=[".*norm.*", "model.fc.bias"],
                perturb_fraction=0.5
            )
            self.assertAlmostEqual(noise_l2, 22.37011788592581)


if __name__ == "__main__":
    unittest.main()
