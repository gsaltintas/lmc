from copy import deepcopy
from collections import defaultdict
import unittest

import torch

from lmc.butterfly.butterfly import get_l2, get_all_init_l2s, get_perturbed_layers, sample_noise_and_perturb
from lmc.experiment_config import PerturbedTrainer
from lmc.utils.seeds import seed_everything
from lmc.utils.setup_training import configure_model
from tests.base import BaseTest


class TestNoise(BaseTest):
    def setUp(self):
        super().setUp()
        self.model, _ = configure_model(self._get_config("resnet20-32", "kaiming_normal"), device="cpu", seed=41)
        expected_l2, expected_l2_per_layer = get_all_init_l2s(self.model)
        self.expected_l2 = expected_l2
        self.expected_l2_per_layer = expected_l2_per_layer

    def _get_config(self, model_name, init_strategy, perturb_scale=1, mode="batch", normalize_perturb=True, scale_to_init_if_normalized=True, dont_perturb_module_patterns=[]):
        config = PerturbedTrainer.from_dict(
            dict(
                n_models=1,
                initialization_strategy=init_strategy,
                training_steps="1ep",
                model_name=model_name,
                dataset="cifar10",
                path=self.data_dir / "cifar10",
                norm="layernorm",
                perturb_scale=perturb_scale,
                perturb_mode=mode,
                normalize_perturb=normalize_perturb,
                scale_to_init_if_normalized=scale_to_init_if_normalized,
                dont_perturb_module_patterns=dont_perturb_module_patterns,
            )
        )
        config.model.norm="layernorm"  #TODO hack as from_dict doesn't set this, see tests in test_config.py
        return config

    def _get_noise(self, **kwargs):
        model = deepcopy(self.model)
        sd = self.model.state_dict()
        config = self._get_config("resnet20-32", "kaiming_normal", **kwargs)
        log_dict = sample_noise_and_perturb(config, model, perturb_seed=42, loss_fn=None)
        perturbed_sd = model.state_dict()
        per_layer_l2 = {}
        total_sqsum = 0
        for k, v in perturbed_sd.items():
            sqdiff = torch.sum((v - sd[k])**2).item()
            per_layer_l2[k] = sqdiff**0.5
            total_sqsum += sqdiff
        return total_sqsum**0.5, per_layer_l2

    def _check_init_std(self, model_name, init_strategy):
        config = self._get_config(model_name, init_strategy)
        with self.subTest(f"{model_name} {init_strategy}"):
            # sample a lot of models
            params = defaultdict(list)
            l2 = []
            N_SAMPLES = 300
            for i in range(N_SAMPLES):
                model, _ = configure_model(config, device="cpu", seed=i, print_output=False)
                sqsum = 0
                norm_sqsum = 0
                for k, v in model.state_dict().items():
                    if "norm" in k:
                        norm_sqsum += torch.sum(v**2).item()
                    else:
                        sqsum += torch.sum(v**2).item()
                        params[k].append(v)
                verify_sqsum = get_l2(model.state_dict())**2
                self.assertAlmostEqual(sqsum + norm_sqsum, verify_sqsum.item(), places=1)
                l2.append(sqsum**0.5)

            # check that random init is close in standard deviation to the normalization constsants
            total, per_layer = get_all_init_l2s(model)
            scales = model.get_init_stds()
            for k, v in params.items():
                params = torch.stack(v, dim=0).reshape(N_SAMPLES, -1)
                self.assertAlmostEqual(scales[k], torch.std(params).item(), places=2)
                self.assertAlmostEqual(per_layer[k], torch.mean(torch.linalg.norm(params, dim=-1)).item(), places=1)
                self.assertAlmostEqual(torch.mean(params).item(), 0, places=2)

            # check that model L2 is close to stdev L2
            self.assertAlmostEqual(torch.mean(torch.tensor(l2)).item(), total, places=1)

    def test_normalization_constants(self):
        self._check_init_std("mlp/128x3", "kaiming_normal")
        self._check_init_std("resnet20-8", "kaiming_uniform")
        self._check_init_std("resnet20-16", "kaiming_normal")

    def test_perturbed_layers(self):
        for n_unperturbed, dont_perturb in [(0, []), (43, [".*norm.*|.*bias.*"]), (23, ['^((?!norm).)*$']), (64, ['^((?!model.fc.weight).)*$'])]:
            with self.subTest(dont_perturb):
                layers = get_perturbed_layers(self.model, dont_perturb)
                self.assertEqual(len(layers), 65 - n_unperturbed)

    def test_get_noise(self):
        def assert_noise_not_zero(per_layer_l2s, dont_perturb, mode, scale):
            for k, v in per_layer_l2s.items():
                if ("norm" in k or "fc.bias" in k) and len(dont_perturb) > 0:
                    self.assertEqual(v, 0)
                else:  # with norm layers every layer should be perturbed
                    self.assertGreater(v, 0)
                # without norm layers we expect gaussian noise to be distributed identically to init
                if mode == "gaussian" and len(dont_perturb) > 0:
                    self.assertAlmostEqual(v, self.expected_l2_per_layer[k] * scale, places=0)

        for dont_perturb in [[], [".*norm.*|.*bias.*"]]:
            for mode in ["batch", "gaussian"]:
                with self.subTest(f"{mode} normalized only {dont_perturb}"):
                    noise_l2, per_layer_l2 = self._get_noise(mode=mode, normalize_perturb=True, scale_to_init_if_normalized=False, dont_perturb_module_patterns=dont_perturb)
                    self.assertAlmostEqual(noise_l2, 1, places=1)
                    assert_noise_not_zero(per_layer_l2, dont_perturb, mode, 1 / self.expected_l2)
                with self.subTest(f"{mode} normalized and rescaled {dont_perturb}"):
                    noise_l2, per_layer_l2 = self._get_noise(mode=mode, perturb_scale=2, normalize_perturb=True, scale_to_init_if_normalized=True, dont_perturb_module_patterns=dont_perturb)
                    self.assertAlmostEqual(noise_l2, self.expected_l2 * 2, places=1)
                    assert_noise_not_zero(per_layer_l2, dont_perturb, mode, 2)
        with self.subTest(f"gaussian unnormalized all layers"):
            noise_l2, per_layer_l2 = self._get_noise(mode="gaussian", normalize_perturb=False, dont_perturb_module_patterns=[])
            self.assertAlmostEqual(noise_l2, self.expected_l2, places=0)
        with self.subTest(f"gaussian unnormalized no norm layers"):
            noise_l2, per_layer_l2 = self._get_noise(mode="gaussian", normalize_perturb=False, dont_perturb_module_patterns=[".*norm.*|.*bias.*"])
            self.assertAlmostEqual(noise_l2, self.expected_l2, places=0)
            assert_noise_not_zero(per_layer_l2, [".*norm.*", "model.fc.bias"], "gaussian", 1)
        # these values are recorded for regression testing purposes
        with self.subTest(f"batch unnormalized all layers"):
            noise_l2, per_layer_l2 = self._get_noise(mode="batch", normalize_perturb=False, dont_perturb_module_patterns=[])
            self.assertAlmostEqual(noise_l2, 31.71325871531403)
        with self.subTest(f"batch unnormalized no norm layers"):
            noise_l2, per_layer_l2 = self._get_noise(mode="batch", normalize_perturb=False, dont_perturb_module_patterns=[".*norm.*", "model.fc.bias"])
            self.assertAlmostEqual(noise_l2, 31.608178063272423)


if __name__ == "__main__":
    unittest.main()
