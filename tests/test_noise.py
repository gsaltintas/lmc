from collections import defaultdict
import unittest

import torch

from lmc.butterfly.butterfly import get_l2
from lmc.experiment_config import Trainer
from lmc.utils.seeds import seed_everything
from lmc.utils.setup_training import configure_model


class TestNoise(unittest.TestCase):
    def setUp(self):
        seed_everything(42)

    def _check_init_std(self, model_name, init_strategy):
        with self.subTest(f"{model_name} {init_strategy}"):
            # sample a lot of models
            params = defaultdict(list)
            l2 = []
            config = Trainer.from_dict(
                dict(
                    n_models=1,
                    initialization_strategy=init_strategy,
                    training_steps="1ep",
                    model_name=model_name,
                    dataset="mnist",
                    norm="layernorm",
                )
            )
            config.model.norm="layernorm"  #TODO hack as from_dict doesn't set this
            for _ in range(100):
                model = configure_model(config, device="cpu", print_output=False)
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
            scales = model.get_init_stds()
            for k, v in params.items():
                params = torch.cat(v)
                self.assertAlmostEqual(scales[k], torch.std(params).item(), places=2)
                self.assertAlmostEqual(torch.mean(params).item(), 0, places=2)

            # check that model L2 is close to stdev L2
            total = model.get_total_init_std()
            self.assertAlmostEqual(torch.mean(torch.tensor(l2)).item(), total, places=1)

    def test_normalization_constants(self):
        self._check_init_std("mlp/128x3", "kaiming_normal")
        self._check_init_std("resnet20-8", "kaiming_uniform")
        self._check_init_std("resnet20-16", "kaiming_normal")

    def test_batch_noise(self):
        pass  # TODO


if __name__ == "__main__":
    unittest.main()
