import unittest

import torch

from lmc.experiment.perturb import PerturbedTrainingRunner
from lmc.experiment_config import PerturbedTrainer
from lmc.utils.seeds import seed_everything
from lmc.utils.setup_training import configure_model
from tests.base import BaseTest


class TestModelInit(BaseTest):

    def _params_equal(self, model_1, model_2):
        sd_1 = model_1.state_dict()
        sd_2 = model_2.state_dict()
        if set(sd_1.keys()) != set(sd_2.keys()):
            return False
        for k, v in sd_1.items():
            if not torch.allclose(v, sd_2[k]):
                return False
        return True

    def test_model_init(self, **kwargs):
        # check that experiment 
        config = PerturbedTrainer.from_dict(
            dict(
                n_models=2,
                training_steps="1ep",
                model_name="resnet20-8",
                dataset="cifar10",
                log_dir=self.log_dir,
                path=self.data_dir / "cifar10",
                norm="layernorm",
                seed1=11,
                seed2=22,
            )
        )
        config.model.norm="layernorm"  #TODO hack as from_dict doesn't set this, see tests in test_config.py
        exp_1 = PerturbedTrainingRunner(config)
        exp_2 = PerturbedTrainingRunner(config)
        exp_1.setup()
        exp_2.setup()
        self.assertTrue(self._params_equal(exp_1.training_elements[0].model, exp_2.training_elements[0].model))
        self.assertTrue(self._params_equal(exp_1.training_elements[1].model, exp_2.training_elements[1].model))
        self.assertFalse(self._params_equal(exp_1.training_elements[0].model, exp_1.training_elements[1].model))
        seed_everything(exp_1.config.seeds.seed1)
        model = configure_model(config, device=exp_1.device, seed=exp_1.config.seeds.seed1, print_output=False)
        self.assertTrue(self._params_equal(exp_1.training_elements[0].model, model))
        seed_everything(exp_1.config.seeds.seed2)
        model = configure_model(config, device=exp_1.device, seed=exp_1.config.seeds.seed2, print_output=False)
        self.assertTrue(self._params_equal(exp_1.training_elements[1].model, model))


if __name__ == "__main__":
    unittest.main()
